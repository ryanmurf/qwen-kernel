use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, mpsc as tokio_mpsc};

use crate::ffi::{Engine, QkConfig, StepOut};

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Cmd>,
    pub n_ctx: u32,
    pub n_slots: u32,
    pub chunk: u32,
    pub n_vocab: u32,
}

pub struct EngineThread {
    handle: EngineHandle,
    join: Option<JoinHandle<()>>,
}

pub struct Job {
    pub prompt_ids: Vec<u32>,
    pub max_gen: u32,
    // History-boundary token count for the cross-turn KV snapshot (0 = disabled).
    pub snap_prefix: u32,
    pub sampling: Sampling,
    pub events: tokio_mpsc::Sender<SlotEvent>,
    pub permit: OwnedSemaphorePermit,
}

/// Per-request sampling policy. `temp <= 0` means greedy — the default, and
/// bit-identical to the pre-sampling engine/wire behavior (token gates rely
/// on this). Sampling itself runs head-side in the split driver (split.rs);
/// the single-box engine path is greedy-only and logs when asked otherwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temp: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Sampling {
    pub const GREEDY: Sampling = Sampling {
        temp: 0.0,
        top_p: 1.0,
        seed: 0,
    };

    pub fn is_greedy(&self) -> bool {
        self.temp <= 0.0
    }
}

pub enum Cmd {
    Submit(Job),
    Shutdown,
}

#[derive(Clone, Debug)]
pub enum SlotEvent {
    Tokens(Vec<u32>),
    Done { reason: FinishReason },
    Error(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Eos,
    Limit,
}

// Shared with the split-mode driver (split.rs), which reproduces the exact
// same event/backpressure semantics over a different forward path.
pub(crate) struct SlotState {
    pub(crate) events: tokio_mpsc::Sender<SlotEvent>,
    pub(crate) emitted: u32,
    pub(crate) max_gen: u32,
    // Generation is over (EOS/limit); the slot lingers only to drain `pending`.
    pub(crate) finished: bool,
    // Events the engine produced but the consumer has not yet accepted. The
    // engine thread is shared across all slots, so it must never block on a
    // single slow/stalled SSE consumer: a full channel would freeze every
    // other slot too (head-of-line block). Instead we buffer here and retry
    // with `try_send` each tick. Growth is bounded by `max_gen` tokens, and a
    // consumer that closes its channel drops the slot outright.
    pub(crate) pending: VecDeque<SlotEvent>,
    _permit: OwnedSemaphorePermit,
}

impl SlotState {
    pub(crate) fn new(
        events: tokio_mpsc::Sender<SlotEvent>,
        max_gen: u32,
        permit: OwnedSemaphorePermit,
    ) -> Self {
        Self {
            events,
            emitted: 0,
            max_gen,
            finished: false,
            pending: VecDeque::new(),
            _permit: permit,
        }
    }
}

// Outcome of trying to hand a slot's buffered events to its consumer.
pub(crate) enum FlushOutcome {
    // Consumer is keeping up (or merely backed up); keep the slot around.
    Live,
    // Nothing left to deliver and generation is finished: release the slot.
    Drained,
    // Consumer closed the channel: abandon the slot immediately.
    Closed,
}

// Push buffered events to the consumer without ever blocking the engine
// thread. On a full channel we stop and try again next tick.
pub(crate) fn flush_slot(state: &mut SlotState) -> FlushOutcome {
    use tokio_mpsc::error::TrySendError;
    while let Some(ev) = state.pending.pop_front() {
        match state.events.try_send(ev) {
            Ok(()) => {}
            Err(TrySendError::Full(ev)) => {
                state.pending.push_front(ev);
                return FlushOutcome::Live;
            }
            Err(TrySendError::Closed(_)) => return FlushOutcome::Closed,
        }
    }
    if state.finished {
        FlushOutcome::Drained
    } else {
        FlushOutcome::Live
    }
}

impl EngineThread {
    /// `split_next`: split-serving worker address (docs/split-serving.md).
    /// When set, this process must hold the HEAD stage (`QK_LAYERS=0:S`) and
    /// generation is driven by the split driver instead of qk_step_chunk.
    pub fn start(
        lib: &Path,
        model: &Path,
        cfg: QkConfig,
        split_next: Option<String>,
    ) -> anyhow::Result<Self> {
        use anyhow::{Context, bail};

        // Open the engine exactly once (the model is ~16 GB of VRAM); read the
        // resolved config off it, then move it into the engine thread. Opening a
        // throwaway "probe" engine here would transiently double VRAM use.
        let engine = Engine::open(lib, model, cfg)?;
        let n_ctx = engine.n_ctx();
        let n_slots = engine.n_slots();
        let chunk = engine.chunk();
        let n_vocab = engine.n_vocab();

        let split = match split_next {
            Some(addr) => {
                let info = engine
                    .stage_info()
                    .context("--split-next requires an engine library with the stage ABI")?;
                if !info.is_first() || info.is_last() {
                    bail!(
                        "--split-next: engine must be the head stage (QK_LAYERS=0:S, S < {}), got layers [{}, {})",
                        info.n_layer,
                        info.first,
                        info.end
                    );
                }
                for var in ["QK_FORK", "QK_SPEC"] {
                    if std::env::var_os(var).is_some() {
                        tracing::warn!("{var} is unsupported in split mode v1 and will not engage");
                    }
                }
                tracing::info!(
                    "split serving: head layers [0, {}) local, layers [{}, {}) via {}",
                    info.end,
                    info.end,
                    info.n_layer,
                    addr
                );
                Some(crate::split::WorkerLink::new(
                    addr,
                    crate::split::WorkerExpect {
                        layer_first: info.end,
                        n_layer: info.n_layer,
                        n_embd: info.n_embd,
                        n_slots,
                        n_ctx,
                    },
                ))
            }
            None => {
                if let Some(info) = engine.stage_info() {
                    if info.is_split() {
                        bail!(
                            "engine owns layers [{}, {}) (QK_LAYERS) but --split-next was not given",
                            info.first,
                            info.end
                        );
                    }
                }
                None
            }
        };

        let (tx, rx) = mpsc::channel();
        let join = thread::Builder::new()
            .name("qk-engine".to_owned())
            .spawn(move || match split {
                Some(link) => crate::split::run_split_engine_thread(engine, link, rx),
                None => run_engine_thread(engine, rx),
            })
            .map_err(anyhow::Error::from)?;
        Ok(Self {
            handle: EngineHandle {
                tx,
                n_ctx,
                n_slots,
                chunk,
                n_vocab,
            },
            join: Some(join),
        })
    }

    pub fn handle(&self) -> EngineHandle {
        self.handle.clone()
    }

    pub fn shutdown(mut self) {
        let _ = self.handle.tx.send(Cmd::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl EngineHandle {
    pub fn submit(&self, job: Job) -> Result<(), mpsc::SendError<Cmd>> {
        self.tx.send(Cmd::Submit(job))
    }
}

fn run_engine_thread(mut engine: Engine, rx: mpsc::Receiver<Cmd>) {
    let n_slots = engine.n_slots();
    let chunk = engine.chunk();
    let mut out = StepOut::new(n_slots, chunk);
    let mut queue = VecDeque::<Job>::new();
    let mut slots = (0..n_slots)
        .map(|_| None)
        .collect::<Vec<Option<SlotState>>>();

    loop {
        // Only park on `recv` when the engine is completely idle. A slot that
        // is finished but still draining buffered events keeps us awake so we
        // can retry its consumer, but must not block admission of new work.
        if slots.iter().all(Option::is_none) && queue.is_empty() {
            match rx.recv() {
                Ok(Cmd::Submit(job)) => queue.push_back(job),
                Ok(Cmd::Shutdown) | Err(_) => break,
            }
        }
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                Cmd::Submit(job) => queue.push_back(job),
                Cmd::Shutdown => return,
            }
        }

        admit_jobs(&mut engine, &mut queue, &mut slots);

        // Retry delivery for slots whose consumer was backed up last tick, and
        // find out whether any slot still needs GPU work.
        let mut any_active = false;
        for (slot, state) in slots.iter_mut().enumerate() {
            let Some(st) = state.as_mut() else { continue };
            if st.finished {
                match flush_slot(st) {
                    FlushOutcome::Live => {}
                    FlushOutcome::Drained | FlushOutcome::Closed => *state = None,
                }
            } else {
                any_active = true;
                // Opportunistically drain any backlog before the next step.
                if let FlushOutcome::Closed = flush_slot(st) {
                    engine.slot_cancel(slot as u32);
                    *state = None;
                }
            }
        }

        if any_active {
            match engine.step_chunk(&mut out) {
                Ok(_) => handle_step(&mut engine, &out, &mut slots),
                Err(err) => {
                    let message = err.to_string();
                    for (slot, state) in slots.iter_mut().enumerate() {
                        if let Some(st) = state.as_mut() {
                            if st.finished {
                                continue; // already generated; let it drain
                            }
                            let _ = st.events.try_send(SlotEvent::Error(message.clone()));
                            engine.slot_cancel(slot as u32);
                            *state = None;
                        }
                    }
                }
            }
        } else if slots.iter().any(Option::is_some) {
            // Only drain-pending slots remain (a stalled consumer). Don't
            // busy-spin while we wait for it to accept buffered events.
            thread::sleep(Duration::from_millis(2));
        }
    }
}

fn admit_jobs(engine: &mut Engine, queue: &mut VecDeque<Job>, slots: &mut [Option<SlotState>]) {
    for (slot, state) in slots.iter_mut().enumerate() {
        if state.is_some() {
            continue;
        }
        let Some(job) = queue.pop_front() else {
            break;
        };
        let events = job.events.clone();
        if !job.sampling.is_greedy() {
            // qk_step_chunk samples inside the engine's fused argmax chain;
            // only the split driver can inject a sampled token. Serve greedy
            // rather than fail — the request is still well-formed.
            tracing::warn!(
                "temperature={} requested but the single-box engine path is greedy-only; serving greedy",
                job.sampling.temp
            );
        }
        match engine.slot_start(slot as u32, &job.prompt_ids, job.max_gen, job.snap_prefix) {
            Ok(()) => {
                *state = Some(SlotState {
                    events,
                    emitted: 0,
                    max_gen: job.max_gen,
                    finished: false,
                    pending: VecDeque::new(),
                    _permit: job.permit,
                });
            }
            Err(err) => {
                let _ = events.try_send(SlotEvent::Error(err.to_string()));
            }
        }
    }
}

fn handle_step(engine: &mut Engine, out: &StepOut, slots: &mut [Option<SlotState>]) {
    for (slot, state) in slots.iter_mut().enumerate() {
        let Some(slot_state) = state.as_mut() else {
            continue;
        };
        if slot_state.finished {
            continue; // draining only; not stepped this tick
        }
        let tokens = out.slot_tokens(slot as u32);
        if !tokens.is_empty() {
            slot_state.emitted = slot_state.emitted.saturating_add(tokens.len() as u32);
            slot_state
                .pending
                .push_back(SlotEvent::Tokens(tokens.to_vec()));
        }
        if out.finished & (1u32 << slot) != 0 {
            let reason = if slot_state.emitted >= slot_state.max_gen {
                FinishReason::Limit
            } else {
                FinishReason::Eos
            };
            slot_state.pending.push_back(SlotEvent::Done { reason });
            slot_state.finished = true;
        }
        // Deliver what we can now; anything the consumer can't take yet stays
        // buffered and is retried next tick without stalling the engine.
        match flush_slot(slot_state) {
            FlushOutcome::Live => {}
            FlushOutcome::Drained => *state = None,
            FlushOutcome::Closed => {
                // Consumer vanished. Cancel the engine slot only if generation
                // was still running (a finished slot is already freed).
                if !slot_state.finished {
                    engine.slot_cancel(slot as u32);
                }
                *state = None;
            }
        }
    }
}
