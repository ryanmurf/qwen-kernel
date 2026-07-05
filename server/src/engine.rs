use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

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
    pub events: tokio_mpsc::Sender<SlotEvent>,
    pub permit: OwnedSemaphorePermit,
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

struct SlotState {
    events: tokio_mpsc::Sender<SlotEvent>,
    emitted: u32,
    max_gen: u32,
    _permit: OwnedSemaphorePermit,
}

impl EngineThread {
    pub fn start(lib: &Path, model: &Path, cfg: QkConfig) -> anyhow::Result<Self> {
        // Open the engine exactly once (the model is ~16 GB of VRAM); read the
        // resolved config off it, then move it into the engine thread. Opening a
        // throwaway "probe" engine here would transiently double VRAM use.
        let engine = Engine::open(lib, model, cfg)?;
        let n_ctx = engine.n_ctx();
        let n_slots = engine.n_slots();
        let chunk = engine.chunk();
        let n_vocab = engine.n_vocab();

        let (tx, rx) = mpsc::channel();
        let join = thread::Builder::new()
            .name("qk-engine".to_owned())
            .spawn(move || run_engine_thread(engine, rx))
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

        if slots.iter().any(Option::is_some) {
            match engine.step_chunk(&mut out) {
                Ok(_) => handle_step(&mut engine, &out, &mut slots),
                Err(err) => {
                    let message = err.to_string();
                    for (slot, state) in slots.iter_mut().enumerate() {
                        if let Some(state) = state.take() {
                            let _ = state
                                .events
                                .blocking_send(SlotEvent::Error(message.clone()));
                            engine.slot_cancel(slot as u32);
                        }
                    }
                }
            }
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
        match engine.slot_start(slot as u32, &job.prompt_ids, job.max_gen) {
            Ok(()) => {
                *state = Some(SlotState {
                    events,
                    emitted: 0,
                    max_gen: job.max_gen,
                    _permit: job.permit,
                });
            }
            Err(err) => {
                let _ = events.blocking_send(SlotEvent::Error(err.to_string()));
            }
        }
    }
}

fn handle_step(engine: &mut Engine, out: &StepOut, slots: &mut [Option<SlotState>]) {
    for (slot, state) in slots.iter_mut().enumerate() {
        let Some(slot_state) = state.as_mut() else {
            continue;
        };
        let tokens = out.slot_tokens(slot as u32);
        if !tokens.is_empty() {
            slot_state.emitted = slot_state.emitted.saturating_add(tokens.len() as u32);
            if slot_state
                .events
                .blocking_send(SlotEvent::Tokens(tokens.to_vec()))
                .is_err()
            {
                engine.slot_cancel(slot as u32);
                *state = None;
                continue;
            }
        }
        if out.finished & (1u32 << slot) != 0 {
            let reason = if slot_state.emitted >= slot_state.max_gen {
                FinishReason::Limit
            } else {
                FinishReason::Eos
            };
            let events = slot_state.events.clone();
            *state = None;
            let _ = events.blocking_send(SlotEvent::Done { reason });
        }
    }
}
