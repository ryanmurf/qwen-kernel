//! Split-model serving (docs/split-serving.md): this process runs the HEAD
//! stage (layers `[0,S)`, owns the embedding and the whole serving brain) and
//! forwards each position's hidden row to a remote WORKER stage — normally a
//! `qk pipe-worker` — which owns the tail of the model and replies with greedy
//! ids. EOS/limit decisions stay on the head, so `SlotEvent` semantics are
//! identical to the serial driver in `engine.rs`.
//!
//! Wire frame (all u32 little-endian): header `{op, slot, n, base}`;
//! op 1 payload = `n * n_embd` f32 hidden rows for positions `[base, base+n)`,
//! reply = `n` u32 greedy ids; op 2 = goodbye (connection close). One frame in
//! flight per connection — this matches the single engine thread on both ends.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::engine::{Cmd, FinishReason, FlushOutcome, SlotEvent, SlotState, flush_slot};
use crate::ffi::Engine;

/// Generous ceiling: a worker prefill chunk is <1 s on a healthy GPU, but a
/// thermally clamped or contended device can stretch far past that.
const IO_TIMEOUT: Duration = Duration::from_secs(120);

/// Head-side prefill chunk width (tokens per stage_run + frame). The engine
/// re-chunks internally at its own batch cap, so this only bounds frame size.
const CHUNK: usize = 128;

pub struct WorkerLink {
    addr: String,
    n_embd: usize,
    stream: Option<TcpStream>,
}

impl WorkerLink {
    pub fn new(addr: String, n_embd: usize) -> Self {
        Self {
            addr,
            n_embd,
            stream: None,
        }
    }

    fn stream(&mut self) -> Result<&mut TcpStream> {
        if self.stream.is_none() {
            let stream = TcpStream::connect(&self.addr)
                .with_context(|| format!("cannot connect to split worker {}", self.addr))?;
            stream.set_nodelay(true)?;
            stream.set_read_timeout(Some(IO_TIMEOUT))?;
            stream.set_write_timeout(Some(IO_TIMEOUT))?;
            self.stream = Some(stream);
        }
        Ok(self.stream.as_mut().expect("just connected"))
    }

    /// Send hidden rows for positions `[base, base+n)` of `slot`; fill
    /// `ids_out` with the worker's `n` greedy ids. Any failure poisons the
    /// connection: it is dropped and the next call reconnects fresh (safe —
    /// sequences always restart at base 0, which resets the worker slot).
    pub fn run(&mut self, slot: u32, base: u32, hidden: &[f32], ids_out: &mut Vec<u32>) -> Result<()> {
        let result = self.run_inner(slot, base, hidden, ids_out);
        if result.is_err() {
            self.stream = None;
        }
        result
    }

    fn run_inner(
        &mut self,
        slot: u32,
        base: u32,
        hidden: &[f32],
        ids_out: &mut Vec<u32>,
    ) -> Result<()> {
        let n = hidden.len() / self.n_embd;
        let stream = self.stream()?;
        let mut frame = Vec::with_capacity(16 + hidden.len() * 4);
        for word in [1u32, slot, n as u32, base] {
            frame.extend_from_slice(&word.to_le_bytes());
        }
        for value in hidden {
            frame.extend_from_slice(&value.to_le_bytes());
        }
        stream.write_all(&frame).context("split worker write")?;
        let mut reply = vec![0u8; n * 4];
        stream.read_exact(&mut reply).context("split worker read")?;
        ids_out.clear();
        ids_out.extend(
            reply
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        Ok(())
    }

    /// Best-effort goodbye so the worker logs a clean disconnect.
    pub fn bye(&mut self) {
        if let Some(stream) = self.stream.as_mut() {
            let mut header = [0u8; 16];
            header[0] = 2;
            let _ = stream.write_all(&header);
        }
        self.stream = None;
    }
}

/// Head-side per-sequence cursor: `next` is the last sampled token, fed at
/// position `pos` to produce the following one.
struct Seq {
    pos: u32,
    next: u32,
}

/// Apply one sampled id to a slot: emit it unless it is EOS, finish on
/// EOS/limit. Returns false when the sequence must stop feeding.
fn apply_sample(state: &mut SlotState, id: u32, eos: u32) -> bool {
    if id == eos {
        state.pending.push_back(SlotEvent::Done {
            reason: FinishReason::Eos,
        });
        state.finished = true;
        return false;
    }
    state.emitted = state.emitted.saturating_add(1);
    state.pending.push_back(SlotEvent::Tokens(vec![id]));
    if state.emitted >= state.max_gen {
        state.pending.push_back(SlotEvent::Done {
            reason: FinishReason::Limit,
        });
        state.finished = true;
        return false;
    }
    true
}

/// Chunked prefill through both stages; returns the first generated token
/// (the worker's id after the final prompt position). base 0 resets the slot
/// on both engines.
fn split_prefill(
    head: &mut Engine,
    link: &mut WorkerLink,
    slot: u32,
    prompt: &[u32],
    n_embd: usize,
    hidden: &mut Vec<f32>,
    ids: &mut Vec<u32>,
) -> Result<u32> {
    let mut base = 0u32;
    let mut last = None;
    for chunk in prompt.chunks(CHUNK) {
        hidden.clear();
        hidden.resize(chunk.len() * n_embd, 0.0);
        head.stage_run(slot, Some(chunk), None, base, Some(hidden.as_mut_slice()), None)?;
        link.run(slot, base, hidden, ids)?;
        last = ids.last().copied();
        base += chunk.len() as u32;
    }
    last.context("split prefill produced no ids")
}

/// Split-mode replacement for `run_engine_thread`: same queue/flush/
/// backpressure semantics, but slots advance one token at a time through the
/// head stage + worker instead of `qk_step_chunk`. A worker or engine failure
/// fails every unfinished slot (worker state is unknown at that point) and
/// the next sequence reconnects from scratch.
pub fn run_split_engine_thread(mut head: Engine, mut link: WorkerLink, rx: mpsc::Receiver<Cmd>) {
    let n_slots = head.n_slots() as usize;
    let eos = head.eos_token();
    let n_embd = link.n_embd;
    let mut queue = VecDeque::new();
    let mut slots: Vec<Option<(SlotState, Seq)>> = (0..n_slots).map(|_| None).collect();
    let mut hidden = Vec::new();
    let mut ids = Vec::new();

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
                Cmd::Shutdown => {
                    link.bye();
                    return;
                }
            }
        }

        // Admit queued jobs into free slots; the whole prefill runs here,
        // exactly like qk_slot_start does on the serial path.
        for (slot, state) in slots.iter_mut().enumerate() {
            if state.is_some() {
                continue;
            }
            let Some(job) = queue.pop_front() else { break };
            match split_prefill(
                &mut head,
                &mut link,
                slot as u32,
                &job.prompt_ids,
                n_embd,
                &mut hidden,
                &mut ids,
            ) {
                Ok(first) => {
                    let mut slot_state = SlotState::new(job.events, job.max_gen, job.permit);
                    let live = apply_sample(&mut slot_state, first, eos);
                    let seq = Seq {
                        pos: job.prompt_ids.len() as u32,
                        next: if live { first } else { 0 },
                    };
                    *state = Some((slot_state, seq));
                }
                Err(err) => {
                    let _ = job.events.try_send(SlotEvent::Error(err.to_string()));
                }
            }
        }

        // Flush backlogs; find out who still needs GPU work.
        let mut any_active = false;
        for state in slots.iter_mut() {
            let Some((slot_state, _)) = state.as_mut() else {
                continue;
            };
            if slot_state.finished {
                match flush_slot(slot_state) {
                    FlushOutcome::Live => {}
                    FlushOutcome::Drained | FlushOutcome::Closed => *state = None,
                }
            } else {
                any_active = true;
                if let FlushOutcome::Closed = flush_slot(slot_state) {
                    // Consumer vanished mid-generation. No engine cancel is
                    // needed: the next sequence on this slot resets at base 0.
                    *state = None;
                }
            }
        }

        if any_active {
            // One token per active slot per pass (round-robin fairness).
            let mut failure: Option<String> = None;
            for (slot, state) in slots.iter_mut().enumerate() {
                let Some((slot_state, seq)) = state.as_mut() else {
                    continue;
                };
                if slot_state.finished {
                    continue;
                }
                hidden.clear();
                hidden.resize(n_embd, 0.0);
                let step = head
                    .stage_run(
                        slot as u32,
                        Some(&[seq.next]),
                        None,
                        seq.pos,
                        Some(hidden.as_mut_slice()),
                        None,
                    )
                    .and_then(|()| link.run(slot as u32, seq.pos, &hidden, &mut ids));
                match step {
                    Ok(()) => {
                        seq.pos += 1;
                        let id = ids[0];
                        if apply_sample(slot_state, id, eos) {
                            seq.next = id;
                        }
                        match flush_slot(slot_state) {
                            FlushOutcome::Live => {}
                            FlushOutcome::Drained => *state = None,
                            FlushOutcome::Closed => *state = None,
                        }
                    }
                    Err(err) => {
                        failure = Some(err.to_string());
                        break;
                    }
                }
            }
            if let Some(message) = failure {
                // Worker (or head engine) state is unknown; fail everything
                // that was still generating. Finished slots keep draining.
                for state in slots.iter_mut() {
                    let Some((slot_state, _)) = state.as_mut() else {
                        continue;
                    };
                    if slot_state.finished {
                        continue;
                    }
                    let _ = slot_state
                        .events
                        .try_send(SlotEvent::Error(message.clone()));
                    *state = None;
                }
            }
        } else if slots.iter().any(Option::is_some) {
            // Only drain-pending slots remain (a stalled consumer).
            thread::sleep(Duration::from_millis(2));
        }
    }
    link.bye();
}
