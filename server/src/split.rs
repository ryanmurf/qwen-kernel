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

/// Connection hello: client sends the magic, worker replies
/// `{magic, layer_first, layer_end, n_layer, n_embd, n_slots, n_ctx}` —
/// mixed builds or a mis-split worker fail loudly at connect instead of
/// streaming garbage. Bump on any wire change. qkp2 = qkp1 + state ops
/// (op3 save / op4 load, snapshot idx in the n field, 4-byte status reply).
const PIPE_MAGIC: u32 = 0x716b_7032;

/// What the head requires of the worker, checked against its hello.
pub struct WorkerExpect {
    pub layer_first: u32, // = head's layer_end (contiguous split)
    pub n_layer: u32,     // worker must run through the final layer
    pub n_embd: u32,
    pub n_slots: u32, // worker must cover at least the head's slots…
    pub n_ctx: u32,   // …and context
}

pub struct WorkerLink {
    addr: String,
    n_embd: usize,
    expect: WorkerExpect,
    stream: Option<TcpStream>,
}

impl WorkerLink {
    pub fn new(addr: String, expect: WorkerExpect) -> Self {
        Self {
            addr,
            n_embd: expect.n_embd as usize,
            expect,
            stream: None,
        }
    }

    fn stream(&mut self) -> Result<&mut TcpStream> {
        if self.stream.is_none() {
            let mut stream = TcpStream::connect(&self.addr)
                .with_context(|| format!("cannot connect to split worker {}", self.addr))?;
            stream.set_nodelay(true)?;
            stream.set_read_timeout(Some(IO_TIMEOUT))?;
            stream.set_write_timeout(Some(IO_TIMEOUT))?;
            stream
                .write_all(&PIPE_MAGIC.to_le_bytes())
                .context("split worker hello write")?;
            let mut raw = [0u8; 28];
            stream
                .read_exact(&mut raw)
                .context("split worker hello read (old/foreign build?)")?;
            let mut hello = [0u32; 7];
            for (word, bytes) in hello.iter_mut().zip(raw.chunks_exact(4)) {
                *word = u32::from_le_bytes(bytes.try_into().expect("4-byte chunk"));
            }
            let e = &self.expect;
            if hello[0] != PIPE_MAGIC {
                anyhow::bail!("split worker hello: bad magic {:#010x}", hello[0]);
            }
            if hello[1] != e.layer_first
                || hello[2] != e.n_layer
                || hello[3] != e.n_layer
                || hello[4] != e.n_embd
                || hello[5] < e.n_slots
                || hello[6] < e.n_ctx
            {
                anyhow::bail!(
                    "split worker mismatch: layers [{}, {}) of {}, n_embd {}, slots {}, ctx {} \
                     (head needs layers [{}, {}) of {}, n_embd {}, slots >= {}, ctx >= {})",
                    hello[1],
                    hello[2],
                    hello[3],
                    hello[4],
                    hello[5],
                    hello[6],
                    e.layer_first,
                    e.n_layer,
                    e.n_layer,
                    e.n_embd,
                    e.n_slots,
                    e.n_ctx
                );
            }
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

    /// State snapshot save (op 3) / load (op 4) of `slot` to/from the
    /// worker's snapshot entry `idx`. Mirrors the head-side qk_state_save/
    /// load so both stages' states move together.
    pub fn state_op(&mut self, save: bool, slot: u32, idx: u32) -> Result<()> {
        let result = (|| {
            let stream = self.stream()?;
            let mut frame = [0u8; 16];
            for (i, word) in [if save { 3u32 } else { 4 }, slot, idx, 0].iter().enumerate() {
                frame[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            stream.write_all(&frame).context("split worker state write")?;
            let mut status = [0u8; 4];
            stream
                .read_exact(&mut status)
                .context("split worker state read")?;
            if u32::from_le_bytes(status) != 0 {
                anyhow::bail!("worker state {} failed", if save { "save" } else { "load" });
            }
            Ok(())
        })();
        if result.is_err() {
            self.stream = None;
        }
        result
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

/// Chunked prefill of `prompt[start..end)` through both stages from
/// base=start (start 0 resets the slot on both engines; start>0 continues
/// restored state); returns the worker's id after position end-1.
#[allow(clippy::too_many_arguments)]
fn split_prefill(
    head: &mut Engine,
    link: &mut WorkerLink,
    slot: u32,
    prompt: &[u32],
    start: u32,
    end: u32,
    n_embd: usize,
    hidden: &mut Vec<f32>,
    ids: &mut Vec<u32>,
) -> Result<u32> {
    let mut base = start;
    let mut last = None;
    for chunk in prompt[start as usize..end as usize].chunks(CHUNK) {
        hidden.clear();
        hidden.resize(chunk.len() * n_embd, 0.0);
        head.stage_run(slot, Some(chunk), None, base, Some(hidden.as_mut_slice()), None)?;
        link.run(slot, base, hidden, ids)?;
        last = ids.last().copied();
        base += chunk.len() as u32;
    }
    last.context("split prefill produced no ids")
}

/// Driver-side snapshot registry: which token prefix lives in each engine
/// snapshot entry (the same index on the head and the worker — states move
/// together). Mirrors the single-box engine's pcache, but keyed host-side.
struct SnapEntry {
    tokens: Vec<u32>,
    lru: u64,
}

/// Longest registered strict prefix of `prompt` (>= 8 tokens), as
/// (entry index, prefix length).
fn best_snap(snaps: &[Option<SnapEntry>], prompt: &[u32]) -> Option<(u32, u32)> {
    let mut best: Option<(u32, u32)> = None;
    for (idx, entry) in snaps.iter().enumerate() {
        let Some(e) = entry else { continue };
        let len = e.tokens.len();
        if len < 8 || len >= prompt.len() || best.is_some_and(|(_, l)| len as u32 <= l) {
            continue;
        }
        if e.tokens[..] == prompt[..len] {
            best = Some((idx as u32, len as u32));
        }
    }
    best
}

/// Admit one job: restore the longest snapshotted prefix if any (both
/// stages), prefill the remainder, and take a history-boundary snapshot at
/// job-specified `snap_prefix` on the way through. Returns the first
/// generated token.
#[allow(clippy::too_many_arguments)]
fn split_admit(
    head: &mut Engine,
    link: &mut WorkerLink,
    slot: u32,
    prompt: &[u32],
    snap_prefix: u32,
    snaps: &mut [Option<SnapEntry>],
    lru_clock: &mut u64,
    n_embd: usize,
    hidden: &mut Vec<f32>,
    ids: &mut Vec<u32>,
) -> Result<u32> {
    let np = prompt.len() as u32;
    let mut start = 0u32;
    if let Some((idx, len)) = best_snap(snaps, prompt) {
        match head
            .state_load(slot, idx)
            .and_then(|()| link.state_op(false, slot, idx))
        {
            Ok(()) => {
                start = len;
                *lru_clock += 1;
                if let Some(e) = snaps[idx as usize].as_mut() {
                    e.lru = *lru_clock;
                }
            }
            Err(err) => {
                // Fall back to a cold prefill; the restore may have half
                // happened, but base=0 resets both stages anyway.
                tracing::warn!("split-cache restore failed, going cold: {err}");
                start = 0;
            }
        }
    }
    let mut snap_at = if snap_prefix > start && snap_prefix < np && !snaps.is_empty() {
        snap_prefix
    } else {
        0
    };
    let first = if snap_at > 0 {
        // Prefill to the history boundary, snapshot BOTH stages there, then
        // finish the prompt — the next turn restores at the boundary and
        // prefills only its delta. The phase boundary holds even if the
        // snapshot fails (re-feeding fed tokens would corrupt DeltaNet state).
        let boundary = snap_at;
        split_prefill(head, link, slot, prompt, start, boundary, n_embd, hidden, ids)?;
        let idx = pick_entry(snaps);
        match head
            .state_save(slot, idx)
            .and_then(|()| link.state_op(true, slot, idx))
        {
            Ok(()) => {
                *lru_clock += 1;
                snaps[idx as usize] = Some(SnapEntry {
                    tokens: prompt[..boundary as usize].to_vec(),
                    lru: *lru_clock,
                });
            }
            Err(err) => {
                // Snapshot is an optimization; the request itself proceeds.
                tracing::warn!("split-cache snapshot failed: {err}");
                snaps[idx as usize] = None;
                snap_at = 0;
            }
        }
        split_prefill(head, link, slot, prompt, boundary, np, n_embd, hidden, ids)?
    } else {
        split_prefill(head, link, slot, prompt, start, np, n_embd, hidden, ids)?
    };
    tracing::info!(
        "[split-cache] slot={slot} prompt={np} reuse={start} prefill={} snap={snap_at}",
        np - start
    );
    Ok(first)
}

/// LRU entry choice: first empty slot, else the least recently used.
fn pick_entry(snaps: &[Option<SnapEntry>]) -> u32 {
    let mut idx = 0u32;
    let mut best = u64::MAX;
    for (i, e) in snaps.iter().enumerate() {
        match e {
            None => return i as u32,
            Some(s) if s.lru < best => {
                best = s.lru;
                idx = i as u32;
            }
            _ => {}
        }
    }
    idx
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
    // Cross-turn snapshot registry (0 entries when the engine lib predates
    // the snapshot ABI — the feature just stays off). Cleared on any worker
    // link failure: a restarted worker has lost its half of every snapshot,
    // and a blip is indistinguishable from a restart.
    let mut snaps: Vec<Option<SnapEntry>> = (0..head.state_n()).map(|_| None).collect();
    let mut lru_clock = 0u64;

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
            match split_admit(
                &mut head,
                &mut link,
                slot as u32,
                &job.prompt_ids,
                job.snap_prefix,
                &mut snaps,
                &mut lru_clock,
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
                    snaps.iter_mut().for_each(|e| *e = None);
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
                // that was still generating and void the snapshot registry
                // (a restarted worker lost its half). Finished slots drain.
                snaps.iter_mut().for_each(|e| *e = None);
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
