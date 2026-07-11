//! Split-model serving (docs/split-serving.md): this process runs the HEAD
//! stage (layers `[0,S)`, owns the embedding and the whole serving brain) and
//! forwards each position's hidden row to a remote WORKER stage — normally a
//! `qk pipe-worker` — which owns the tail of the model and replies with greedy
//! ids. EOS/limit decisions stay on the head, so `SlotEvent` semantics are
//! identical to the serial driver in `engine.rs`.
//!
//! Wire frame (all u32 little-endian): header `{op, slot, n, base, topk}`;
//! op 1 payload = `n * n_embd` f32 hidden rows for positions `[base, base+n)`,
//! reply = `n` u32 greedy ids, then — iff `topk > 0` — `topk` (u32 id,
//! f32 logit) pairs for the FINAL position, descending. The head samples from
//! those candidates and feeds its pick as the next position, so sampling
//! lives entirely on this side; a greedy request never sets `topk` and the
//! worker's compute path is untouched. op 2 = goodbye (connection close).
//! One frame in flight per connection — this matches the single engine
//! thread on both ends.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::engine::{Cmd, FinishReason, FlushOutcome, Sampling, SlotEvent, SlotState, flush_slot};
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
/// qkp3 = qkp2 + the 5th header word `topk` (sampling candidates, see above).
const PIPE_MAGIC: u32 = 0x716b_7033;

/// Candidates requested per sampled token. Nucleus mass beyond the top 64 is
/// negligible at sane temperatures; the reply stays a 512-byte tail.
const TOPK: u32 = 64;

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
    // Reply shapes (n ids, topk pairs) for op1 frames sent but not yet
    // answered — the worker processes frames strictly in order, so replies
    // match FIFO. Pipelining rule: at most 2 in flight (a 1 MiB prefill
    // frame fits the socket buffers; replies are <= 1 KiB so the worker
    // never blocks on writing one, hence no deadlock).
    pending: VecDeque<(u32, u32)>,
}

impl WorkerLink {
    pub fn new(addr: String, expect: WorkerExpect) -> Self {
        Self {
            addr,
            n_embd: expect.n_embd as usize,
            expect,
            stream: None,
            pending: VecDeque::new(),
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

    /// Fire an op1 frame without waiting for the reply (pipelining: the
    /// worker computes it while the head computes the next chunk/slot).
    /// `topk > 0` asks the worker to append that many (id, logit) sampling
    /// candidates for the frame's final position.
    pub fn send_run(&mut self, slot: u32, base: u32, hidden: &[f32], topk: u32) -> Result<()> {
        let n = (hidden.len() / self.n_embd) as u32;
        let result = (|| {
            let stream = self.stream()?;
            let mut frame = Vec::with_capacity(20 + hidden.len() * 4);
            for word in [1u32, slot, n, base, topk] {
                frame.extend_from_slice(&word.to_le_bytes());
            }
            for value in hidden {
                frame.extend_from_slice(&value.to_le_bytes());
            }
            stream.write_all(&frame).context("split worker write")
        })();
        match result {
            Ok(()) => {
                self.pending.push_back((n, topk));
                Ok(())
            }
            Err(err) => {
                self.poison();
                Err(err)
            }
        }
    }

    /// Receive the oldest in-flight frame's reply (FIFO order): `n` ids into
    /// `ids_out`, and the frame's `topk` candidates into `cands_out` (cleared
    /// — so it holds candidates only right after a topk frame's reply).
    pub fn recv_ids(&mut self, ids_out: &mut Vec<u32>, cands_out: &mut Vec<(u32, f32)>) -> Result<()> {
        let Some((n, topk)) = self.pending.front().copied() else {
            anyhow::bail!("split worker recv with nothing in flight");
        };
        let result = (|| {
            let stream = self.stream()?;
            let mut reply = vec![0u8; n as usize * 4];
            stream.read_exact(&mut reply).context("split worker read")?;
            ids_out.clear();
            ids_out.extend(
                reply
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
            cands_out.clear();
            if topk > 0 {
                let mut pairs = vec![0u8; topk as usize * 8];
                stream
                    .read_exact(&mut pairs)
                    .context("split worker topk read")?;
                cands_out.extend(pairs.chunks_exact(8).map(|b| {
                    (
                        u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                        f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
                    )
                }));
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.pending.pop_front();
                Ok(())
            }
            Err(err) => {
                self.poison();
                Err(err)
            }
        }
    }

    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }

    fn poison(&mut self) {
        self.stream = None;
        self.pending.clear();
    }

    /// State snapshot save (op 3) / load (op 4) of `slot` to/from the
    /// worker's snapshot entry `idx`. Mirrors the head-side qk_state_save/
    /// load so both stages' states move together. Must not interleave with
    /// in-flight op1 frames (replies would misalign).
    /// `n_tok` bounds the attention-KV copy to the snapshot's live length on
    /// the worker (0 = full stripes); ride it in the header's 5th word.
    pub fn state_op(&mut self, save: bool, slot: u32, idx: u32, n_tok: u32) -> Result<()> {
        if !self.pending.is_empty() {
            anyhow::bail!("state_op with {} run frames in flight", self.pending.len());
        }
        let result = (|| {
            let stream = self.stream()?;
            let mut frame = [0u8; 20];
            for (i, word) in [if save { 3u32 } else { 4 }, slot, idx, 0, n_tok]
                .iter()
                .enumerate()
            {
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
            self.poison();
        }
        result
    }

    /// Best-effort goodbye so the worker logs a clean disconnect.
    pub fn bye(&mut self) {
        if let Some(stream) = self.stream.as_mut() {
            let mut header = [0u8; 20];
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
    sampling: Sampling,
    rng: SplitMix64,
}

/// A slot mid-prefill. Prefill is INTERLEAVED with other slots' decode: the
/// main loop feeds one chunk per pass (two, pipelined, when the slot is
/// alone), so admitting a long prompt no longer blocks a decoding neighbor
/// for the whole prefill — the worst-case decode stall is one chunk.
struct Prefill {
    prompt: Vec<u32>,
    done: u32,
    reuse: u32,   // restored prefix length (for the [split-cache] line)
    snap_at: u32, // pending history-boundary snapshot (0 = none/failed)
    snapped: bool,
    sampling: Sampling,
    rng: SplitMix64,
    last_id: Option<u32>,
    cands: Vec<(u32, f32)>,
}

enum Phase {
    Prefilling(Prefill),
    Decoding(Seq),
}

/// Prefill chunk width: 128 when the slot is alone (bulk throughput), 32 when
/// sharing the engine with another active slot (bounds their decode stall).
fn chunk_cap(shared: bool) -> u32 {
    if shared { 32 } else { CHUNK as u32 }
}

/// SplitMix64: tiny deterministic per-request stream (one draw per sampled
/// token). Not cryptographic — it only has to decorrelate token draws.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform draw in [0, 1) with 53 mantissa bits.
    pub(crate) fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Pick a token from `(id, logit)` candidates: softmax at `temp`, nucleus-
/// truncate at `top_p` (mass relative to the K candidates — standard
/// top-k-then-top-p semantics), then spend `r` in [0, 1) across the kept
/// mass. Candidates arrive descending from the worker, but order is
/// re-derived here so a mis-sorted peer degrades to correct-but-slower.
pub(crate) fn sample_candidates(cands: &[(u32, f32)], temp: f32, top_p: f32, r: f64) -> u32 {
    debug_assert!(!cands.is_empty());
    let max = cands.iter().fold(f32::NEG_INFINITY, |m, c| m.max(c.1));
    let temp = f64::from(temp.max(1e-4));
    let weights: Vec<f64> = cands
        .iter()
        .map(|c| (f64::from(c.1 - max) / temp).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Smallest descending prefix whose mass reaches top_p of the total.
    let threshold = total * f64::from(top_p.clamp(1e-6, 1.0));
    let mut kept = 0usize;
    let mut mass = 0.0f64;
    for &i in &order {
        kept += 1;
        mass += weights[i];
        if mass >= threshold {
            break;
        }
    }
    let mut x = r.clamp(0.0, 1.0 - f64::EPSILON) * mass;
    for &i in &order[..kept] {
        if x < weights[i] {
            return cands[i].0;
        }
        x -= weights[i];
    }
    cands[order[kept - 1]].0
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
/// stages — a fast, bounded state copy) and hand back a `Prefill` cursor.
/// The prompt itself is fed chunk-by-chunk from the main loop so other
/// slots keep decoding while this one prefills.
fn admit_job(
    head: &mut Engine,
    link: &mut WorkerLink,
    slot: u32,
    prompt: Vec<u32>,
    snap_prefix: u32,
    sampling: Sampling,
    snaps: &mut [Option<SnapEntry>],
    lru_clock: &mut u64,
) -> Prefill {
    let np = prompt.len() as u32;
    let mut start = 0u32;
    if let Some((idx, len)) = best_snap(snaps, &prompt) {
        match head
            .state_load(slot, idx, len)
            .and_then(|()| link.state_op(false, slot, idx, len))
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
    let snap_at = if snap_prefix > start && snap_prefix < np && !snaps.is_empty() {
        snap_prefix
    } else {
        0
    };
    if !sampling.is_greedy() {
        tracing::info!(
            "[sample] slot={slot} temp={} top_p={} seed={:#018x}",
            sampling.temp,
            sampling.top_p,
            sampling.seed
        );
    }
    let seed = sampling.seed;
    Prefill {
        prompt,
        done: start,
        reuse: start,
        snap_at,
        snapped: false,
        sampling,
        rng: SplitMix64::new(seed),
        last_id: None,
        cands: Vec::new(),
    }
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
    let mut slots: Vec<Option<(SlotState, Phase)>> = (0..n_slots).map(|_| None).collect();
    let mut hidden = Vec::new();
    let mut ids = Vec::new();
    let mut cands: Vec<(u32, f32)> = Vec::new();
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

        // Admit queued jobs into free slots. Admission only restores the
        // longest snapshotted prefix (bounded state copy); the prompt itself
        // is fed chunk-by-chunk below, interleaved with other slots' decode.
        for (slot, state) in slots.iter_mut().enumerate() {
            if state.is_some() {
                continue;
            }
            let Some(job) = queue.pop_front() else { break };
            let prefill = admit_job(
                &mut head,
                &mut link,
                slot as u32,
                job.prompt_ids,
                job.snap_prefix,
                job.sampling,
                &mut snaps,
                &mut lru_clock,
            );
            let slot_state = SlotState::new(job.events, job.max_gen, job.permit);
            *state = Some((slot_state, Phase::Prefilling(prefill)));
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
            // ONE PASS: every decoding slot advances one token, every
            // prefilling slot advances one chunk (two, pipelined, when it is
            // the only occupant), all PIPELINED through the worker: send
            // phase fires every frame, recv phase collects in send order.
            // A long admit therefore stalls a decoding neighbor by at most
            // one small chunk, not a whole prompt.
            let occupied = slots.iter().filter(|s| s.is_some()).count();
            let shared = occupied > 1;
            let mut failure: Option<String> = None;
            // (slot, chunk_len) per sent frame, in order; chunk_len 0 = decode frame.
            let mut sent: Vec<(usize, u32)> = Vec::with_capacity(n_slots + 1);
            for (slot, state) in slots.iter_mut().enumerate() {
                let Some((slot_state, phase)) = state.as_mut() else {
                    continue;
                };
                if slot_state.finished {
                    continue;
                }
                match phase {
                    Phase::Decoding(seq) => {
                        hidden.clear();
                        hidden.resize(n_embd, 0.0);
                        let topk = if seq.sampling.is_greedy() { 0 } else { TOPK };
                        let ok = head
                            .stage_run(
                                slot as u32,
                                Some(&[seq.next]),
                                None,
                                seq.pos,
                                Some(hidden.as_mut_slice()),
                                None,
                            )
                            .and_then(|()| link.send_run(slot as u32, seq.pos, &hidden, topk));
                        match ok {
                            Ok(()) => sent.push((slot, 0)),
                            Err(err) => {
                                failure = Some(err.to_string());
                                break;
                            }
                        }
                    }
                    Phase::Prefilling(p) => {
                        let np = p.prompt.len() as u32;
                        let bursts = if shared { 1 } else { 2 };
                        for _ in 0..bursts {
                            let stop = if p.snap_at > 0 && !p.snapped { p.snap_at } else { np };
                            let n = chunk_cap(shared).min(stop - p.done);
                            if n == 0 {
                                break; // waiting on the snapshot barrier below
                            }
                            let chunk = &p.prompt[p.done as usize..(p.done + n) as usize];
                            hidden.clear();
                            hidden.resize(n as usize * n_embd, 0.0);
                            let topk = if p.done + n == np && !p.sampling.is_greedy() {
                                TOPK
                            } else {
                                0
                            };
                            let ok = head
                                .stage_run(
                                    slot as u32,
                                    Some(chunk),
                                    None,
                                    p.done,
                                    Some(hidden.as_mut_slice()),
                                    None,
                                )
                                .and_then(|()| link.send_run(slot as u32, p.done, &hidden, topk));
                            match ok {
                                Ok(()) => {
                                    sent.push((slot, n));
                                    p.done += n; // reply pending; positions are fed
                                }
                                Err(err) => {
                                    failure = Some(err.to_string());
                                    break;
                                }
                            }
                        }
                        if failure.is_some() {
                            break;
                        }
                    }
                }
            }
            for (slot, chunk_len) in sent {
                if failure.is_some() {
                    break;
                }
                match link.recv_ids(&mut ids, &mut cands) {
                    Ok(()) => {
                        let Some((slot_state, phase)) = slots[slot].as_mut() else {
                            continue;
                        };
                        match phase {
                            Phase::Decoding(seq) => {
                                seq.pos += 1;
                                let id = if cands.is_empty() {
                                    ids[0]
                                } else {
                                    sample_candidates(
                                        &cands,
                                        seq.sampling.temp,
                                        seq.sampling.top_p,
                                        seq.rng.next_f64(),
                                    )
                                };
                                if apply_sample(slot_state, id, eos) {
                                    seq.next = id;
                                }
                                match flush_slot(slot_state) {
                                    FlushOutcome::Live => {}
                                    FlushOutcome::Drained | FlushOutcome::Closed => {
                                        slots[slot] = None
                                    }
                                }
                            }
                            Phase::Prefilling(p) => {
                                let _ = chunk_len;
                                p.last_id = ids.last().copied();
                                if !cands.is_empty() {
                                    p.cands = cands.clone();
                                }
                            }
                        }
                    }
                    Err(err) => failure = Some(err.to_string()),
                }
            }
            // Barrier work (pipe now empty): boundary snapshots and
            // prefill->decode transitions.
            if failure.is_none() {
                for (slot, state) in slots.iter_mut().enumerate() {
                    let Some((slot_state, phase)) = state.as_mut() else {
                        continue;
                    };
                    let Phase::Prefilling(p) = phase else { continue };
                    let np = p.prompt.len() as u32;
                    if p.snap_at > 0 && !p.snapped && p.done == p.snap_at {
                        let idx = pick_entry(&snaps);
                        match head
                            .state_save(slot as u32, idx, p.snap_at)
                            .and_then(|()| link.state_op(true, slot as u32, idx, p.snap_at))
                        {
                            Ok(()) => {
                                lru_clock += 1;
                                snaps[idx as usize] = Some(SnapEntry {
                                    tokens: p.prompt[..p.snap_at as usize].to_vec(),
                                    lru: lru_clock,
                                });
                            }
                            Err(err) => {
                                // Snapshot is an optimization; the request proceeds.
                                tracing::warn!("split-cache snapshot failed: {err}");
                                snaps[idx as usize] = None;
                                p.snap_at = 0;
                            }
                        }
                        p.snapped = true;
                    }
                    if p.done == np {
                        tracing::info!(
                            "[split-cache] slot={slot} prompt={np} reuse={} prefill={} snap={}",
                            p.reuse,
                            np - p.reuse,
                            if p.snapped { p.snap_at } else { 0 }
                        );
                        let first = match (p.last_id, p.cands.is_empty()) {
                            (Some(_), false) => sample_candidates(
                                &p.cands,
                                p.sampling.temp,
                                p.sampling.top_p,
                                p.rng.next_f64(),
                            ),
                            (Some(id), true) => id,
                            (None, _) => {
                                let _ = slot_state
                                    .events
                                    .try_send(SlotEvent::Error("prefill produced no ids".into()));
                                *state = None;
                                continue;
                            }
                        };
                        let live = apply_sample(slot_state, first, eos);
                        let seq = Seq {
                            pos: np,
                            next: if live { first } else { 0 },
                            sampling: p.sampling,
                            rng: SplitMix64::new(0), // replaced below
                        };
                        // Move the request rng into the decode phase so draws
                        // continue the same per-request stream.
                        let rng = std::mem::replace(&mut p.rng, SplitMix64::new(0));
                        let mut seq = seq;
                        seq.rng = rng;
                        *phase = Phase::Decoding(seq);
                        match flush_slot(slot_state) {
                            FlushOutcome::Live => {}
                            FlushOutcome::Drained | FlushOutcome::Closed => *state = None,
                        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_zero_temperature_is_greedy() {
        let cands = vec![(7u32, 10.0f32), (3, 9.9), (5, 2.0)];
        for r in [0.0, 0.3, 0.7, 0.999] {
            assert_eq!(sample_candidates(&cands, 1e-4, 1.0, r), 7);
        }
    }

    #[test]
    fn tiny_top_p_keeps_only_the_head() {
        let cands = vec![(7u32, 5.0f32), (3, 4.99), (5, 4.98)];
        for r in [0.0, 0.5, 0.999] {
            assert_eq!(sample_candidates(&cands, 1.0, 1e-6, r), 7);
        }
    }

    #[test]
    fn equal_logits_spread_by_draw() {
        // Two equal candidates at top_p=1: r below/above 0.5 picks each side.
        let cands = vec![(1u32, 3.0f32), (2, 3.0)];
        assert_eq!(sample_candidates(&cands, 1.0, 1.0, 0.25), 1);
        assert_eq!(sample_candidates(&cands, 1.0, 1.0, 0.75), 2);
    }

    #[test]
    fn top_p_drops_the_tail() {
        // Head carries ~73% of mass at temp 1 (gap of 1 nat); top_p=0.5
        // keeps only it, so every draw returns the head.
        let cands = vec![(9u32, 2.0f32), (4, 1.0)];
        for r in [0.1, 0.6, 0.99] {
            assert_eq!(sample_candidates(&cands, 1.0, 0.5, r), 9);
        }
        // top_p=1 with a high draw reaches the tail.
        assert_eq!(sample_candidates(&cands, 1.0, 1.0, 0.9), 4);
    }

    #[test]
    fn unsorted_candidates_still_sample_correctly() {
        let cands = vec![(4u32, 1.0f32), (9, 2.0)]; // ascending — mis-sorted peer
        for r in [0.1, 0.6, 0.99] {
            assert_eq!(sample_candidates(&cands, 1.0, 0.5, r), 9);
        }
    }

    #[test]
    fn splitmix_is_deterministic_and_in_range() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        let mut c = SplitMix64::new(43);
        let mut differs = false;
        for _ in 0..64 {
            let x = a.next_f64();
            assert_eq!(x, b.next_f64());
            assert!((0.0..1.0).contains(&x));
            differs |= x != c.next_f64();
        }
        assert!(differs, "different seeds must produce different streams");
    }
}
