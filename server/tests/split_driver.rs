//! Split-serving driver against the deterministic stub: an in-process "worker"
//! (stub engine, layers [20,40)) serves the pipe-worker frame protocol over
//! TCP while EngineThread runs the head stage (layers [0,20)) through the
//! split driver. The emitted stream must equal the stub's serial LCG rule.
//! Single #[test]: this binary mutates QK_LAYERS (process-global env).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use server::engine::{EngineThread, FinishReason, Job, SlotEvent};
use server::ffi::{Engine, QkConfig};
use tokio::sync::{Semaphore, mpsc};

const CFG: QkConfig = QkConfig {
    n_slots: 2,
    n_ctx: 64,
    chunk: 4,
};
const N_EMBD: usize = 4; // stub's hidden row width

fn open_stub() -> Engine {
    Engine::open(Path::new(env!("QK_STUB_LIB")), Path::new("/dev/null"), CFG)
        .expect("stub engine opens")
}

fn lcg(prev: u32) -> u32 {
    (prev.wrapping_mul(1_103_515_245).wrapping_add(12_345)) % 248_000
}

const PIPE_MAGIC: u32 = 0x716b_7032;

/// (op, slot, n_or_idx, base) of every frame the worker saw — the reuse test
/// asserts turn 2 restores (op4) and prefills only its delta.
type FrameLog = Arc<Mutex<Vec<(u32, u32, u32, u32)>>>;

/// Minimal `qk pipe-worker` in Rust: hello on connect, then op1 run frames /
/// op3-op4 state frames until the peer disconnects or says op2.
/// `hello_n_embd` lets a test advertise a mismatched topology.
fn serve_worker(mut engine: Engine, listener: TcpListener, hello_n_embd: u32, frames: FrameLog) {
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        stream.set_nodelay(true).ok();
        let mut magic = [0u8; 4];
        if stream.read_exact(&mut magic).is_err() || u32::from_le_bytes(magic) != PIPE_MAGIC {
            continue;
        }
        let mut hello = Vec::new();
        for word in [PIPE_MAGIC, 20, 40, 40, hello_n_embd, CFG.n_slots, CFG.n_ctx] {
            hello.extend_from_slice(&word.to_le_bytes());
        }
        if stream.write_all(&hello).is_err() {
            continue;
        }
        serve_conn(&mut engine, &mut stream, &frames);
    }
}

fn serve_conn(engine: &mut Engine, stream: &mut TcpStream, frames: &FrameLog) {
    loop {
        let mut header = [0u8; 16];
        if stream.read_exact(&mut header).is_err() {
            return;
        }
        let word = |i: usize| u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
        let (op, slot, n, base) = (word(0), word(1), word(2), word(3));
        frames.lock().unwrap().push((op, slot, n, base));
        match op {
            3 | 4 => {
                let rc = if op == 3 {
                    engine.state_save(slot, n)
                } else {
                    engine.state_load(slot, n)
                };
                let status = u32::from(rc.is_err());
                if stream.write_all(&status.to_le_bytes()).is_err() {
                    return;
                }
            }
            1 => {
                let n = n as usize;
                let mut payload = vec![0u8; n * N_EMBD * 4];
                if stream.read_exact(&mut payload).is_err() {
                    return;
                }
                let hidden: Vec<f32> = payload
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let mut ids = vec![0u32; n];
                engine
                    .stage_run(slot, None, Some(&hidden), base, None, Some(&mut ids))
                    .expect("worker stage_run");
                let mut reply = Vec::with_capacity(n * 4);
                for id in &ids {
                    reply.extend_from_slice(&id.to_le_bytes());
                }
                if stream.write_all(&reply).is_err() {
                    return;
                }
            }
            _ => return,
        }
    }
}

/// Submit one prompt and collect (tokens, finish) off the event channel.
fn run_job(
    engine: &EngineThread,
    sem: &Arc<Semaphore>,
    prompt: Vec<u32>,
    max_gen: u32,
) -> (Vec<u32>, Result<FinishReason, String>) {
    run_job_snap(engine, sem, prompt, max_gen, 0)
}

fn run_job_snap(
    engine: &EngineThread,
    sem: &Arc<Semaphore>,
    prompt: Vec<u32>,
    max_gen: u32,
    snap_prefix: u32,
) -> (Vec<u32>, Result<FinishReason, String>) {
    let (tx, mut rx) = mpsc::channel(64);
    engine
        .handle()
        .submit(Job {
            prompt_ids: prompt,
            max_gen,
            snap_prefix,
            events: tx,
            permit: sem.clone().try_acquire_owned().expect("permit"),
        })
        .expect("submit");
    let mut tokens = Vec::new();
    loop {
        match rx.blocking_recv().expect("engine dropped channel early") {
            SlotEvent::Tokens(t) => tokens.extend(t),
            SlotEvent::Done { reason } => return (tokens, Ok(reason)),
            SlotEvent::Error(e) => return (tokens, Err(e)),
        }
    }
}

#[test]
fn split_driver_serves_the_serial_stream() {
    // Worker: stub engine holding layers [20,40) behind the frame protocol.
    unsafe { std::env::set_var("QK_LAYERS", "20:40") };
    let worker_engine = open_stub();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr").to_string();
    let frames: FrameLog = Arc::new(Mutex::new(Vec::new()));
    let worker_frames = frames.clone();
    thread::spawn(move || serve_worker(worker_engine, listener, N_EMBD as u32, worker_frames));

    // A second worker advertising the wrong n_embd: the hello must reject it.
    unsafe { std::env::set_var("QK_LAYERS", "20:40") };
    let bad_engine = open_stub();
    let bad_listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let bad_addr = bad_listener.local_addr().expect("addr").to_string();
    let bad_frames: FrameLog = Arc::new(Mutex::new(Vec::new()));
    thread::spawn(move || serve_worker(bad_engine, bad_listener, 8, bad_frames));

    // Misconfigurations are rejected at startup, before serving anything:
    // a split engine with no --split-next…
    unsafe { std::env::set_var("QK_LAYERS", "0:20") };
    let Err(err) =
        EngineThread::start(Path::new(env!("QK_STUB_LIB")), Path::new("/dev/null"), CFG, None)
    else {
        panic!("split engine without --split-next must fail")
    };
    assert!(err.to_string().contains("--split-next"), "{err}");

    // Head: stub engine holding layers [0,20), split driver to the worker.
    let engine = EngineThread::start(
        Path::new(env!("QK_STUB_LIB")),
        Path::new("/dev/null"),
        CFG,
        Some(addr),
    )
    .expect("head starts");
    unsafe { std::env::remove_var("QK_LAYERS") };

    // …and --split-next with an unsplit engine.
    let Err(err) = EngineThread::start(
        Path::new(env!("QK_STUB_LIB")),
        Path::new("/dev/null"),
        CFG,
        Some("127.0.0.1:1".into()),
    ) else {
        panic!("--split-next with an unsplit engine must fail")
    };
    assert!(err.to_string().contains("head stage"), "{err}");

    let sem = Arc::new(Semaphore::new(4));

    // Happy path: the emitted stream is exactly the stub's serial LCG chain.
    let (tokens, finish) = run_job(&engine, &sem, vec![3, 9, 11], 6);
    let mut expect = Vec::new();
    let mut prev = 11u32;
    for _ in 0..6 {
        prev = lcg(prev);
        expect.push(prev);
    }
    assert_eq!(tokens, expect);
    assert_eq!(finish, Ok(FinishReason::Limit));

    // Slot + connection reuse: a second sequence starts clean at base 0.
    let (tokens, finish) = run_job(&engine, &sem, vec![42, 7, 100, 11], 3);
    assert_eq!(tokens, vec![lcg(11), lcg(lcg(11)), lcg(lcg(lcg(11)))]);
    assert_eq!(finish, Ok(FinishReason::Limit));

    // A prompt longer than the head's prefill chunk still chains correctly
    // (CHUNK=128 in split.rs; n_ctx=64 here, so use 40 tokens ending in 11 —
    // multiple stage_run chunks inside the engine, one frame per 128 anyway).
    let mut long = vec![5u32; 39];
    long.push(11);
    let (tokens, _) = run_job(&engine, &sem, long, 2);
    assert_eq!(tokens, vec![lcg(11), lcg(lcg(11))]);

    // Cross-turn reuse: turn 1 snapshots at the history boundary (op3);
    // turn 2's prompt extends that prefix, so the driver restores (op4) and
    // prefills ONLY the delta.
    let p1: Vec<u32> = (1..=39).chain([11]).collect(); // 40 tokens
    let boundary = 32u32;
    let mark = frames.lock().unwrap().len();
    let (t1, _) = run_job_snap(&engine, &sem, p1.clone(), 2, boundary);
    assert_eq!(t1[0], lcg(11));
    {
        let f = frames.lock().unwrap();
        let turn1 = &f[mark..];
        assert!(
            turn1.iter().any(|&(op, _, _, _)| op == 3),
            "turn 1 must snapshot at the boundary: {turn1:?}"
        );
        // op1 frames must split exactly at the boundary (no re-fed tokens).
        let runs: Vec<_> = turn1.iter().filter(|f| f.0 == 1).collect();
        assert_eq!(runs.first().map(|f| (f.2, f.3)), Some((boundary, 0)));
    }
    let mut p2 = p1[..boundary as usize].to_vec();
    p2.extend([7, 7, 7, 7, 11]); // extends the boundary prefix, new tail
    let mark = frames.lock().unwrap().len();
    let (t2, _) = run_job_snap(&engine, &sem, p2.clone(), 2, p2.len() as u32 - 2);
    assert_eq!(t2[0], lcg(11)); // stub ids derive from the fed token alone
    {
        let f = frames.lock().unwrap();
        let turn2 = &f[mark..];
        assert!(
            turn2.iter().any(|&(op, _, _, _)| op == 4),
            "turn 2 must restore the boundary snapshot: {turn2:?}"
        );
        // Prefill frames only (decode steps run at base >= |prompt|).
        let fed: u32 = turn2
            .iter()
            .filter(|f| f.0 == 1 && f.3 < p2.len() as u32)
            .map(|f| f.2)
            .sum();
        assert_eq!(
            fed,
            p2.len() as u32 - boundary,
            "turn 2 must prefill only the delta: {turn2:?}"
        );
        // …and the delta starts at the boundary, not at 0.
        let first_run = turn2.iter().find(|f| f.0 == 1).expect("a run frame");
        assert_eq!(first_run.3, boundary);
    }

    engine.shutdown();

    // Topology-mismatch worker: the hello rejects it with a clear error.
    unsafe { std::env::set_var("QK_LAYERS", "0:20") };
    let engine = EngineThread::start(
        Path::new(env!("QK_STUB_LIB")),
        Path::new("/dev/null"),
        CFG,
        Some(bad_addr),
    )
    .expect("head starts (hello happens lazily)");
    unsafe { std::env::remove_var("QK_LAYERS") };
    let (tokens, finish) = run_job(&engine, &sem, vec![3, 9, 11], 4);
    assert!(tokens.is_empty());
    let err = finish.expect_err("mismatched worker must be rejected");
    assert!(err.contains("mismatch"), "{err}");
    engine.shutdown();

    // Dead worker: requests fail with an Error event; startup still succeeds
    // (the link connects lazily) and the thread survives to serve the error.
    unsafe { std::env::set_var("QK_LAYERS", "0:20") };
    let dead_port = {
        let l = TcpListener::bind("127.0.0.1:0").expect("bind");
        l.local_addr().expect("addr").to_string()
    }; // listener dropped: nothing serves this port
    let engine = EngineThread::start(
        Path::new(env!("QK_STUB_LIB")),
        Path::new("/dev/null"),
        CFG,
        Some(dead_port),
    )
    .expect("head starts even with the worker down");
    unsafe { std::env::remove_var("QK_LAYERS") };
    let (tokens, finish) = run_job(&engine, &sem, vec![3, 9, 11], 4);
    assert!(tokens.is_empty());
    let err = finish.expect_err("job against a dead worker must error");
    assert!(err.contains("split worker"), "{err}");
    engine.shutdown();
}
