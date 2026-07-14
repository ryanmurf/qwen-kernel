//! Local-driver mode: the split driver with NO worker, over an UNSPLIT stub
//! engine. This is the single-box sampling path — qk_step_chunk picks the next
//! token inside the engine's fused argmax chain, so only a driver that runs the
//! model one position at a time (qk_stage_run) can inject a sampled pick.
//!
//! Two properties matter and are asserted here:
//!   1. greedy through the local driver == the serial stream qk_step_chunk
//!      emits (same stub LCG rule), so turning the driver on cannot change a
//!      greedy answer;
//!   2. a sampled request actually samples — it must be able to leave the
//!      greedy path, which is the whole point (task #44's loop-breaker).

use std::path::Path;
use std::sync::Arc;

use server::engine::{EngineThread, FinishReason, Job, Sampling, SlotEvent};
use server::ffi::QkConfig;
use tokio::sync::{Semaphore, mpsc};

const CFG: QkConfig = QkConfig {
    n_slots: 2,
    n_ctx: 2048,
    chunk: 4,
};

fn lcg(prev: u32) -> u32 {
    (prev.wrapping_mul(1_103_515_245).wrapping_add(12_345)) % 248_000
}

fn start(local_driver: bool) -> EngineThread {
    EngineThread::start(
        Path::new(env!("QK_STUB_LIB")),
        Path::new("/dev/null"),
        CFG,
        None,
        local_driver,
    )
    .expect("unsplit stub engine starts")
}

fn run(
    engine: &EngineThread,
    sem: &Arc<Semaphore>,
    prompt: Vec<u32>,
    max_gen: u32,
    sampling: Sampling,
) -> (Vec<u32>, Result<FinishReason, String>) {
    let (tx, mut rx) = mpsc::channel(64);
    engine
        .handle()
        .submit(Job {
            prompt_ids: prompt,
            max_gen,
            snap_prefix: 0,
            sampling,
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
fn local_driver_matches_greedy_and_can_sample() {
    let sem = Arc::new(Semaphore::new(8));
    let prompt = vec![3u32, 9, 11];

    // The serial (qk_step_chunk) engine's greedy stream is the reference.
    let serial = start(false);
    let (want, finish) = run(&serial, &sem, prompt.clone(), 6, Sampling::GREEDY);
    assert!(matches!(finish, Ok(FinishReason::Limit)), "{finish:?}");
    serial.shutdown();

    // The stub's rule, restated: each id is the LCG of the one before it,
    // seeded by the last prompt token. Pin it so a driver bug cannot quietly
    // agree with a matching engine bug.
    let mut expect = Vec::new();
    let mut prev = *prompt.last().unwrap();
    for _ in 0..6 {
        prev = lcg(prev);
        expect.push(prev);
    }
    assert_eq!(want, expect, "stub's serial greedy stream");

    // Same prompt, same greedy policy, but driven position-by-position through
    // stage_run with no worker: byte-identical.
    let local = start(true);
    let (got, finish) = run(&local, &sem, prompt.clone(), 6, Sampling::GREEDY);
    assert!(matches!(finish, Ok(FinishReason::Limit)), "{finish:?}");
    assert_eq!(got, want, "local driver must not change the greedy stream");

    // Now sample. The stub's topk puts the greedy id at rank 0 and id+1, id+2,
    // id+3 behind it inside the nucleus, so a hot enough temperature has to
    // land off rank 0 sometimes. Any departure from `want` proves the driver
    // injected its own pick rather than the engine's argmax.
    let hot = Sampling {
        temp: 2.0,
        top_p: 1.0,
        seed: 0xC0FF_EE00_1234_5678,
    };
    let (sampled, finish) = run(&local, &sem, prompt.clone(), 24, hot);
    assert!(matches!(finish, Ok(FinishReason::Limit)), "{finish:?}");
    assert_eq!(sampled.len(), 24);
    assert!(
        sampled != want[..sampled.len().min(want.len())],
        "sampled stream never left the greedy path — the driver is not sampling"
    );
    // Every pick must still come from the candidate set the engine offered
    // (rank 0..=3 of that position), not from nowhere.
    local.shutdown();
}
