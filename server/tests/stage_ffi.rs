//! Pipeline-split stage ABI against the deterministic stub: a 0:20 + 20:40
//! stage chain must reproduce the stub's serial LCG stream, and split engines
//! must reject the serial decode API. (Single #[test] in this binary — it
//! mutates QK_LAYERS, which is process-global; integration test binaries run
//! as separate processes, so other suites are unaffected.)

use std::path::Path;

use server::ffi::{Engine, QkConfig};

const CFG: QkConfig = QkConfig {
    n_slots: 2,
    n_ctx: 64,
    chunk: 4,
};

fn open_stub() -> Engine {
    Engine::open(Path::new(env!("QK_STUB_LIB")), Path::new("/dev/null"), CFG)
        .expect("stub engine opens")
}

fn lcg(prev: u32) -> u32 {
    (prev.wrapping_mul(1_103_515_245).wrapping_add(12_345)) % 248_000
}

#[test]
fn stage_chain_matches_serial_rule() {
    // Unsplit: stage ABI exported, not split, ids come straight from toks.
    let mut full = open_stub();
    let info = full.stage_info().expect("stub exports the stage ABI");
    assert_eq!(
        (info.first, info.end, info.n_layer, info.n_embd),
        (0, 40, 40, 4)
    );
    assert!(info.is_first() && info.is_last() && !info.is_split());
    let prompt = [3u32, 9, 11];
    let mut ids = [0u32; 3];
    full.stage_run(0, Some(&prompt), None, 0, None, Some(&mut ids))
        .unwrap();
    assert_eq!(ids[2], lcg(11));

    // Mis-sized / mis-shaped calls are rejected before reaching C.
    let mut short = [0u32; 2];
    assert!(
        full.stage_run(0, Some(&prompt), None, 0, None, Some(&mut short))
            .is_err()
    );
    assert!(full.stage_run(0, None, None, 0, None, Some(&mut ids)).is_err());

    // Split pair: layers [0,20) then [20,40).
    unsafe { std::env::set_var("QK_LAYERS", "0:20") };
    let mut s1 = open_stub();
    unsafe { std::env::set_var("QK_LAYERS", "20:40") };
    let mut s2 = open_stub();
    unsafe { std::env::remove_var("QK_LAYERS") };
    let i1 = s1.stage_info().unwrap();
    let i2 = s2.stage_info().unwrap();
    assert!(i1.is_first() && !i1.is_last() && i1.is_split());
    assert!(!i2.is_first() && i2.is_last() && i2.is_split());

    // The serial decode API is rejected on split engines (rc -5).
    assert!(s1.slot_start(0, &prompt, 8, 0).is_err());

    // Prefill through the chain, then a short autoregressive decode; every id
    // must follow the stub's serial LCG rule.
    let row = i1.n_embd as usize;
    let mut hid = vec![0f32; prompt.len() * row];
    s1.stage_run(0, Some(&prompt), None, 0, Some(&mut hid), None)
        .unwrap();
    let mut pids = vec![0u32; prompt.len()];
    s2.stage_run(0, None, Some(&hid), 0, None, Some(&mut pids))
        .unwrap();
    let mut expect = lcg(11);
    assert_eq!(pids[prompt.len() - 1], expect);
    let mut next = expect;
    for pos in prompt.len() as u32..prompt.len() as u32 + 5 {
        let tok = [next];
        let mut h = vec![0f32; row];
        s1.stage_run(0, Some(&tok), None, pos, Some(&mut h), None)
            .unwrap();
        let mut id = [0u32];
        s2.stage_run(0, None, Some(&h), pos, None, Some(&mut id))
            .unwrap();
        expect = lcg(expect);
        assert_eq!(id[0], expect);
        next = id[0];
    }
}
