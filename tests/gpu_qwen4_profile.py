#!/usr/bin/env python3
"""Per-dispatch GPU-time profile of the single-device native Flash decode (dedicated GPU window).

Loads one engine with all layers and the head on --device, feeds a distinct
64-token prompt (ids 1000..1063) as one batch, then runs --steps greedy decode
positions (the argmax token is fed back; EOS is not treated specially, so the
workload is deterministic for a given build) with QK_FLASH_PROFILE=1 and
QK_FLASH_REPLAY=0. With profiling on, libqk fences and timestamps EVERY
dispatch, so each row below is exclusive GPU time for that shader; the fenced
run loses the small overlap that unfenced projections get in production, so
the per-token total runs slightly above the production total. The last
--steps profiles are averaged into ms per token, share, dispatch count and
microseconds per dispatch. Host phases (QK_FLASH_TIMING=1) are printed by the
library every 16 tokens. Keep production runs with profiling OFF."""
import argparse, ctypes as C, json, os, re, subprocess, sys, time

class Config(C.Structure):
    _fields_ = [("slots", C.c_uint32), ("ctx", C.c_uint32), ("chunk", C.c_uint32)]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--library", default="build-halo/libqk.so")
    p.add_argument("--device", default="STRIX_HALO")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not (4 <= args.steps <= 256 and 128 <= args.ctx <= 32768): p.error("steps 4..256, ctx 128..32768")
    if 64 + args.steps > args.ctx: p.error("--ctx must be at least 64 + --steps (prompt plus decode positions)")
    if not args.inner:
        # Re-exec with the profiling environment so stderr can be captured and parsed.
        env = dict(os.environ, QK_FLASH_PROFILE="1", QK_FLASH_REPLAY="0", QK_FLASH_TIMING="1",
                   QK_NATIVE_FLASH="1", QK_PLE_PREFETCH="0", QK_FLASH_COOPMAT="0",
                   QK_SHADER_DIR=os.path.abspath("build-halo/shaders"), QK_DEVICE_NAME=args.device)
        for k in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_LAYER_DUMP", "QK_LAYERS"): env.pop(k, None)
        proc = subprocess.run([sys.executable, __file__, *sys.argv[1:], "--inner"], env=env,
                              capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-3000:]); raise SystemExit(proc.returncode)
        # Parse the per-token profiles: blocks start with "[flash profile] serial forward, 1 token: X ms GPU"
        blocks = re.split(r"\[flash profile\] serial forward, 1 token: ", proc.stderr)[1:]
        blocks = blocks[-args.steps:]
        totals, agg = [], {}
        for b in blocks:
            head = b.split("\n", 1)[0]
            totals.append(float(head.split(" ms")[0]))
            for m in re.finditer(r"\[flash profile\]\s+(\S+)\s+([0-9.]+) ms\s+[0-9.]+% \((\d+) dispatches", b):
                name, ms, n = m.group(1), float(m.group(2)), int(m.group(3))
                a = agg.setdefault(name, [0.0, 0]); a[0] += ms; a[1] += n
        if len(blocks) != args.steps:
            raise SystemExit(f"captured {len(blocks)} per-token profiles, expected exactly {args.steps}")
        if not agg: raise SystemExit("no per-shader rows parsed from the profile output")
        k = len(blocks)
        rows = sorted(((v[0]/k, v[1]//k, name) for name, v in agg.items()), reverse=True)
        total = sum(totals)/k
        print(json.dumps({"device": args.device, "profiled_tokens": k, "gpu_ms_per_token_mean": total,
                          "gpu_ms_per_token_min": min(totals), "gpu_ms_per_token_max": max(totals),
                          "attribution": "exclusive per dispatch (every dispatch fenced while profiling)"}))
        print(f"{'shader':36s} {'ms/token':>9s} {'share':>6s} {'dispatches':>10s} {'us/dispatch':>11s}")
        for ms, n, name in rows:
            print(f"{name:36s} {ms:9.3f} {100*ms/total:5.1f}% {n:10d} {1000*ms/max(n,1):11.1f}")
        timing = [l for l in proc.stderr.splitlines() if "[flash timing]" in l]
        for l in timing[-2:]: print(l)
        return
    lib = C.CDLL(os.path.abspath(args.library))
    u32, f32 = C.c_uint32, C.c_float
    lib.qk_open.argtypes = [C.c_char_p, C.POINTER(Config), C.c_void_p, C.c_size_t]; lib.qk_open.restype = C.c_void_p
    lib.qk_close.argtypes = [C.c_void_p]
    lib.qk_stage_run.argtypes = [C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32),u32,u32,C.POINTER(f32),C.POINTER(u32)]
    err = C.create_string_buffer(512)
    t0 = time.monotonic()
    e = lib.qk_open(os.fsencode(args.model), C.byref(Config(1, args.ctx, 1)), err, len(err))
    if not e: raise RuntimeError(err.value.decode())
    print(json.dumps({"loaded": "0:48", "device": args.device, "seconds": time.monotonic()-t0}), flush=True)
    try:
        n = 64
        prompt = (u32*n)(*range(1000, 1000+n)); ids = (u32*n)()
        assert lib.qk_stage_run(e, 0, prompt, None, n, 0, None, ids) == 0
        tok = int(ids[n-1]); one = (u32*1)()
        wall = []; generated = []
        for step in range(args.steps):
            t = (u32*1)(tok); a = time.monotonic()
            assert lib.qk_stage_run(e, 0, t, None, 1, n+step, None, one) == 0
            wall.append(time.monotonic()-a); generated.append(tok); tok = int(one[0])
        wall.sort()
        print(json.dumps({"decode_steps": args.steps, "wall_ms_median": 1000*wall[len(wall)//2], "wall_ms_min": 1000*wall[0],
                          "prompt_ids": "1000..1063", "decoded_input_ids": generated, "continuation": "greedy argmax, EOS not stopped"}), flush=True)
    finally:
        lib.qk_close(e)

if __name__ == "__main__": main()
