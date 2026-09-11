#!/usr/bin/env python3
"""Token-level agreement of the batched prefill tiers (dedicated GPU window).

Greedy-decodes a model-generated sequence with the serial path, then
teacher-forces it through the serial path, the exact batched path (F32 GEMMs)
and the coopmat batched path (f16 inputs), comparing the greedy id after every
position and the final logits. It also re-runs the head one position at a time
on each tier's batched hidden rows and streams per-position KL divergence and
next-token log-probability deltas, so the reduced-precision tier gets a
measured label. This is not an F32-oracle parity test.

Memory: per-position logit rows are kept as compact float32 arrays (one
row of 248320 floats is 0.95 MiB), and the KL comparison streams one row at a
time, so RAM stays bounded by tokens * 1 MiB per tier plus the engine's own
mapped weights."""
import argparse
import ctypes as C
import json
import math
import os
import time
from array import array

VOCAB = 248320
ROW = 10240

class Config(C.Structure):
    _fields_ = [("slots", C.c_uint32), ("ctx", C.c_uint32), ("chunk", C.c_uint32)]

def relative(a, b):
    err = energy = 0.0
    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise AssertionError("nonfinite logits")
        err += (x-y)**2; energy += y*y
    return math.sqrt(err/max(energy, 1e-20))

def logsoftmax(row):
    m = max(row)
    z = math.log(sum(math.exp(v-m) for v in row))
    return [v-m-z for v in row]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--library", default="build-halo/libqk.so")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--split", type=int, default=37)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--seed-tokens", type=int, default=16, help="prompt ids 198.. seeding the generated sequence")
    args = p.parse_args()
    if not (34 <= args.split <= 38): p.error("--split must be 34..38")
    if not (2 <= args.tokens <= 512): p.error("--tokens must be 2..512 (one batch)")
    if not (1 <= args.seed_tokens <= args.tokens): p.error("--seed-tokens must be 1..tokens")
    if args.ctx < args.tokens or args.ctx > 32768: p.error("--ctx must cover --tokens and stay <= 32768")
    for key in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_LAYER_DUMP", "QK_FLASH_COOPMAT"):
        os.environ.pop(key, None)
    os.environ.update(QK_NATIVE_FLASH="1", QK_SHADER_DIR=os.path.abspath("build-halo/shaders"))
    os.environ.setdefault("QK_PLE_PREFETCH", "0")  # harness runs do not warm the tables
    lib = C.CDLL(os.path.abspath(args.library))
    u32, f32 = C.c_uint32, C.c_float
    lib.qk_open.argtypes = [C.c_char_p, C.POINTER(Config), C.c_void_p, C.c_size_t]
    lib.qk_open.restype = C.c_void_p
    lib.qk_close.argtypes = [C.c_void_p]
    lib.qk_stage_run.argtypes = [C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32),u32,u32,C.POINTER(f32),C.POINTER(u32)]
    lib.qk_stage_run.restype = C.c_int
    lib.qk_stage_logits.argtypes = [C.c_void_p,C.POINTER(f32),u32]
    lib.qk_stage_logits.restype = C.c_int
    engines = []
    try:
        for device, layers in [("STRIX_HALO",f"0:{args.split}"),("NAVI31",f"{args.split}:48")]:
            os.environ.update(QK_DEVICE_NAME=device,QK_LAYERS=layers)
            err = C.create_string_buffer(512)
            start = time.monotonic()
            engine = lib.qk_open(os.fsencode(args.model),C.byref(Config(1,args.ctx,1)),err,len(err))
            if not engine:
                raise RuntimeError(f"{device} {layers}: {err.value.decode()}")
            engines.append(engine)
            print(json.dumps({"loaded": layers, "device": device, "seconds": time.monotonic()-start}), flush=True)
        n = args.tokens
        hidden = (f32*(ROW*n))(); ids = (u32*n)(); logits = (f32*VOCAB)()
        def stage(engine, toks, hin, width, base, hout, idsout):
            rc = lib.qk_stage_run(engine, 0, toks, hin, width, base, hout, idsout)
            if rc != 0: raise RuntimeError(f"qk_stage_run rc={rc} width={width} base={base}")
        def take_logits():
            rc = lib.qk_stage_logits(engines[1], logits, VOCAB)
            if rc != 0: raise RuntimeError(f"qk_stage_logits rc={rc}")
            return array("f", logits)
        # 1. serial greedy generation from the seed -> n-token sequence, keeping
        #    the serial per-position logit rows (the reference for both tiers).
        seq = list(range(198, 198+args.seed_tokens))
        serial_rows = []
        for pos in range(n):
            tok = (u32*1)(seq[pos])
            stage(engines[0], tok, None, 1, pos, hidden, None)
            stage(engines[1], None, hidden, 1, pos, None, ids)
            serial_rows.append(take_logits())
            if pos+1 < n and pos+1 >= len(seq): seq.append(int(ids[0]))
        seq = seq[:n]
        if any(not (0 <= t < VOCAB) for t in seq): raise AssertionError("generated id out of vocabulary")
        def run(coop, chunks):
            os.environ["QK_FLASH_COOPMAT"] = "1" if coop else "0"
            done = 0; start = time.monotonic()
            for width in chunks:
                width = min(width, n-done)
                if width <= 0: break
                tok = (u32*width)(*seq[done:done+width])
                hp = C.cast(C.addressof(hidden)+done*ROW*4, C.POINTER(f32))
                stage(engines[0], tok, None, width, done, hp, None)
                stage(engines[1], None, hp, width, done, None, C.cast(C.addressof(ids)+done*4, C.POINTER(u32)))
                done += width
            if done != n: raise AssertionError("prompt not fully fed")
            return list(ids), take_logits(), time.monotonic()-start
        # 2. per-position logits of each first-stage tier (head re-run serially
        #    on the batched hidden rows): compact float32 rows, streamed KL.
        def head_rows(coop):
            os.environ["QK_FLASH_COOPMAT"] = "1" if coop else "0"
            tok = (u32*n)(*seq)
            stage(engines[0], tok, None, n, 0, hidden, None)
            rows = []
            one = (u32*1)()
            for i in range(n):
                stage(engines[1], None, C.cast(C.addressof(hidden)+i*ROW*4, C.POINTER(f32)), 1, i, None, one)
                rows.append(take_logits())
            return rows
        exact_rows = head_rows(False)
        coop_rows = head_rows(True)
        def kl_of(ps, pq): return sum(math.exp(a)*(a-b) for a, b in zip(ps, pq))
        def argmax(p): return max(range(VOCAB), key=p.__getitem__)
        kl = array("d"); dlogp = array("d"); agree_rows = 0
        kl_exact = array("d"); agree_exact_rows = 0; first_exact_divergence = None
        for i in range(n):
            ps = logsoftmax(serial_rows[i]); pe = logsoftmax(exact_rows[i]); pc_ = logsoftmax(coop_rows[i])
            kl.append(kl_of(pe, pc_))
            kl_exact.append(kl_of(ps, pe))
            if kl_exact[-1] > 1e-4 and first_exact_divergence is None: first_exact_divergence = i
            if i+1 < n: dlogp.append(pc_[seq[i+1]] - pe[seq[i+1]])
            agree_rows += int(argmax(pe) == argmax(pc_))
            agree_exact_rows += int(argmax(ps) == argmax(pe))
            serial_rows[i] = exact_rows[i] = coop_rows[i] = None  # release as we go
            ps = pe = pc_ = None
        del serial_rows, exact_rows, coop_rows
        # 3. end-to-end greedy agreement of the three paths
        serial_ids, serial_logits, serial_s = run(False, [1]*n)
        f32_ids, f32_logits, f32_s = run(False, [n])
        coop_ids, coop_logits, coop_s = run(True, [n])
        def agree(a, b): return sum(1 for x, y in zip(a, b) if x == y)
        kl_sorted = sorted(kl)
        record = {"positions": n, "seed_tokens": args.seed_tokens, "ctx": args.ctx,
                  "serial_seconds": serial_s, "batched_f32_seconds": f32_s, "batched_coopmat_seconds": coop_s,
                  "f32_vs_serial_agree": agree(f32_ids, serial_ids), "coopmat_vs_serial_agree": agree(coop_ids, serial_ids),
                  "coopmat_vs_f32_agree": agree(coop_ids, f32_ids),
                  "f32_vs_serial_final_logit_relative_rms": relative(f32_logits, serial_logits),
                  "coopmat_vs_serial_final_logit_relative_rms": relative(coop_logits, serial_logits),
                  "coopmat_mismatch_positions": [i for i in range(n) if coop_ids[i] != serial_ids[i]][:32],
                  "first_stage_kl_exact_vs_coopmat_mean_nats": sum(kl)/n, "first_stage_kl_median_nats": kl_sorted[n//2],
                  "first_stage_kl_p99_nats": kl_sorted[int(0.99*(n-1))], "first_stage_kl_max_nats": kl_sorted[-1],
                  "first_stage_next_token_mean_logprob_delta": sum(dlogp)/len(dlogp),
                  "first_stage_argmax_agree_rows": agree_rows,
                  "exact_batched_vs_serial_kl_mean_nats": sum(kl_exact)/n,
                  "exact_batched_vs_serial_kl_max_nats": max(kl_exact),
                  "exact_batched_vs_serial_first_position_kl_above_1e-4": first_exact_divergence,
                  "exact_batched_vs_serial_argmax_agree_rows": agree_exact_rows,
                  "exact_tier_batched_matches_serial": agree(f32_ids, serial_ids) == n}
        print(json.dumps(record), flush=True)
        if agree(f32_ids, serial_ids) != n:
            raise SystemExit("exact batched tier disagreed with serial: investigate before trusting the reduced tier numbers")
    finally:
        for engine in reversed(engines): lib.qk_close(engine)

if __name__ == "__main__": main()
