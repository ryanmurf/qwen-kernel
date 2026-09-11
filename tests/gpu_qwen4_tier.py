#!/usr/bin/env python3
"""Token-level agreement of the batched prefill tiers (dedicated GPU window).

Greedy-decodes a model-generated 256-token sequence with the serial path, then
teacher-forces it through the serial path, the F32 batched path and the
coopmat (f16-input) batched path, comparing the greedy id after every position
and the final logits. This labels the reduced-precision tier with a measured
agreement rate; it is not an F32-oracle parity test."""
import argparse
import ctypes as C
import json
import math
import os
import time

class Config(C.Structure):
    _fields_ = [("slots", C.c_uint32), ("ctx", C.c_uint32), ("chunk", C.c_uint32)]

def relative(a, b):
    err = energy = 0.0
    for x, y in zip(a, b):
        assert math.isfinite(x) and math.isfinite(y)
        err += (x-y)**2; energy += y*y
    return math.sqrt(err/max(energy, 1e-20))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--library", default="build-halo/libqk.so")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--split", type=int, default=37)
    args = p.parse_args()
    for key in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_LAYER_DUMP", "QK_FLASH_COOPMAT"):
        os.environ.pop(key, None)
    os.environ.update(QK_NATIVE_FLASH="1", QK_SHADER_DIR=os.path.abspath("build-halo/shaders"))
    lib = C.CDLL(os.path.abspath(args.library))
    u32, f32 = C.c_uint32, C.c_float
    lib.qk_open.argtypes = [C.c_char_p, C.POINTER(Config), C.c_void_p, C.c_size_t]
    lib.qk_open.restype = C.c_void_p
    lib.qk_close.argtypes = [C.c_void_p]
    lib.qk_stage_run.argtypes = [C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32),u32,u32,C.POINTER(f32),C.POINTER(u32)]
    lib.qk_stage_logits.argtypes = [C.c_void_p,C.POINTER(f32),u32]
    engines = []
    try:
        for device, layers in [("STRIX_HALO",f"0:{args.split}"),("NAVI31",f"{args.split}:48")]:
            os.environ.update(QK_DEVICE_NAME=device,QK_LAYERS=layers)
            err = C.create_string_buffer(512)
            engine = lib.qk_open(os.fsencode(args.model),C.byref(Config(1,1024,1)),err,len(err))
            if not engine: raise RuntimeError(err.value.decode())
            engines.append(engine)
        n = args.tokens
        hidden = (f32*(10240*n))(); ids = (u32*n)(); logits = (f32*248320)()
        # 1. serial greedy generation from a 16-token seed -> sequence of n tokens
        seq = list(range(198, 214))
        for pos in range(n):
            tok = (u32*1)(seq[pos])
            assert lib.qk_stage_run(engines[0],0,tok,None,1,pos,hidden,None)==0
            assert lib.qk_stage_run(engines[1],0,None,hidden,1,pos,None,ids)==0
            if pos+1 < n and pos+1 >= len(seq): seq.append(int(ids[0]))
        assert len(seq) >= n
        seq = seq[:n]
        prompt = (u32*n)(*seq)
        def run(label, coop, chunks):
            os.environ["QK_FLASH_COOPMAT"] = "1" if coop else "0"
            out = []; done = 0; start = time.monotonic()
            for width in chunks:
                width = min(width, n-done)
                if width <= 0: break
                tok = (u32*width)(*seq[done:done+width])
                assert lib.qk_stage_run(engines[0],0,tok,None,width,done,C.cast(C.addressof(hidden)+done*10240*4,C.POINTER(f32)),None)==0
                assert lib.qk_stage_run(engines[1],0,None,C.cast(C.addressof(hidden)+done*10240*4,C.POINTER(f32)),width,done,None,C.cast(C.addressof(ids)+done*4,C.POINTER(u32)))==0
                done += width
            seconds = time.monotonic()-start
            assert lib.qk_stage_logits(engines[1],logits,len(logits))==0
            return list(ids), list(logits), seconds
        serial_ids, serial_logits, serial_s = run("serial", False, [1]*n)
        # Per-position logits of the batched Halo tiers: the head stage is re-run
        # one position at a time (serial F32 GEMVs) on the batched hidden rows,
        # so the comparison isolates the first stage's tier.
        def head_rows(coop):
            os.environ["QK_FLASH_COOPMAT"] = "1" if coop else "0"
            tok = (u32*n)(*seq)
            assert lib.qk_stage_run(engines[0],0,tok,None,n,0,hidden,None)==0
            rows = []
            one = (u32*1)()
            for i in range(n):
                assert lib.qk_stage_run(engines[1],0,None,C.cast(C.addressof(hidden)+i*10240*4,C.POINTER(f32)),1,i,None,one)==0
                assert lib.qk_stage_logits(engines[1],logits,len(logits))==0
                rows.append(list(logits))
            return rows
        def logsoftmax(row):
            m = max(row); z = math.log(sum(math.exp(v-m) for v in row))
            return [v-m-z for v in row]
        exact_rows = head_rows(False)
        coop_rows = head_rows(True)
        kl = []; dlogp = []; agree_rows = 0
        for i in range(n):
            pe = logsoftmax(exact_rows[i]); pcp = logsoftmax(coop_rows[i])
            kl.append(sum(math.exp(a)*(a-b) for a, b in zip(pe, pcp)))
            if i+1 < n: dlogp.append(pcp[seq[i+1]] - pe[seq[i+1]])
            agree_rows += int(max(range(len(pe)), key=pe.__getitem__) == max(range(len(pcp)), key=pcp.__getitem__))
        f32_ids, f32_logits, f32_s = run("batched_f32", False, [n])
        coop_ids, coop_logits, coop_s = run("batched_coopmat", True, [n])
        def agree(a, b): return sum(1 for x, y in zip(a, b) if x == y)
        kl_sorted = sorted(kl)
        record = {"positions": n, "seed_tokens": 16,
                  "serial_seconds": serial_s, "batched_f32_seconds": f32_s, "batched_coopmat_seconds": coop_s,
                  "f32_vs_serial_agree": agree(f32_ids, serial_ids), "coopmat_vs_serial_agree": agree(coop_ids, serial_ids),
                  "coopmat_vs_f32_agree": agree(coop_ids, f32_ids),
                  "f32_vs_serial_final_logit_relative_rms": relative(f32_logits, serial_logits),
                  "coopmat_vs_serial_final_logit_relative_rms": relative(coop_logits, serial_logits),
                  "coopmat_mismatch_positions": [i for i in range(n) if coop_ids[i] != serial_ids[i]][:32],
                  "first_stage_kl_exact_vs_coopmat_mean_nats": sum(kl)/n, "first_stage_kl_median_nats": kl_sorted[n//2],
                  "first_stage_kl_p99_nats": kl_sorted[int(0.99*(n-1))], "first_stage_kl_max_nats": kl_sorted[-1],
                  "first_stage_next_token_mean_logprob_delta": sum(dlogp)/len(dlogp),
                  "first_stage_argmax_agree_rows": agree_rows}
        print(json.dumps(record), flush=True)
    finally:
        for engine in reversed(engines): lib.qk_close(engine)

if __name__ == "__main__": main()
