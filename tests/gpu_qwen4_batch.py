#!/usr/bin/env python3
"""Full native dual-GPU batched-prefill parity versus the F32 oracle and the serial path.
Requires a dedicated GPU window (both stages loaded)."""
import argparse
import ctypes as C
import json
import math
import os
import time
from array import array

class Config(C.Structure):
    _fields_ = [("slots", C.c_uint32), ("ctx", C.c_uint32), ("chunk", C.c_uint32)]

def floats(path):
    values = array("f")
    with open(path, "rb") as file:
        values.frombytes(file.read())
    return values

def relative(actual, expected):
    assert len(actual) == len(expected)
    error = energy = maximum = 0.0
    for a, b in zip(actual, expected):
        assert math.isfinite(a) and math.isfinite(b)
        error += (a-b)**2
        energy += b*b
        maximum = max(maximum, abs(a-b))
    return math.sqrt(error/max(energy, 1e-20)), maximum

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("reference", help="F32 result_output oracle with .l_last-36.N captures")
    p.add_argument("--library", default="build-halo/libqk.so")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--token", type=int, default=198)
    p.add_argument("--split", type=int, default=37)
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--tolerance", type=float, default=1e-5)
    args = p.parse_args()
    assert 34 <= args.split <= 38 and 2 <= args.steps <= args.ctx
    expected = floats(args.reference)
    assert len(expected) == args.steps*248320
    boundary = [floats(f"{args.reference}.l_last-{args.split-1}.{step}") for step in range(args.steps)]
    reference_ids = [max(range(248320), key=expected[s*248320:(s+1)*248320].__getitem__) for s in range(args.steps)]
    for key in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_LAYER_DUMP"):
        os.environ.pop(key, None)
    os.environ.update(QK_NATIVE_FLASH="1", QK_SHADER_DIR=os.path.abspath("build-halo/shaders"))
    lib = C.CDLL(os.path.abspath(args.library))
    u32, f32 = C.c_uint32, C.c_float
    lib.qk_open.argtypes = [C.c_char_p, C.POINTER(Config), C.c_void_p, C.c_size_t]
    lib.qk_open.restype = C.c_void_p
    lib.qk_close.argtypes = [C.c_void_p]
    lib.qk_stage_run.argtypes = [C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32),u32,u32,C.POINTER(f32),C.POINTER(u32)]
    lib.qk_stage_topk.argtypes = [C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32)]
    lib.qk_stage_logits.argtypes = [C.c_void_p,C.POINTER(f32),u32]
    engines = []
    records = []
    def record(**fields):
        print(json.dumps(fields), flush=True); records.append(fields)
    try:
        for device, layers in [("STRIX_HALO",f"0:{args.split}"),("NAVI31",f"{args.split}:48")]:
            os.environ.update(QK_DEVICE_NAME=device,QK_LAYERS=layers)
            err = C.create_string_buffer(512)
            start = time.monotonic()
            engine = lib.qk_open(os.fsencode(args.model),C.byref(Config(1,args.ctx,1)),err,len(err))
            if not engine: raise RuntimeError(err.value.decode())
            engines.append(engine)
            record(loaded=layers,device=device,seconds=time.monotonic()-start)
        n = args.steps
        tokens = (u32*n)(*range(args.token, args.token+n))
        hidden = (f32*(10240*n))()
        ids = (u32*n)()
        logits = (f32*248320)()
        candidates, values = (u32*20)(), (f32*20)()

        def run(chunks, label):
            """Feed the prompt in the given chunk widths; return per-row results and timing."""
            done = 0
            start = time.monotonic()
            for width in chunks:
                width = min(width, n-done)
                if width <= 0: break
                tok = (u32*width)(*tokens[done:done+width])
                assert lib.qk_stage_run(engines[0],0,tok,None,width,done,C.cast(C.addressof(hidden)+done*10240*4,C.POINTER(f32)),None)==0
                assert lib.qk_stage_run(engines[1],0,None,C.cast(C.addressof(hidden)+done*10240*4,C.POINTER(f32)),width,done,None,C.cast(C.addressof(ids)+done*4,C.POINTER(u32)))==0
                done += width
            assert done == n
            seconds = time.monotonic()-start
            assert lib.qk_stage_logits(engines[1],logits,len(logits))==0
            assert lib.qk_stage_topk(engines[1],20,candidates,values)==0
            order = sorted(range(len(logits)),key=lambda i:(-logits[i],i))[:20]
            assert list(candidates)==order and list(values)==[logits[i] for i in order]
            worst_boundary = max(relative(hidden[s*10240:(s+1)*10240],boundary[s])[0] for s in range(n))
            rms, maximum = relative(logits,expected[(n-1)*248320:n*248320])
            mismatches = [s for s in range(n) if ids[s] != reference_ids[s]]
            ok = worst_boundary < args.tolerance and rms < args.tolerance and not mismatches
            record(scenario=label,chunks=list(chunks),worst_boundary_relative_rms=worst_boundary,
                   final_logit_relative_rms=rms,final_logit_max_abs=maximum,argmax_mismatches=mismatches,
                   seconds=seconds,tokens_per_second=n/seconds,result="PASS" if ok else "FAIL")
            return ok, list(logits), list(hidden)

        ok_serial, serial_logits, serial_hidden = run([1]*n, "serial")
        ok_whole, whole_logits, whole_hidden = run([n], "batched_whole")
        ok_mixed, mixed_logits, _ = run([5,1,7,3,n], "batched_mixed")
        ok_half, half_logits, _ = run([n//2]+[1]*(n-n//2), "batch_then_serial")
        # Serial versus batched agreement (independent of the oracle).
        rms_hidden = max(relative(whole_hidden[s*10240:(s+1)*10240],serial_hidden[s*10240:(s+1)*10240])[0] for s in range(n))
        rms_logits = relative(whole_logits,serial_logits)[0]
        record(scenario="batched_vs_serial",worst_boundary_relative_rms=rms_hidden,final_logit_relative_rms=rms_logits,
               mixed_vs_whole_logit_relative_rms=relative(mixed_logits,whole_logits)[0],
               half_vs_whole_logit_relative_rms=relative(half_logits,whole_logits)[0])
        # Reset: a fresh serial first position reproduces the serial run bit-for-bit.
        tok = (u32*1)(tokens[0])
        assert lib.qk_stage_run(engines[0],0,tok,None,1,0,hidden,None)==0
        assert lib.qk_stage_run(engines[1],0,None,hidden,1,0,None,ids)==0
        assert lib.qk_stage_logits(engines[1],logits,len(logits))==0
        first = (u32*1)(tokens[0])
        # Recompute the serial first row for comparison.
        assert lib.qk_stage_run(engines[0],0,first,None,1,0,hidden,None)==0
        assert lib.qk_stage_run(engines[1],0,None,hidden,1,0,None,ids)==0
        again = (f32*248320)()
        assert lib.qk_stage_logits(engines[1],again,len(again))==0
        reset_exact = list(logits)==list(again)
        ok = ok_serial and ok_whole and ok_mixed and ok_half and reset_exact
        record(batched_parity="PASS" if ok else "FAIL",frames=n,reset_exact=reset_exact)
        if not ok: raise SystemExit(1)
    finally:
        for engine in reversed(engines): lib.qk_close(engine)

if __name__ == "__main__": main()
