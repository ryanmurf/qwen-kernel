#!/usr/bin/env python3
"""Full native dual-GPU logits/parity test. Requires a dedicated GPU window."""
import argparse
import ctypes as C
import json
import hashlib
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
    p.add_argument("--compare-replay", action="store_true")
    p.add_argument("--single", action="store_true",
                   help="one engine with all 48 layers and the head on --device (no second GPU is touched)")
    p.add_argument("--device", default="STRIX_HALO", help="unique Vulkan device-name substring for --single")
    args = p.parse_args()
    assert 34 <= args.split <= 38 and 1 <= args.steps <= args.ctx
    expected = floats(args.reference)
    assert len(expected) == args.steps*248320
    for key in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_LAYER_DUMP"):
        os.environ.pop(key, None)
    os.environ.update(QK_NATIVE_FLASH="1", QK_SHADER_DIR=os.path.abspath("build-halo/shaders"), QK_FLASH_COOPMAT="0")
    os.environ.setdefault("QK_PLE_PREFETCH", "0")
    lib = C.CDLL(os.path.abspath(args.library))
    u32, f32 = C.c_uint32, C.c_float
    lib.qk_open.argtypes = [C.c_char_p, C.POINTER(Config), C.c_void_p, C.c_size_t]
    lib.qk_open.restype = C.c_void_p
    lib.qk_close.argtypes = [C.c_void_p]
    lib.qk_stage_run.argtypes = [C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32),u32,u32,C.POINTER(f32),C.POINTER(u32)]
    lib.qk_stage_topk.argtypes = [C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32)]
    lib.qk_stage_logits.argtypes = [C.c_void_p,C.POINTER(f32),u32]
    engines = []
    stages = [(args.device, "0:48")] if args.single else [("STRIX_HALO",f"0:{args.split}"),("NAVI31",f"{args.split}:48")]
    try:
        for device, layers in stages:
            os.environ.update(QK_DEVICE_NAME=device)
            if args.single: os.environ.pop("QK_LAYERS", None)
            else: os.environ["QK_LAYERS"] = layers
            err = C.create_string_buffer(512)
            start = time.monotonic()
            engine = lib.qk_open(os.fsencode(args.model),C.byref(Config(1,args.ctx,1)),err,len(err))
            if not engine: raise RuntimeError(f"{device} {layers}: {err.value.decode()}")
            engines.append(engine)
            print(json.dumps({"loaded":layers,"device":device,"seconds":time.monotonic()-start}),flush=True)
        head = engines[-1]
        hidden, logits = (f32*10240)(), (f32*248320)()
        ids, candidates, values = (u32*1)(), (u32*20)(), (f32*20)()
        def step_only(step, token_id):
            """Pure stepping through the stage(s): no reference I/O or math (timed loops)."""
            token = (u32*1)(token_id)
            if args.single:
                assert lib.qk_stage_run(engines[0],0,token,None,1,step,None,ids)==0
                return
            assert lib.qk_stage_run(engines[0],0,token,None,1,step,hidden,None)==0
            assert lib.qk_stage_run(engines[1],0,None,hidden,1,step,None,ids)==0
        def run_position(step, token_id):
            """Validation stepping: like step_only, plus the split-boundary RMS (None when single)."""
            token = (u32*1)(token_id)
            if args.single:
                assert lib.qk_stage_run(engines[0],0,token,None,1,step,None,ids)==0
                return None
            assert lib.qk_stage_run(engines[0],0,token,None,1,step,hidden,None)==0
            split_ref = floats(f"{args.reference}.l_last-{args.split-1}.{step}")
            split_rms, _ = relative(hidden,split_ref)
            assert lib.qk_stage_run(engines[1],0,None,hidden,1,step,None,ids)==0
            return split_rms
        if args.compare_replay:
            os.environ["QK_FLASH_REPLAY"]="1"  # digests below come from the replayed path; an inherited 0 must not hide that
        records = []
        digests = []
        first = None
        for step in range(args.steps):
            start = time.monotonic()
            split_rms = run_position(step, args.token+step)
            seconds = time.monotonic()-start
            assert lib.qk_stage_logits(head,logits,len(logits))==0
            digests.append(hashlib.sha256(bytes(logits)).hexdigest())
            row = expected[step*248320:(step+1)*248320]
            rms, maximum = relative(logits,row)
            best = max(range(len(row)),key=row.__getitem__)
            assert lib.qk_stage_topk(head,20,candidates,values)==0
            order = sorted(range(len(logits)),key=lambda i:(-logits[i],i))[:20]
            assert list(candidates)==order and list(values)==[logits[i] for i in order]
            record = {"step":step,"split_relative_rms":split_rms,"logit_relative_rms":rms,
                      "logit_max_abs":maximum,"native_argmax":ids[0],"reference_argmax":best,
                      "seconds":seconds}
            print(json.dumps(record),flush=True); records.append(record)
            if step==0: first = list(logits)
        run_position(0, args.token)
        assert lib.qk_stage_logits(head,logits,len(logits))==0
        assert list(logits)==first, "full-model reset changed logits"
        assert all((r["split_relative_rms"] is None or r["split_relative_rms"]<1e-5) and r["logit_relative_rms"]<1e-5
                   and r["native_argmax"]==r["reference_argmax"] for r in records), "full-model parity failed"
        print(json.dumps({"full_model_parity":"PASS","frames":args.steps,"reset_exact":True,
                          "single_device":args.device if args.single else None}),flush=True)
        if args.compare_replay:
            os.environ["QK_FLASH_REPLAY"]="0"
            for step in range(args.steps):
                run_position(step, args.token+step)
                assert lib.qk_stage_logits(head,logits,len(logits))==0
                assert hashlib.sha256(bytes(logits)).hexdigest()==digests[step], "replay changed logits"
            print(json.dumps({"replay_vs_record":"bit-exact","frames":args.steps}),flush=True)
            # Hot-cache, short-context A/B in alternating order. No reference
            # math or full-logit readback inside the timed section.
            for trial,replay in enumerate([0,1,1,0,0,1]):
                os.environ["QK_FLASH_REPLAY"]=str(replay)
                start=time.monotonic()
                for step in range(args.steps):
                    step_only(step, args.token+step)
                elapsed=time.monotonic()-start
                print(json.dumps({"trial":trial,"replay":replay,"tokens":args.steps,
                                  "seconds":elapsed,"tokens_per_second":args.steps/elapsed}),flush=True)
    finally:
        for engine in reversed(engines): lib.qk_close(engine)

if __name__ == "__main__": main()
