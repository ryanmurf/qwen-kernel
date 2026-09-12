#!/usr/bin/env python3
"""Cross-build teacher-forced comparison on the single device (dedicated GPU window).

`dump` mode loads one engine (all layers and the head on --device), feeds a
fixed deterministic token sequence one position at a time and writes every
position's logit row (float32, 248320 per row) to --out. Run it once per
kernel configuration (for example with QK_FLASH_FUSE=0 QK_MOE_GU=v1 as the
control and with the defaults), then `compare` reports per-position KL
divergence (control || candidate), argmax agreement and the largest relative
RMS between the two dumps. The sequence is ids 1000 + 7*i mod 248320 for i in
0..n-1 (all in range, no EOS), so both dumps see identical inputs; this is a
kernel-change check, not an oracle parity test."""
import argparse
import ctypes as C
import json
import math
import os
import time
from array import array

VOCAB = 248320

class Config(C.Structure):
    _fields_ = [("slots", C.c_uint32), ("ctx", C.c_uint32), ("chunk", C.c_uint32)]

def sequence(n):
    return [(1000 + 7*i) % VOCAB for i in range(n)]

def dump(args):
    if not (1 <= args.tokens <= 4096 and args.tokens <= args.ctx <= 32768): raise SystemExit("tokens 1..4096 and tokens <= ctx <= 32768")
    for key in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_LAYER_DUMP", "QK_LAYERS"): os.environ.pop(key, None)
    os.environ.update(QK_NATIVE_FLASH="1", QK_SHADER_DIR=os.path.abspath("build-halo/shaders"), QK_DEVICE_NAME=args.device)
    os.environ.setdefault("QK_PLE_PREFETCH", "0"); os.environ.setdefault("QK_FLASH_COOPMAT", "0")
    lib = C.CDLL(os.path.abspath(args.library))
    u32, f32 = C.c_uint32, C.c_float
    lib.qk_open.argtypes = [C.c_char_p, C.POINTER(Config), C.c_void_p, C.c_size_t]; lib.qk_open.restype = C.c_void_p
    lib.qk_close.argtypes = [C.c_void_p]
    lib.qk_stage_run.argtypes = [C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32),u32,u32,C.POINTER(f32),C.POINTER(u32)]
    lib.qk_stage_logits.argtypes = [C.c_void_p,C.POINTER(f32),u32]
    err = C.create_string_buffer(512); t0 = time.monotonic()
    e = lib.qk_open(os.fsencode(args.model), C.byref(Config(1, args.ctx, 1)), err, len(err))
    if not e: raise RuntimeError(err.value.decode())
    print(json.dumps({"loaded": "0:48", "device": args.device, "seconds": time.monotonic()-t0,
                      "config": {k: os.environ.get(k) for k in ("QK_FLASH_FUSE", "QK_MOE_GU", "QK_GDN_STEP", "QK_PLE_ROW_PREFETCH", "QK_FLASH_COOPMAT")}}), flush=True)
    try:
        seq = sequence(args.tokens)
        logits = (f32*VOCAB)(); one = (u32*1)()
        with open(args.out, "wb") as f:
            t0 = time.monotonic()
            for pos, tok in enumerate(seq):
                t = (u32*1)(tok)
                rc = lib.qk_stage_run(e, 0, t, None, 1, pos, None, one)
                if rc != 0: raise RuntimeError(f"stage_run rc={rc} at position {pos}")
                if lib.qk_stage_logits(e, logits, VOCAB) != 0: raise RuntimeError("stage_logits failed")
                f.write(bytes(logits))
            seconds = time.monotonic()-t0
        print(json.dumps({"dumped": args.tokens, "out": args.out, "serial_seconds": seconds, "tokens_per_second": args.tokens/seconds}), flush=True)
    finally:
        lib.qk_close(e)

def logsoftmax(row):
    m = max(row); z = math.log(sum(math.exp(v-m) for v in row))
    return [v-m-z for v in row]

def compare(args):
    size = os.path.getsize(args.control)
    if size != os.path.getsize(args.candidate) or size % (VOCAB*4): raise SystemExit("dumps differ in size or are not whole rows")
    n = size // (VOCAB*4)
    kl = array("d"); rms = array("d"); agree = 0; first_div = None
    with open(args.control, "rb") as fa, open(args.candidate, "rb") as fb:
        for i in range(n):
            a = array("f"); a.frombytes(fa.read(VOCAB*4)); b = array("f"); b.frombytes(fb.read(VOCAB*4))
            if any(not math.isfinite(v) for v in a) or any(not math.isfinite(v) for v in b): raise SystemExit(f"nonfinite logits at position {i}")
            pa, pb = logsoftmax(a), logsoftmax(b)
            kl.append(sum(math.exp(x)*(x-y) for x, y in zip(pa, pb)))
            err = sum((x-y)**2 for x, y in zip(a, b)); energy = sum(y*y for y in a)
            rms.append(math.sqrt(err/max(energy, 1e-20)))
            ia = max(range(VOCAB), key=a.__getitem__); ib = max(range(VOCAB), key=b.__getitem__)
            agree += int(ia == ib)
            if first_div is None and kl[-1] > 1e-4: first_div = i
    ks = sorted(kl)
    print(json.dumps({"positions": n, "argmax_agree": agree, "kl_mean_nats": sum(kl)/n, "kl_median_nats": ks[n//2],
                      "kl_p99_nats": ks[int(0.99*(n-1))], "kl_max_nats": ks[-1], "relative_rms_max": max(rms),
                      "first_position_kl_above_1e-4": first_div, "control": args.control, "candidate": args.candidate}), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("model"); d.add_argument("--out", required=True)
    d.add_argument("--tokens", type=int, default=256); d.add_argument("--ctx", type=int, default=1024)
    d.add_argument("--device", default="STRIX_HALO"); d.add_argument("--library", default="build-halo/libqk.so")
    c = sub.add_parser("compare"); c.add_argument("control"); c.add_argument("candidate")
    args = p.parse_args()
    dump(args) if args.cmd == "dump" else compare(args)

if __name__ == "__main__": main()
