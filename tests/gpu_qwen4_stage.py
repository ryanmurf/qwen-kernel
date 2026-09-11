#!/usr/bin/env python3
"""Actual native stage-ABI parity, bounds and reset checks on a bounded prefix."""
import argparse
import ctypes as C
import json
import math
import os
from array import array
import urllib.request


class Config(C.Structure):
    _fields_ = [("slots", C.c_uint32), ("ctx", C.c_uint32), ("chunk", C.c_uint32)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("reference", help="16-frame F32 l_last-1 oracle, starting at token 198")
    p.add_argument("--library", default="build-halo/libqk.so")
    p.add_argument("--device", default="STRIX_HALO")
    p.add_argument("--layers", choices=["0:2", "2:4"], default="0:2")
    p.add_argument("--input", help="16-frame first-stage residual oracle, required for 2:4")
    p.add_argument("--status-url", default="http://127.0.0.1:8091/handoff/status")
    args = p.parse_args()
    if args.status_url:
        with urllib.request.urlopen(args.status_url, timeout=5) as response:
            if json.load(response)["slots"]["busy"]:
                raise SystemExit("Serving request active; stopping stage tests")
    expected = array("f")
    with open(args.reference, "rb") as file:
        expected.frombytes(file.read())
    assert len(expected) == 16 * 10240
    inputs = array("f")
    if args.layers == "2:4":
        if not args.input:
            p.error("2:4 requires --input residuals from layers 0:2")
        with open(args.input, "rb") as file:
            inputs.frombytes(file.read())
        assert len(inputs) == len(expected)
    for name in ("QK_DEVICE_PCI", "QK_DEVICE", "QK_LAYER_DUMP"):
        os.environ.pop(name, None)
    os.environ.update(QK_NATIVE_FLASH="1", QK_LAYERS=args.layers, QK_DEVICE_NAME=args.device,
                      QK_SHADER_DIR=os.path.abspath("build-halo/shaders"))
    lib = C.CDLL(os.path.abspath(args.library))
    u32, f32 = C.c_uint32, C.c_float
    lib.qk_open.argtypes = [C.c_char_p, C.POINTER(Config), C.c_void_p, C.c_size_t]
    lib.qk_open.restype = C.c_void_p
    lib.qk_close.argtypes = [C.c_void_p]
    for name in ("qk_n_embd", "qk_n_layer", "qk_state_n"):
        fn = getattr(lib, name); fn.argtypes = [C.c_void_p]; fn.restype = u32
    lib.qk_stage_run.argtypes = [C.c_void_p, u32, C.POINTER(u32), C.POINTER(f32), u32, u32,
                                C.POINTER(f32), C.POINTER(u32)]
    lib.qk_stage_run.restype = C.c_int
    err = C.create_string_buffer(512)
    cfg = Config(1, 64, 1)
    engine = lib.qk_open(os.fsencode(args.model), C.byref(cfg), err, len(err))
    if not engine:
        raise RuntimeError(err.value.decode())
    try:
        assert lib.qk_n_embd(engine) == 10240 and lib.qk_n_layer(engine) == 48
        assert lib.qk_state_n(engine) == 0  # unimplemented snapshots stay disabled
        def run(chunks):
            out = []
            base = 0
            for count in chunks:
                tokens = (u32 * count)(*range(198 + base, 198 + base + count))
                residual = (f32 * (10240 * count))(*inputs[base*10240:(base+count)*10240]) if inputs else None
                hidden = (f32 * (10240 * count))()
                rc = lib.qk_stage_run(engine, 0, None if inputs else tokens, residual, count, base, hidden, None)
                assert rc == 0, rc
                out.extend(hidden); base += count
            return out
        whole = run([16])
        chunked = run([5, 1, 7, 3])
        assert whole == chunked, "chunk boundary/reset changed the native result"
        energy = error = 0.0
        for a, b in zip(whole, expected):
            assert math.isfinite(a) and math.isfinite(b)
            error += (a-b)**2; energy += b*b
        relative = math.sqrt(error / max(energy, 1e-20))
        assert relative < 1e-5, relative
        bad = (u32 * 1)(248320)
        hidden = (f32 * 10240)()
        if not inputs:
            assert lib.qk_stage_run(engine, 0, bad, None, 1, 0, hidden, None) < 0
        residual = (f32 * 10240)(*inputs[:10240]) if inputs else None
        valid = (u32 * 1)(198)
        tokens = None if inputs else valid
        assert lib.qk_stage_run(engine, 1, tokens, residual, 1, 0, hidden, None) < 0
        assert lib.qk_stage_run(engine, 0, tokens, residual, 1, 63, hidden, None) < 0
        assert lib.qk_stage_run(engine, 0, tokens, residual, 2, 63, hidden, None) < 0
        if inputs:
            residual[0] = float("nan")
            assert lib.qk_stage_run(engine, 0, None, residual, 1, 0, hidden, None) < 0
        assert run([16]) == whole
        print(json.dumps({"device": args.device, "layers": args.layers, "frames": 16,
                          "relative_rms": relative, "chunk_reset_exact": True,
                          "invalid_input_slot_position_bounds": "rejected"}), flush=True)
    finally:
        lib.qk_close(engine)


if __name__ == "__main__":
    main()
