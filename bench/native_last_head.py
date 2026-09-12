#!/usr/bin/env python3
"""Bounded native Flash last-output gate: layers 46:48 + head, synthetic input.

This is NOT full-model inference or end-to-end serving performance. Run in a
dedicated cgroup with MemoryHigh=8G, MemoryMax=10G, MemorySwapMax=256M; only one
GPU engine may be active. No cache cleanup, device reset or system tuning.
JSONL evidence goes to stdout, native diagnostics to stderr. Defaults off in
serving; this tool explicitly exercises both C ABIs in the SAME library.
"""
import argparse
from array import array
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

GIB = 1 << 30
U = C.c_uint32
F = C.c_float
P = C.c_void_p
JSON_OUT = sys.stdout


class Config(C.Structure):
    _fields_ = [("slots", U), ("ctx", U), ("chunk", U)]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def emit(**row):
    print(json.dumps(row, allow_nan=False), file=JSON_OUT, flush=True)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def cgroup():
    path = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split("::")[1].lstrip("/")
    return {name: (path / name).read_text().strip() for name in
            ("memory.high", "memory.max", "memory.swap.max", "memory.peak", "memory.events")}


def admission():
    expected = {"QK_NATIVE_FLASH": "1", "QK_LAYERS": "46:48", "QK_FLASH_BATCH": "512",
                "QK_DEVICE_PCI": "0000:c1:00.0", "QK_FLASH_COOPMAT": "0",
                "QK_FLASH_GEMM": "baseline", "QK_FLASH_ATTN_BATCH": "baseline",
                "QK_ATTN_DECODE": "serial", "QK_PLE_PREFETCH": "0"}
    require(all(os.environ.get(k) == v for k, v in expected.items()), "explicit partial-model environment required")
    allowed = set(expected) | {"QK_SHADER_DIR", "QK_FLASH_PROFILE"}
    require(not {k for k in os.environ if k.startswith("QK_")} - allowed, "unexpected QK override")
    require(mem_available() >= 12 * GIB, "partial-model load requires 12 GiB available; full-model guard is unchanged")
    limits = cgroup()
    require(int(limits["memory.high"]) <= 8 * GIB and int(limits["memory.max"]) <= 10 * GIB
            and int(limits["memory.swap.max"]) <= GIB // 4, "dedicated bounded cgroup required")
    gpu = {}
    for pci in ("0000:c1:00.0", "0000:68:00.0"):
        path = Path("/sys/bus/pci/devices") / pci
        gpu[pci] = {name: int((path / name).read_text().strip(), 0) for name in
                    ("vendor", "device", "mem_info_gtt_used", "mem_info_vram_used")}
        require(gpu[pci]["mem_info_gtt_used"] < GIB, "GPU GTT has not drained")
    require(gpu["0000:c1:00.0"]["vendor"] == 0x1002 and gpu["0000:c1:00.0"]["device"] == 0x1586,
            "PCI selection is not Strix Halo")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            comm = (proc / "comm").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(comm not in ("qk", "llama-server", "server", "llama-bench"), "another inference process is active")
    return dict(environment={k: v for k, v in os.environ.items() if k.startswith("QK_")},
                mem_available=mem_available(), cgroup=limits, gpu=gpu)


class Engine:
    def __init__(self, library, model, reference=False):
        self.lib = C.CDLL(str(library))
        self.engine = None
        signatures = {
            "qk_open": (P, [C.c_char_p, C.POINTER(Config), C.c_char_p, C.c_size_t]),
            "qk_close": (None, [P]),
            "qk_stage_run": (C.c_int, [P, U, P, P, U, U, P, P]),
            "qk_stage_run_last": (C.c_int, [P, U, P, P, U, U, P]),
            "qk_stage_logits": (C.c_int, [P, P, U]),
            "qk_stage_topk": (C.c_int, [P, U, P, P]),
        }
        for name in ("qk_n_embd", "qk_n_vocab", "qk_layer_first", "qk_layer_end"):
            signatures[name] = (U, [P])
        if reference:
            del signatures["qk_stage_run_last"]
        for name, (result, args) in signatures.items():
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = result, args
        err = C.create_string_buffer(4096)
        self.engine = self.lib.qk_open(os.fsencode(model), C.byref(Config(1, 2048, 1)), err, len(err))
        require(self.engine, "qk_open: " + err.value.decode(errors="replace"))
        require((self.lib.qk_layer_first(self.engine), self.lib.qk_layer_end(self.engine)) == (46, 48), "wrong loaded stage")
        self.width = self.lib.qk_n_embd(self.engine)
        self.vocab = self.lib.qk_n_vocab(self.engine)
        require((self.width, self.vocab) == (10240, 248320), "wrong architecture")

    def close(self):
        if self.engine:
            self.lib.qk_close(self.engine)
            self.engine = None

    def run(self, last, values, n, base=0, offset=0):
        require(mem_available() >= 8 * GIB, "partial-run memory floor reached")
        require(len(values) >= (offset + n) * self.width, "input buffer too short")
        hidden = (F * len(values)).from_buffer(values)
        count = 1 if last else n
        ids = (U * (count + 2))(*([0xD15EA5ED] * (count + 2)))
        dest = C.byref(ids, 4)
        source = C.byref(hidden, offset * self.width * 4)
        start = time.perf_counter()
        if last:
            rc = self.lib.qk_stage_run_last(self.engine, 0, None, source, n, base, dest)
        else:
            rc = self.lib.qk_stage_run(self.engine, 0, None, source, n, base, None, dest)
        elapsed = time.perf_counter() - start
        require(rc == 0, f"stage run failed: {rc}")
        require(ids[0] == ids[count + 1] == 0xD15EA5ED, "output canary changed")
        require(all(v < self.vocab for v in ids[1:count+1]), "invalid output id")
        self.ids_sha256 = hashlib.sha256(bytes(ids)[4:-4]).hexdigest()
        return ids[count], elapsed

    def output(self, final_id):
        logits = (F * self.vocab)()
        require(self.lib.qk_stage_logits(self.engine, logits, self.vocab) == 0, "logit copy failed")
        require(all(math.isfinite(v) for v in logits), "nonfinite logits")
        top_ids, top_vals = (U * 20)(), (F * 20)()
        require(self.lib.qk_stage_topk(self.engine, 20, top_ids, top_vals) == 0, "top-k failed")
        expected = sorted(range(self.vocab), key=lambda i: (-logits[i], i))[:20]
        require(list(top_ids) == expected and top_ids[0] == final_id, "top-k/argmax mismatch")
        require(all(top_vals[i] == logits[t] for i, t in enumerate(top_ids)), "top-k values mismatch")
        return bytes(logits)

    def invalid_args(self, values):
        hidden = (F * len(values)).from_buffer(values)
        out = U(0xD15EA5ED)
        run = self.lib.qk_stage_run_last
        for slot, source, n, base, dest in ((1, hidden, 1, 0, C.byref(out)),
                (0, hidden, 0, 0, C.byref(out)), (0, hidden, 1, 2**32-1, C.byref(out)),
                (0, hidden, 2049, 0, C.byref(out)), (0, None, 1, 0, C.byref(out)),
                (0, hidden, 1, 0, None), (0, hidden, 1, 2047, C.byref(out))):
            require(run(self.engine, slot, None, source, n, base, dest) < 0, "invalid call accepted")
            require(out.value == 0xD15EA5ED, "invalid call wrote output")
        bad = (F * self.width)()
        bad[0] = float("nan")
        require(run(self.engine, 0, None, bad, 1, 0, C.byref(out)) < 0, "nonfinite input accepted")


def main():
    global JSON_OUT
    # Native C/C++ startup diagnostics use stdout too. Preserve a dedicated
    # JSON descriptor, then route their buffered output to the diagnostic log.
    sys.stdout.flush()
    JSON_OUT = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="old all-ID ABI only, for a separate preserved-library run")
    args = parser.parse_args()
    require(not (args.profile_only and args.reference_only), "choose profile or reference")
    require(os.environ.get("QK_FLASH_PROFILE", "0") == ("2" if args.profile_only else "0"), "profile mode mismatch")
    metadata = admission()
    metadata.update(library_sha256=digest(args.library), harness_sha256=digest(__file__),
                    shader_sha256={p.name: digest(p) for p in sorted(Path(os.environ["QK_SHADER_DIR"]).glob("*.spv"))},
                    model_shards=[dict(name=p.name, bytes=p.stat().st_size) for p in sorted(args.model.parent.glob("*.gguf"))],
                    layers=[46, 48], full_model_validation=False, profile=args.profile_only, reference=args.reference_only)
    emit(type="metadata", **metadata)
    engine = Engine(args.library, args.model, args.reference_only)
    try:
        drm = {}
        for path in Path("/proc/self/fdinfo").iterdir():
            try:
                fields = [line for line in path.read_text().splitlines() if line.startswith("drm-")]
            except FileNotFoundError:
                continue
            if fields:
                drm[path.name] = fields
        emit(type="loaded", mem_available=mem_available(), cgroup=cgroup(), drm=drm)
        rng = random.Random(59270)
        values = array("f", (rng.uniform(-0.5, 0.5) for _ in range(1090 * engine.width)))
        if args.profile_only:
            for last in (False, True):
                print(f"[last-head profile] last_only={last} rows=512", file=sys.stderr, flush=True)
                emit(type="profile_begin", last_only=last, rows=512)
                final, elapsed = engine.run(last, values, 512)
                emit(type="profile_end", last_only=last, seconds=elapsed, final_id=final)
            return
        for n in (1, 2, 63, 64, 65, 127, 128, 129, 309, 511, 512, 513, 1024):
            outputs, times, all_id_hashes = [], [], []
            for last in ((False,) if args.reference_only else (False, True)):
                final, elapsed = engine.run(last, values, n)
                id_hashes = [engine.ids_sha256]
                current = [engine.output(final)]
                base = n
                for step in (1, 2, 63):
                    final, _ = engine.run(last, values, step, base, base)
                    id_hashes.append(engine.ids_sha256)
                    current.append(engine.output(final))
                    base += step
                outputs.append(current)
                times.append(elapsed)
                if not last:
                    all_id_hashes = id_hashes
            if args.reference_only:
                emit(type="reference", rows=n, sha256=[hashlib.sha256(row).hexdigest() for row in outputs[0]],
                     all_ids_sha256=all_id_hashes)
                continue
            require(outputs[0] == outputs[1], f"non-exact final logits or continuation at N={n}")
            emit(type="parity", rows=n, bit_exact=True, compared_logit_rows=4,
                 continuation_sizes=[1, 2, 63], sha256=[hashlib.sha256(row).hexdigest() for row in outputs[0]],
                 all_ids_sha256=all_id_hashes, seconds_all=times[0], seconds_last=times[1])
        if args.reference_only:
            emit(type="reference_complete", rows=13)
            return
        engine.invalid_args(values)
        zeros = array("f", [0.0]) * (65 * engine.width)
        a, _ = engine.run(False, zeros, 65)
        before = engine.output(a)
        b, _ = engine.run(True, zeros, 65)
        require(before == engine.output(b), "zero-input/reset parity failed")
        emit(type="validation", bad_args_rejected=True, zero_input_exact=True)
        for n in (128, 512, 1024):
            samples = {False: [], True: []}
            # Warm each shape, then alternate ABBA to balance ordering effects.
            for last in (False, True):
                engine.run(last, values, n)
            for last in (False, True, True, False) * 3:
                final, elapsed = engine.run(last, values, n)
                samples[last].append(elapsed)
                emit(type="sample", rows=n, last_only=last, seconds=elapsed, final_id=final)
            all_median, last_median = (statistics.median(samples[k]) for k in (False, True))
            emit(type="timing", rows=n, all_seconds=samples[False], last_seconds=samples[True],
                 all_median=all_median, last_median=last_median, speedup=all_median/last_median,
                 partial_stage_only=True)
        emit(type="result", result="PASS", full_model_validation=False, cgroup=cgroup(), mem_available=mem_available())
    finally:
        engine.close()
        emit(type="closed", mem_available=mem_available(), cgroup=cgroup())


if __name__ == "__main__":
    main()
