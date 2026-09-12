#!/usr/bin/env python3
"""Serial versus split-K decode attention on a long actual-model prompt (dedicated GPU window).

The decode-attention mode (QK_ATTN_DECODE=serial|split) is fixed per engine
process, so the comparison runs as three steps on the single device:

  generate  --ids-out IDS [--tokens N | --fixture JSON --size N | --ids-file FILE]
      Writes exactly N ids. From the shared benchmark fixture (validated with
      bench/prefill_matrix.py's own load_fixture/prompt_for, so it is the
      prompt the benchmark sends for a declared size N), from a file of real
      token ids (one per line; fewer than N is an error), or as a greedy
      continuation of ids 198..213 (the only branch that loads the engine).
      With --fixture, --teacher-tail M appends M deterministic ids after
      the exact N-token benchmark prompt. Dump with --tail M to test the
      prefill-to-decode transition at that prompt length (not N-M).
  dump      --mode serial|split --ids IDS --out DUMP [--tail M] [--ctx C]
      Engine in the requested mode with a forced F32 environment (coopmat 0,
      table warming 0, profiling/timing/taps cleared; every QK_* knob in
      effect is recorded). Captures the clean first-position row from a fresh
      reset, feeds ids[:N-M] through batched prefill (identical in both
      modes), then feeds the last M ids one position at a time so the serial
      decode path runs full attention over N-M..N-1 keys, recording each
      position's logits and greedy id. Then: reset check (a fresh first
      position must reproduce the clean row bit for bit) and repeat check (a
      fresh prefill of the prefix followed by ALL M tail positions again must
      reproduce every tail row bit for bit). Writes DUMP (rows) and
      DUMP.json (manifest bound to the dump by its sha256: ids hash, every
      GGUF shard's identity, library hash, hash of every shader, context,
      tail, knobs, check results). Outputs are created exclusively.
  compare   SERIAL_DUMP SPLIT_DUMP [--rms-bound 1e-5] [--allow-argmax-flips 0]
      Strict gate: both manifests must exist with every required key, match
      their dumps' sha256, report passed reset/repeat checks, and agree on
      every identity key except the mode; every row must be finite; the
      clean rows are compared; the largest tail relative RMS must stay under
      the bound and argmax flips within the allowance. Prints one JSON line
      ending in "result": "PASS" or "FAIL" and exits non-zero on FAIL.

Both dumps must come from the same build and IDS file. This is a
kernel-change check (F32 association order differs), not an oracle test."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from array import array

VOCAB = 248320
CHUNK = 512  # batched prefill frame width
MANIFEST_VERSION = 2
FORCED_ENV = {"QK_NATIVE_FLASH": "1", "QK_FLASH_COOPMAT": "0", "QK_PLE_PREFETCH": "0"}
CLEARED_ENV = ("QK_FLASH_PROFILE", "QK_FLASH_TIMING", "QK_LAYER_DUMP", "QK_DEVICE_PCI", "QK_DEVICE", "QK_LAYERS")
# Every manifest must carry these, non-null; all but the per-run keys must agree between the two dumps.
REQUIRED_KEYS = ("manifest_version", "mode", "out", "dump_sha256", "ids", "ids_sha256", "n", "tail", "ctx", "vocab",
                 "model", "library_sha256", "shaders", "knobs", "reset_exact", "repeat_exact", "greedy_after_tail", "seconds")
PER_RUN_KEYS = {"mode", "out", "dump_sha256", "seconds", "greedy_after_tail", "knobs"}

class Config(C.Structure):
    _fields_ = [("slots", C.c_uint32), ("ctx", C.c_uint32), ("chunk", C.c_uint32)]

# ---------------------------------------------------------------- pure helpers (CPU-testable)

def tail_plan(prefix, m):
    """Positions the tail feeds, in order: prefix .. prefix+m-1 (contiguous, all m of them)."""
    return [prefix + i for i in range(m)]

def check_contiguous(plan, prefix, m):
    """A tail replay must start at the prefix and feed every position once, in order."""
    if len(plan) != m or plan != list(range(prefix, prefix + m)):
        raise ValueError(f"tail replay must feed all {m} positions {prefix}..{prefix+m-1} in order, got {plan[:4]}..")

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def shader_identity(shader_dir):
    """sha256 of every shader binary in the directory, keyed by name."""
    names = sorted(n for n in os.listdir(shader_dir) if n.endswith(".spv"))
    if not names: raise ValueError(f"no shaders in {shader_dir}")
    return {n: file_sha256(os.path.join(shader_dir, n)) for n in names}

def resolve_shards(model):
    """All GGUF shards of a split model (first shard named -00001-of-NNNNN), or the single file."""
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(model))
    if not m: return [model]
    if m.group(1) != "00001": raise ValueError("pass the first shard of a split model")
    count = int(m.group(2)); base = model[:-len(m.group(0))]
    return [f"{base}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]

def model_identity(model):
    shards = []
    for path in resolve_shards(model):
        st = os.stat(path)  # a missing shard raises
        shards.append({"path": os.path.abspath(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return {"shards": shards}

def manifest_problems(a, b):
    """Schema and identity checks for two manifests; returns a list of problems (empty = compatible)."""
    bad = []
    for tag, mf in (("serial", a), ("split", b)):
        for k in REQUIRED_KEYS:
            if k not in mf or mf[k] is None: bad.append(f"{tag} manifest missing {k}")
        if mf.get("manifest_version") != MANIFEST_VERSION: bad.append(f"{tag} manifest version is not {MANIFEST_VERSION}")
        if not isinstance(mf.get("shaders"), dict) or not mf.get("shaders"): bad.append(f"{tag} manifest has no shader hashes")
        if not isinstance(mf.get("model"), dict) or not mf["model"].get("shards"): bad.append(f"{tag} manifest has no model shards")
    if bad: return bad
    for k in REQUIRED_KEYS:
        if k not in PER_RUN_KEYS and a[k] != b[k]: bad.append(k)
    ka, kb = dict(a["knobs"]), dict(b["knobs"])
    ka.pop("QK_ATTN_DECODE", None); kb.pop("QK_ATTN_DECODE", None)
    if ka != kb: bad.append("knobs")
    if a["mode"] != "serial" or b["mode"] != "split": bad.append("modes must be serial then split")
    for tag, mf in (("serial", a), ("split", b)):
        if not (mf["reset_exact"] is True and mf["repeat_exact"] is True): bad.append(f"{tag} dump did not pass its reset/repeat checks")
    return bad

def gate(a_clean, b_clean, tail_pairs, rms_bound, allow_flips):
    """Compare logit rows; tail_pairs yields (serial_row, split_row). Returns (record, ok)."""
    if not (math.isfinite(rms_bound) and rms_bound > 0): raise ValueError("rms bound must be finite and positive")
    if allow_flips < 0: raise ValueError("allowed flips must be >= 0")
    def finite(r): return all(math.isfinite(v) for v in r)
    def rel_rms(a, b): return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)) / max(sum(y*y for y in a), 1e-20))
    def lsm(row):
        mx = max(row); z = math.log(sum(math.exp(v-mx) for v in row)); return [v-mx-z for v in row]
    def argmax(r): return max(range(len(r)), key=r.__getitem__)
    if not (finite(a_clean) and finite(b_clean)): return {"result": "FAIL", "reason": "nonfinite clean row"}, False
    if len(a_clean) != len(b_clean): return {"result": "FAIL", "reason": "clean row width mismatch"}, False
    clean_rms = rel_rms(a_clean, b_clean)
    kl = []; rms = []; flips = 0; n = 0
    for a, b in tail_pairs:
        n += 1
        if len(a) != len(b): return {"result": "FAIL", "reason": f"row width mismatch at tail position {n-1}"}, False
        if not (finite(a) and finite(b)): return {"result": "FAIL", "reason": f"nonfinite logits at tail position {n-1}"}, False
        pa, pb = lsm(a), lsm(b)
        kl.append(sum(math.exp(x)*(x-y) for x, y in zip(pa, pb)))
        rms.append(rel_rms(a, b))
        flips += int(argmax(a) != argmax(b))
    if n == 0: return {"result": "FAIL", "reason": "no tail rows"}, False
    ks = sorted(kl)
    worst = max(max(rms), clean_rms)
    ok = worst < rms_bound and flips <= allow_flips
    rec = {"tail_positions": n, "argmax_agree": n - flips, "argmax_flips": flips, "kl_mean_nats": sum(kl)/n,
           "kl_median_nats": ks[n//2], "kl_p99_nats": ks[int(0.99*(n-1))], "kl_max_nats": ks[-1],
           "relative_rms_max": max(rms), "clean_first_row_relative_rms": clean_rms, "clean_first_row_equal": a_clean == b_clean,
           "rms_bound": rms_bound, "allowed_flips": allow_flips, "result": "PASS" if ok else "FAIL"}
    return rec, ok

def run_dump_flow(runner, ids, m):
    """The dump procedure against an abstract runner (step(tokens, base) -> greedy id, row() -> array).

    Returns (clean_row, tail_rows, greedy, reset_exact, repeat_exact). The repeat check replays
    the whole tail after a fresh prefix prefill and compares every row."""
    n = len(ids); prefix = n - m
    runner.step([ids[0]], 0); clean = runner.row()
    def prefill():
        pos = 0
        while pos < prefix:
            width = min(CHUNK, prefix - pos); runner.step(ids[pos:pos+width], pos); pos += width
    prefill()
    plan = tail_plan(prefix, m); check_contiguous(plan, prefix, m)
    greedy = []; rows = []
    for pos in plan:
        greedy.append(runner.step([ids[pos]], pos)); rows.append(runner.row())
    runner.step([ids[0]], 0); reset_exact = runner.row() == clean
    prefill()
    repeat_exact = True
    for i, pos in enumerate(plan):
        runner.step([ids[pos]], pos); repeat_exact = repeat_exact and (runner.row() == rows[i])
    return clean, rows, greedy, reset_exact, repeat_exact

def read_ids(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines or not all(l.lstrip("-").isdigit() for l in lines): raise SystemExit("ids file must hold one integer per line")
    ids = [int(l) for l in lines]
    if any(not (0 <= t < VOCAB) for t in ids): raise SystemExit("ids file has ids out of vocabulary")
    return ids

def bench_module():
    """bench/prefill_matrix.py (root-owned), imported read-only for its fixture validation."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "prefill_matrix.py")
    spec = importlib.util.spec_from_file_location("prefill_matrix", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def fixture_prompt(path, size, bench=None):
    """The benchmark's own load_fixture (version, canonical sha256, id types, sizes) and prompt_for."""
    bench = bench or bench_module()
    try:
        fixture = bench.load_fixture(path)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as err:
        raise SystemExit(f"invalid fixture: {err}")
    if type(size) is not int or size not in fixture["sizes"]: raise SystemExit(f"--size must be one of the fixture sizes {fixture['sizes']}")
    prompt = bench.prompt_for(fixture, size)
    if any(type(t) is not int or not (0 <= t < VOCAB) for t in prompt): raise SystemExit("fixture ids out of this model's vocabulary")
    return prompt, fixture["sha256"]

def write_exclusive(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as f: f.write(data)

def generate_ids(args, open_serial_runner=None):
    """Returns (ids, source, fixture_sha). Loads the engine only for greedy continuation."""
    teacher_tail = getattr(args, "teacher_tail", 0)
    if type(teacher_tail) is not int or not 0 <= teacher_tail <= 512:
        raise SystemExit("--teacher-tail must be 0..512")
    if teacher_tail and not args.fixture:
        raise SystemExit("--teacher-tail requires --fixture")
    if args.fixture and args.ids_file:
        raise SystemExit("choose --fixture or --ids-file, not both")
    if args.fixture:
        if args.size is None: raise SystemExit("--fixture needs --size")
        ids, sha = fixture_prompt(args.fixture, args.size)
        source = f"fixture {args.fixture} size {args.size}"
        if teacher_tail:
            if len(ids) + teacher_tail >= 32768:
                raise SystemExit("fixture + teacher tail must leave one position in context32768")
            ids += [(1000 + 7*i) % VOCAB for i in range(teacher_tail)]
            source += f" + {teacher_tail} teacher ids (1000 + 7*i) mod {VOCAB}"
        return ids, source, sha
    if args.ids_file:
        ids = read_ids(args.ids_file)
        if len(ids) < args.tokens: raise SystemExit(f"--ids-file has {len(ids)} ids, fewer than the requested {args.tokens}")
        return ids[:args.tokens], f"file {args.ids_file}", None
    if not (2 <= args.tokens <= 32768): raise SystemExit("--tokens 2..32768")
    if open_serial_runner is None: raise SystemExit("greedy generation needs the engine")
    runner, close = open_serial_runner(max(args.tokens + 1, 128))
    try:
        ids = list(range(198, 214))[:args.tokens]; pos = 0
        while len(ids) < args.tokens or pos < len(ids):
            width = min(CHUNK, len(ids) - pos); last = runner.step(ids[pos:pos+width], pos); pos += width
            if len(ids) < args.tokens: ids.append(last)
    finally:
        close()
    return ids, "greedy continuation of ids 198..213 (serial engine)", None

# ---------------------------------------------------------------- engine

def load_lib(library):
    lib = C.CDLL(os.path.abspath(library))
    u32, f32 = C.c_uint32, C.c_float
    lib.qk_open.argtypes = [C.c_char_p, C.POINTER(Config), C.c_void_p, C.c_size_t]; lib.qk_open.restype = C.c_void_p
    lib.qk_close.argtypes = [C.c_void_p]
    lib.qk_stage_run.argtypes = [C.c_void_p,u32,C.POINTER(u32),C.POINTER(f32),u32,u32,C.POINTER(f32),C.POINTER(u32)]
    lib.qk_stage_run.restype = C.c_int
    lib.qk_stage_logits.argtypes = [C.c_void_p,C.POINTER(f32),u32]; lib.qk_stage_logits.restype = C.c_int
    return lib

def prepare_env(device, mode):
    for key in CLEARED_ENV: os.environ.pop(key, None)
    os.environ.update(FORCED_ENV)
    os.environ.update(QK_SHADER_DIR=os.path.abspath("build-halo/shaders"), QK_DEVICE_NAME=device, QK_ATTN_DECODE=mode)
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("QK_")}

class EngineRunner:
    def __init__(self, lib, engine):
        self.lib, self.e = lib, engine
        self.logits = (C.c_float*VOCAB)()
    def step(self, tok_list, base):
        u32 = C.c_uint32; width = len(tok_list); tok = (u32*width)(*tok_list); out = (u32*width)()
        rc = self.lib.qk_stage_run(self.e, 0, tok, None, width, base, None, out)
        if rc != 0: raise RuntimeError(f"stage_run rc={rc} base={base} width={width}")
        return int(out[width-1])
    def row(self):
        if self.lib.qk_stage_logits(self.e, self.logits, VOCAB) != 0: raise RuntimeError("stage_logits failed")
        return array("f", self.logits)

def open_engine(lib, model, ctx, knobs):
    err = C.create_string_buffer(512); t0 = time.monotonic()
    e = lib.qk_open(os.fsencode(model), C.byref(Config(1, ctx, 1)), err, len(err))
    if not e: raise RuntimeError(f"{knobs.get('QK_ATTN_DECODE')}: {err.value.decode()}")
    print(json.dumps({"loaded": "0:48", "ctx": ctx, "seconds": time.monotonic()-t0, "knobs": knobs}), flush=True)
    return e

def generate(args):
    def open_serial_runner(ctx):
        lib = load_lib(args.library); knobs = prepare_env(args.device, "serial")
        e = open_engine(lib, args.model, ctx, knobs)
        return EngineRunner(lib, e), (lambda: lib.qk_close(e))
    ids, source, fixture_sha = generate_ids(args, open_serial_runner)
    write_exclusive(args.ids_out, ("\n".join(map(str, ids)) + "\n").encode())
    print(json.dumps({"ids_out": args.ids_out, "count": len(ids), "source": source, "fixture_sha256": fixture_sha,
                      "fixture_prefix_tokens": args.size if args.fixture else None,
                      "teacher_tail_tokens": getattr(args, "teacher_tail", 0),
                      "ids_sha256": file_sha256(args.ids_out)}), flush=True)

def dump(args):
    ids = read_ids(args.ids)
    n, m = len(ids), args.tail
    if not (1 <= m <= min(n - 1, 512)): raise SystemExit("--tail must be 1..min(N-1, 512)")
    ctx = args.ctx or (n + 8)
    if ctx < n + 1 or ctx > 32768: raise SystemExit("--ctx must cover N+1 positions and stay <= 32768")
    if args.mode not in ("serial", "split"): raise SystemExit("--mode serial|split")
    if os.path.exists(args.out) or os.path.exists(args.out + ".json"): raise SystemExit("output exists; refusing to overwrite")
    identity = {"ids_sha256": file_sha256(args.ids), "model": model_identity(args.model), "library_sha256": file_sha256(args.library)}
    lib = load_lib(args.library); knobs = prepare_env(args.device, args.mode)
    identity["shaders"] = shader_identity(os.environ["QK_SHADER_DIR"])
    e = open_engine(lib, args.model, ctx, knobs)
    try:
        t0 = time.monotonic()
        clean, rows, greedy, reset_exact, repeat_exact = run_dump_flow(EngineRunner(lib, e), ids, m)
        seconds = time.monotonic() - t0
    finally:
        lib.qk_close(e)
    payload = bytes(clean) + b"".join(bytes(r) for r in rows)
    write_exclusive(args.out, payload)
    manifest = {"manifest_version": MANIFEST_VERSION, "mode": args.mode, "out": os.path.abspath(args.out),
                "dump_sha256": hashlib.sha256(payload).hexdigest(), "ids": os.path.abspath(args.ids), **identity,
                "n": n, "tail": m, "ctx": ctx, "vocab": VOCAB, "knobs": knobs,
                "reset_exact": reset_exact, "repeat_exact": repeat_exact, "greedy_after_tail": greedy, "seconds": seconds}
    write_exclusive(args.out + ".json", json.dumps(manifest, indent=1).encode())
    print(json.dumps({k: manifest[k] for k in ("mode", "out", "n", "tail", "ctx", "reset_exact", "repeat_exact", "greedy_after_tail", "seconds")}), flush=True)
    if not (reset_exact and repeat_exact): raise SystemExit("reset or repeat check failed (recorded in the manifest)")

def compare(args):
    if not (math.isfinite(args.rms_bound) and args.rms_bound > 0): raise SystemExit("--rms-bound must be finite and positive")
    if args.allow_argmax_flips < 0: raise SystemExit("--allow-argmax-flips must be >= 0")
    def fail(reason, **extra):
        print(json.dumps({"result": "FAIL", "reason": reason, **extra}), flush=True); raise SystemExit(1)
    try:
        with open(args.serial + ".json") as fa, open(args.split + ".json") as fb:
            ma, mb = json.load(fa), json.load(fb)
    except (OSError, json.JSONDecodeError) as err:
        fail(f"missing or unreadable manifest: {err}")
    problems = manifest_problems(ma, mb)
    if problems: fail("manifest mismatch", problems=problems)
    for tag, path, mf in (("serial", args.serial, ma), ("split", args.split, mb)):
        if file_sha256(path) != mf["dump_sha256"]: fail(f"{tag} dump does not match its manifest (dump_sha256)")
    rows = ma["tail"] + 1; vocab = ma["vocab"]
    for path in (args.serial, args.split):
        if os.path.getsize(path) != rows * vocab * 4: fail(f"{path} has the wrong size for {rows} rows of {vocab}")
    def read_row(f):
        r = array("f"); r.frombytes(f.read(vocab*4)); return r
    with open(args.serial, "rb") as fa, open(args.split, "rb") as fb:
        a_clean, b_clean = read_row(fa), read_row(fb)
        def pairs():
            for _ in range(rows - 1): yield read_row(fa), read_row(fb)
        rec, ok = gate(a_clean, b_clean, pairs(), args.rms_bound, args.allow_argmax_flips)
    rec.update({"serial": args.serial, "split": args.split, "n": ma["n"], "tail": ma["tail"], "ctx": ma["ctx"], "ids_sha256": ma["ids_sha256"],
                "greedy_serial": ma["greedy_after_tail"], "greedy_split": mb["greedy_after_tail"]})
    print(json.dumps(rec), flush=True)
    if not ok: raise SystemExit(1)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("model"); g.add_argument("--ids-out", required=True)
    g.add_argument("--tokens", type=int, default=2048); g.add_argument("--ids-file"); g.add_argument("--fixture"); g.add_argument("--size", type=int)
    g.add_argument("--teacher-tail", type=int, default=0, help="append deterministic teacher ids after the exact fixture prompt (0..512)")
    g.add_argument("--device", default="STRIX_HALO"); g.add_argument("--library", default="build-halo/libqk.so")
    d = sub.add_parser("dump"); d.add_argument("model"); d.add_argument("--mode", required=True); d.add_argument("--ids", required=True)
    d.add_argument("--out", required=True); d.add_argument("--tail", type=int, default=64); d.add_argument("--ctx", type=int, default=0)
    d.add_argument("--device", default="STRIX_HALO"); d.add_argument("--library", default="build-halo/libqk.so")
    c = sub.add_parser("compare"); c.add_argument("serial"); c.add_argument("split")
    c.add_argument("--rms-bound", type=float, default=1e-5); c.add_argument("--allow-argmax-flips", type=int, default=0)
    args = p.parse_args(argv)
    {"generate": generate, "dump": dump, "compare": compare}[args.cmd](args)

if __name__ == "__main__": main()
