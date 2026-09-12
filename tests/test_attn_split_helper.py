#!/usr/bin/env python3
"""CPU-only regression tests for tests/gpu_qwen4_attn_split.py (no GPU, no model, no libqk).

Covers the tail replay sequence (including a deliberately skipped tail), the
dump flow against a stub engine that enforces the stage-ABI base contract,
the comparison gates (finite, RMS bound, argmax flips), manifest schema and
identity checks (missing keys, differing runtimes accepted, dump binding,
same-size corrupt dumps), the compare command's PASS/FAIL exits, model shard
resolution, and the generate branches that must not load the engine."""
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from array import array
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("attn_split", os.path.join(HERE, "gpu_qwen4_attn_split.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class StubRunner:
    """Deterministic stand-in for the engine: rows depend on the exact fed history,
    and a step whose base is not the current position (or 0 for a reset) fails,
    like qk_stage_run's -4 for a base that does not continue the sequence."""
    def __init__(self, width=8):
        self.width = width; self.history = []; self.fed = []
    def step(self, tokens, base):
        if base == 0: self.history = []
        elif base != len(self.history): raise RuntimeError(f"base {base} does not continue position {len(self.history)}")
        self.history.extend(tokens); self.fed.append((base, list(tokens)))
        return int(self.row()[0] * 1000)  # a deterministic "greedy id"
    def row(self):
        h = hashlib.sha256(",".join(map(str, self.history)).encode()).digest()
        return array("f", [b / 255.0 for b in h[:self.width]])


class TailPlanTests(unittest.TestCase):
    def test_plan_is_contiguous(self):
        plan = mod.tail_plan(100, 5)
        self.assertEqual(plan, [100, 101, 102, 103, 104])
        mod.check_contiguous(plan, 100, 5)

    def test_skipped_tail_is_rejected(self):
        with self.assertRaises(ValueError): mod.check_contiguous([103, 104], 100, 5)  # the old bug
        with self.assertRaises(ValueError): mod.check_contiguous([100, 102, 103, 104, 105], 100, 5)


class DumpFlowTests(unittest.TestCase):
    def test_flow_feeds_every_tail_position_twice_and_checks_pass(self):
        ids = list(range(1000, 1000 + 1100)); m = 6; prefix = len(ids) - m
        r = StubRunner()
        clean, rows, greedy, reset_exact, repeat_exact = mod.run_dump_flow(r, ids, m)
        self.assertTrue(reset_exact and repeat_exact)
        self.assertEqual(len(rows), m); self.assertEqual(len(greedy), m)
        tail_bases = [b for b, toks in r.fed if len(toks) == 1 and b >= prefix]
        self.assertEqual(tail_bases, list(range(prefix, prefix + m)) * 2)  # main pass and full repeat
        chunk_bases = [b for b, toks in r.fed if len(toks) > 1]
        self.assertEqual(chunk_bases, [0, 512, 1024] * 2)  # 1094-token prefix in 512-wide frames, twice

    def test_skipped_repeat_violates_engine_contract(self):
        ids = list(range(1000, 1100)); m = 4; prefix = len(ids) - m
        r = StubRunner(); r.step(ids[:prefix], 0)
        with self.assertRaises(RuntimeError): r.step([ids[prefix + m - 2]], prefix + m - 2)

    def test_state_corruption_is_detected(self):
        class Drifting(StubRunner):
            def row(self):
                r = super().row(); r[0] += 1e-3 * (len(self.fed) % 3 == 0); return r
        _, _, _, reset_exact, repeat_exact = mod.run_dump_flow(Drifting(), list(range(50)), 3)
        self.assertFalse(reset_exact and repeat_exact)


class GateTests(unittest.TestCase):
    def rows(self, n, seed=1, delta=0.0, flip=False):
        out = []
        for i in range(n):
            r = array("f", [math.sin(seed + i * 0.37 + j * 0.11) for j in range(64)])
            if delta: r[3] += delta
            if flip: r[7] = max(r) + 1.0
            out.append(r)
        return out

    def test_identical_rows_pass(self):
        a = self.rows(4); b = self.rows(4)
        rec, ok = mod.gate(a[0], b[0], zip(a[1:], b[1:]), 1e-5, 0)
        self.assertTrue(ok); self.assertEqual(rec["result"], "PASS"); self.assertTrue(rec["clean_first_row_equal"])

    def test_large_error_fails(self):
        a = self.rows(4); b = self.rows(4, delta=0.5)
        rec, ok = mod.gate(a[0], b[0], zip(a[1:], b[1:]), 1e-5, 0)
        self.assertFalse(ok); self.assertEqual(rec["result"], "FAIL"); self.assertGreater(rec["relative_rms_max"], 1e-5)

    def test_argmax_flip_gate(self):
        a = self.rows(3); b = self.rows(3, flip=True)
        _, ok0 = mod.gate(a[0], b[0], zip(a[1:], b[1:]), 10.0, 0)
        _, ok2 = mod.gate(a[0], b[0], zip(a[1:], b[1:]), 10.0, 2)
        self.assertFalse(ok0); self.assertTrue(ok2)

    def test_nonfinite_empty_and_bad_bound(self):
        a = self.rows(2); b = self.rows(2); b[1][0] = float("nan")
        rec, ok = mod.gate(a[0], b[0], zip(a[1:], b[1:]), 1e-5, 0)
        self.assertFalse(ok); self.assertIn("nonfinite", rec["reason"])
        _, ok = mod.gate(a[0], b[0], iter([]), 1e-5, 0); self.assertFalse(ok)
        with self.assertRaises(ValueError): mod.gate(a[0], b[0], zip(a[1:], b[1:]), 0.0, 0)
        with self.assertRaises(ValueError): mod.gate(a[0], b[0], zip(a[1:], b[1:]), 1e-5, -1)


def manifest(mode, **over):
    m = {"manifest_version": mod.MANIFEST_VERSION, "mode": mode, "out": f"/tmp/{mode}.f32", "dump_sha256": "d" + mode,
         "ids": "/tmp/ids", "ids_sha256": "abc", "n": 8, "tail": 1, "ctx": 16, "vocab": 64,
         "model": {"shards": [{"path": "/m-00001-of-00002.gguf", "size": 1, "mtime_ns": 2}, {"path": "/m-00002-of-00002.gguf", "size": 3, "mtime_ns": 4}]},
         "library_sha256": "lib", "shaders": {"fa_attn_srv.spv": "s1", "gemv_q5_k.spv": "s2"},
         "knobs": {"QK_ATTN_DECODE": mode, "QK_FLASH_COOPMAT": "0", "QK_PLE_PREFETCH": "0"}, "reset_exact": True, "repeat_exact": True,
         "greedy_after_tail": [5], "seconds": 1.0 if mode == "serial" else 2.5}
    m.update(over); return m


class ManifestTests(unittest.TestCase):
    def test_mode_runtime_and_greedy_differences_are_accepted(self):
        self.assertEqual(mod.manifest_problems(manifest("serial"), manifest("split", seconds=99.0, greedy_after_tail=[6])), [])

    def test_identity_differences_are_reported(self):
        self.assertIn("ids_sha256", mod.manifest_problems(manifest("serial"), manifest("split", ids_sha256="zzz")))
        self.assertIn("library_sha256", mod.manifest_problems(manifest("serial"), manifest("split", library_sha256="other")))
        self.assertIn("shaders", mod.manifest_problems(manifest("serial"), manifest("split", shaders={"fa_attn_srv.spv": "s1", "gemv_q5_k.spv": "changed"})))
        other_model = {"shards": [{"path": "/m-00001-of-00002.gguf", "size": 1, "mtime_ns": 2}, {"path": "/m-00002-of-00002.gguf", "size": 9, "mtime_ns": 4}]}
        self.assertIn("model", mod.manifest_problems(manifest("serial"), manifest("split", model=other_model)))
        self.assertIn("ctx", mod.manifest_problems(manifest("serial"), manifest("split", ctx=32)))
        knobs = {"QK_ATTN_DECODE": "split", "QK_FLASH_COOPMAT": "1", "QK_PLE_PREFETCH": "0"}
        self.assertIn("knobs", mod.manifest_problems(manifest("serial"), manifest("split", knobs=knobs)))

    def test_missing_required_keys_fail_even_when_missing_in_both(self):
        a, b = manifest("serial"), manifest("split")
        del a["library_sha256"]; del b["library_sha256"]
        problems = mod.manifest_problems(a, b)
        self.assertTrue(any("missing library_sha256" in p for p in problems))
        a, b = manifest("serial", shaders={}), manifest("split", shaders={})
        self.assertTrue(any("no shader hashes" in p for p in mod.manifest_problems(a, b)))
        a, b = manifest("serial", manifest_version=1), manifest("split", manifest_version=1)
        self.assertTrue(any("version" in p for p in mod.manifest_problems(a, b)))

    def test_mode_order_and_checks(self):
        self.assertTrue(any("modes" in p for p in mod.manifest_problems(manifest("split"), manifest("serial"))))
        self.assertTrue(any("reset/repeat" in p for p in mod.manifest_problems(manifest("serial"), manifest("split", repeat_exact=False))))


class CompareCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, delta=0.0):
        base = [array("f", [math.cos(i * 0.3 + j * 0.05) for j in range(64)]) for i in range(2)]
        if delta: base[1][2] += delta
        return base

    def write_dump(self, name, mode, rows, mf, bind=True):
        path = os.path.join(self.dir, name)
        payload = b"".join(bytes(r) for r in rows)
        with open(path, "wb") as f: f.write(payload)
        if mf is not None:
            if bind: mf = dict(mf, dump_sha256=hashlib.sha256(payload).hexdigest())
            with open(path + ".json", "w") as f: json.dump(mf, f)
        return path

    def run_compare(self, a, b, extra=()):
        out = io.StringIO()
        with redirect_stdout(out):
            try:
                mod.main(["compare", a, b, *extra]); code = 0
            except SystemExit as e:
                code = e.code
        return code, json.loads(out.getvalue().strip().splitlines()[-1])

    def test_pass_with_different_runtimes(self):
        a = self.write_dump("s.f32", "serial", self.rows(), manifest("serial", seconds=1.0))
        b = self.write_dump("p.f32", "split", self.rows(), manifest("split", seconds=7.5))
        code, rec = self.run_compare(a, b)
        self.assertEqual(code, 0); self.assertEqual(rec["result"], "PASS")

    def test_large_error_fails_with_nonzero_exit(self):
        a = self.write_dump("s.f32", "serial", self.rows(), manifest("serial"))
        b = self.write_dump("p.f32", "split", self.rows(delta=0.3), manifest("split"))
        code, rec = self.run_compare(a, b)
        self.assertEqual(code, 1); self.assertEqual(rec["result"], "FAIL")

    def test_missing_manifest_fails(self):
        a = self.write_dump("s.f32", "serial", self.rows(), manifest("serial"))
        b = self.write_dump("p.f32", "split", self.rows(), None)
        code, rec = self.run_compare(a, b)
        self.assertEqual(code, 1); self.assertIn("manifest", rec["reason"])

    def test_missing_required_key_fails(self):
        mf = manifest("split"); del mf["ids_sha256"]
        a = self.write_dump("s.f32", "serial", self.rows(), manifest("serial"))
        b = self.write_dump("p.f32", "split", self.rows(), mf)
        code, rec = self.run_compare(a, b)
        self.assertEqual(code, 1); self.assertTrue(any("missing ids_sha256" in p for p in rec["problems"]))

    def test_same_size_corrupt_dump_fails_binding(self):
        a = self.write_dump("s.f32", "serial", self.rows(), manifest("serial"))
        mf = dict(manifest("split"), dump_sha256=hashlib.sha256(b"".join(bytes(r) for r in self.rows())).hexdigest())
        b = self.write_dump("p.f32", "split", self.rows(delta=0.3), mf, bind=False)  # manifest says clean, file is corrupt
        code, rec = self.run_compare(a, b)
        self.assertEqual(code, 1); self.assertIn("dump_sha256", rec["reason"])

    def test_unrelated_dumps_fail(self):
        a = self.write_dump("s.f32", "serial", self.rows(), manifest("serial"))
        b = self.write_dump("p.f32", "split", self.rows(), manifest("split", ids_sha256="different"))
        code, rec = self.run_compare(a, b)
        self.assertEqual(code, 1); self.assertIn("ids_sha256", rec["problems"])

    def test_failed_repeat_check_blocks_compare(self):
        a = self.write_dump("s.f32", "serial", self.rows(), manifest("serial"))
        b = self.write_dump("p.f32", "split", self.rows(), manifest("split", repeat_exact=False))
        code, rec = self.run_compare(a, b)
        self.assertEqual(code, 1); self.assertTrue(any("reset/repeat" in p for p in rec["problems"]))

    def test_bad_bound_rejected(self):
        a = self.write_dump("s.f32", "serial", self.rows(), manifest("serial"))
        b = self.write_dump("p.f32", "split", self.rows(), manifest("split"))
        with self.assertRaises(SystemExit): mod.main(["compare", a, b, "--rms-bound", "0"])


class ShardAndGenerateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve_shards(self):
        first = os.path.join(self.dir, "m-00001-of-00003.gguf")
        self.assertEqual([os.path.basename(p) for p in mod.resolve_shards(first)], ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf"])
        self.assertEqual(mod.resolve_shards("/x/single.gguf"), ["/x/single.gguf"])
        with self.assertRaises(ValueError): mod.resolve_shards(os.path.join(self.dir, "m-00002-of-00003.gguf"))

    def test_model_identity_lists_every_shard_and_requires_them(self):
        for i in (1, 2):
            with open(os.path.join(self.dir, f"m-{i:05d}-of-00002.gguf"), "wb") as f: f.write(b"x" * i)
        ident = mod.model_identity(os.path.join(self.dir, "m-00001-of-00002.gguf"))
        self.assertEqual([s["size"] for s in ident["shards"]], [1, 2])
        os.remove(os.path.join(self.dir, "m-00002-of-00002.gguf"))
        with self.assertRaises(FileNotFoundError): mod.model_identity(os.path.join(self.dir, "m-00001-of-00002.gguf"))

    def args(self, **kw):
        base = dict(fixture=None, size=None, ids_file=None, tokens=8)
        base.update(kw); return types.SimpleNamespace(**base)

    def test_generate_from_ids_file_does_not_load_engine(self):
        path = os.path.join(self.dir, "ids.txt")
        with open(path, "w") as f: f.write("\n".join(str(t) for t in range(10, 30)) + "\n")
        def no_engine(ctx): raise AssertionError("engine must not be opened for file input")
        ids, source, sha = mod.generate_ids(self.args(ids_file=path, tokens=8), no_engine)
        self.assertEqual(ids, list(range(10, 18))); self.assertIsNone(sha)
        with self.assertRaises(SystemExit): mod.generate_ids(self.args(ids_file=path, tokens=64), no_engine)  # too short, no truncation
        with open(path, "w") as f: f.write("5\nabc\n")
        with self.assertRaises(SystemExit): mod.read_ids(path)
        with open(path, "w") as f: f.write(f"5\n{mod.VOCAB}\n")
        with self.assertRaises(SystemExit): mod.read_ids(path)

    def test_generate_from_fixture_uses_bench_validation(self):
        bench = types.SimpleNamespace(
            load_fixture=lambda p: {"sizes": [8, 16], "source_ids": list(range(100, 140)), "suffix_ids": [7, 8, 9], "sha256": "fx"},
            prompt_for=lambda fx, n: fx["source_ids"][:n-len(fx["suffix_ids"])] + fx["suffix_ids"])
        ids, sha = mod.fixture_prompt("unused", 8, bench)
        self.assertEqual(ids, [100, 101, 102, 103, 104, 7, 8, 9]); self.assertEqual(sha, "fx")
        with self.assertRaises(SystemExit): mod.fixture_prompt("unused", 12, bench)   # not a declared size
        with self.assertRaises(SystemExit): mod.fixture_prompt("unused", True, bench)  # bool is not an int size
        bad = types.SimpleNamespace(load_fixture=lambda p: (_ for _ in ()).throw(ValueError("fixture version/hash mismatch")), prompt_for=None)
        with self.assertRaises(SystemExit): mod.fixture_prompt("unused", 8, bad)

    def test_real_bench_fixture_roundtrip(self):
        """Uses the actual bench module: a fixture with the benchmark's canonical hash validates, a tampered one fails."""
        bench = mod.bench_module()
        fixture = {"version": bench.VERSION, "model_id": "m", "sizes": [64, 128], "source_text": "s", "suffix_text": "x",
                   "source_ids": list(range(1000, 1200)), "suffix_ids": [1, 2, 3], "workload": "test"}
        fixture["sha256"] = bench.digest(fixture)
        path = os.path.join(self.dir, "fx.json")
        with open(path, "w") as f: json.dump(fixture, f)
        ids, sha = mod.fixture_prompt(path, 64)
        self.assertEqual(len(ids), 64); self.assertEqual(ids[-3:], [1, 2, 3]); self.assertEqual(sha, fixture["sha256"])
        fixture["source_ids"][0] = 1001
        with open(path, "w") as f: json.dump(fixture, f)  # hash no longer matches
        with self.assertRaises(SystemExit): mod.fixture_prompt(path, 64)

    def test_generate_greedy_needs_engine(self):
        with self.assertRaises(SystemExit): mod.generate_ids(self.args(tokens=8), None)

    def test_fixture_teacher_tail_preserves_exact_prompt_without_engine(self):
        bench = mod.bench_module()
        fixture = {"version": bench.VERSION, "model_id": "m", "sizes": [128], "source_text": "s", "suffix_text": "x",
                   "source_ids": list(range(1000, 1200)), "suffix_ids": [1, 2, 3], "workload": "test"}
        fixture["sha256"] = bench.digest(fixture)
        path = os.path.join(self.dir, "teacher-fx.json")
        with open(path, "w") as f: json.dump(fixture, f)
        def no_engine(ctx): raise AssertionError("fixture + teacher tail must not load the engine")
        ids, source, sha = mod.generate_ids(self.args(fixture=path, size=128, teacher_tail=3), no_engine)
        self.assertEqual(ids[:128], bench.prompt_for(fixture, 128))
        self.assertEqual(ids[128:], [1000, 1007, 1014])
        self.assertEqual(sha, fixture["sha256"])
        self.assertIn("3 teacher ids", source)
        runner = StubRunner()
        _, _, _, reset, repeat = mod.run_dump_flow(runner, ids, 3)
        self.assertTrue(reset and repeat)
        self.assertEqual([b for b, t in runner.fed if len(t) == 3], [])
        self.assertEqual([b for b, t in runner.fed if len(t) == 1 and b >= 128], [128,129,130]*2)

    def test_teacher_tail_bad_options_fail_without_engine(self):
        for count in (-1, 513, True, 1.5):
            with self.subTest(count=count), self.assertRaises(SystemExit):
                mod.generate_ids(self.args(teacher_tail=count))
        with self.assertRaises(SystemExit): mod.generate_ids(self.args(teacher_tail=1))
        with self.assertRaises(SystemExit):
            mod.generate_ids(self.args(fixture="x", ids_file="y", size=128))

    def test_teacher_tail_must_fit_context(self):
        from unittest.mock import patch
        with patch.object(mod, "fixture_prompt", return_value=([1]*32767, "fx")), self.assertRaises(SystemExit):
            mod.generate_ids(self.args(fixture="x", size=32767, teacher_tail=1))


if __name__ == "__main__":
    unittest.main()
