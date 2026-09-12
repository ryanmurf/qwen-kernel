"""CPU-only evidence checks. No model, network, or GPU execution."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("attention_ab", Path(__file__).resolve().parents[1]
                                            / "bench/compare_native_attention.py")
ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ab)


def fixture(mode):
    metadata = {"actual_qk_environment": {"QK_ATTN_DECODE": mode, "QK_DEVICE_NAME": "STRIX_HALO"},
                "actual_decode_attention": "serial" if mode == "serial" else "ordered-F32",
                "device_pci": "0000:c1:00.0", "kv_precision": "F32", "mtp": False,
                "server_command": ["launch", f"--unit={mode}", f"QK_ATTN_DECODE={mode}"],
                "actual_pid": 1 if mode == "serial" else 2, "shaders": {"a": "a" * 64},
                "model": {"shards": [{"path": "m", "size": 100, "mtime_ns": 42}]},
                "library_sha256": "b" * 64, "long_gate_sha256": "c" * 64}
    base = {"run_id": mode, "backend": mode, "model_id": "model", "fixture_sha256": "fixture",
            "context": 32768, "config_sha256": ab.digest(metadata), "requested_output_tokens": 128}
    sample = {**base, "prompt_tokens": 8192, "repetition": 1, "prompt_sha256": "d" * 64,
              "cache_condition": "first at size", "server_prompt_tokens": 8192, "server_cached_tokens": None}
    return [{**base, "type": "run_start", "metadata": metadata, "sizes": [8192], "repetitions": 1,
             "cache_procedure": "probe then stream", "host": "max", "url": "http://127.0.0.1:8194"},
            {**sample, "type": "prefill", "prefill_plus_one_token_seconds": 10, "output_tokens": [16]},
            {**sample, "type": "decode", "ttft_seconds": 10,
             "decode_tokens_per_second": 10 if mode == "serial" else 11,
             "exact_output_length": True, "complete_chunk_token_counts": True,
             "coherent_counting_prefix": True, "streamed_tokens": 128,
             "output_token_sha256": "e" * 64, "output_text_sha256": "f" * 64},
            {**base, "type": "run_complete"}]


def rehash(rows):
    sha = ab.digest(rows[0]["metadata"])
    for row in rows:
        row["config_sha256"] = sha


class ComparisonTests(unittest.TestCase):
    def test_matching_recorded_runs(self):
        out = ab.compare(fixture("serial"), fixture("ordered"))
        self.assertTrue(out["comparison_valid"])
        self.assertTrue(out["exploratory"])
        self.assertEqual(out["rows"][0]["decode_tokens_per_second"]["ordered_over_serial"], 1.1)

    def test_incomplete_duplicate_reordered_error(self):
        source = fixture("ordered")
        for rows in (source[:-1], source[:2] + source[3:], source + source,
                     source[:2] + source[1:2] + source[2:], [source[0], source[2], source[1], source[3]],
                     source[:-1] + [{"type": "run_error"}] + source[-1:]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                ab.compare(fixture("serial"), rows)

    def test_metadata_integrity_and_full_config_match(self):
        for key, value in (("kv_precision", "F16"), ("mtp", True),
                           ("library_sha256", "x" * 64), ("shaders", {"a": "x" * 64}),
                           ("device_pci", "0000:68:00.0"), ("actual_decode_attention", "serial")):
            rows = fixture("ordered")
            rows[0]["metadata"][key] = value
            with self.assertRaisesRegex(ValueError, "metadata hash"):
                ab.compare(fixture("serial"), rows)
            rehash(rows)
            with self.subTest(key=key), self.assertRaises(ValueError):
                ab.compare(fixture("serial"), rows)

    def test_environment_and_launch_binding(self):
        for field in ("QK_ATTN_DECODE", "QK_DEVICE_NAME", "QK_FLASH_COOPMAT"):
            rows = fixture("ordered")
            rows[0]["metadata"]["actual_qk_environment"][field] = "changed"
            rehash(rows)
            with self.assertRaises(ValueError):
                ab.compare(fixture("serial"), rows)
        rows = fixture("ordered")
        rows[0]["metadata"]["server_command"].append("QK_ATTN_DECODE=serial")
        rehash(rows)
        with self.assertRaises(ValueError):
            ab.compare(fixture("serial"), rows)

    def test_wrong_workload_output_and_counts(self):
        changes = [(0, "fixture_sha256", "wrong"), (1, "prompt_sha256", "wrong"),
                   (1, "output_tokens", [17]), (2, "output_token_sha256", "a" * 64),
                   (2, "output_text_sha256", "a" * 64), (2, "streamed_tokens", 127),
                   (2, "server_prompt_tokens", 8191), (2, "server_cached_tokens", 8192),
                   (2, "coherent_counting_prefix", False), (2, "exact_output_length", False)]
        for index, key, value in changes:
            rows = fixture("ordered")
            rows[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ab.compare(fixture("serial"), rows)

    def test_invalid_timings(self):
        for value in (0, -1, True, float("nan"), float("inf")):
            rows = fixture("ordered")
            rows[2]["decode_tokens_per_second"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                ab.compare(fixture("serial"), rows)

    def test_repeated_samples_and_no_input_mutation(self):
        pair = [fixture("serial"), fixture("ordered")]
        for rows in pair:
            rows[0]["repetitions"] = 2
            repeated = copy.deepcopy(rows[1:3])
            for row in repeated:
                row["repetition"] = 2
                row["cache_condition"] = "repeated working set"
            rows[3:3] = repeated
        saved = copy.deepcopy(pair)
        out = ab.compare(*pair)
        self.assertFalse(out["exploratory"])
        self.assertEqual(out["rows"][0]["samples_per_mode"], 2)
        self.assertEqual(pair, saved)


if __name__ == "__main__":
    unittest.main()
