"""CPU-only tests for the benchmark; never load a model or touch a GPU."""
import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("matrix", Path(__file__).resolve().parents[1]/"bench/prefill_matrix.py")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body))
        if self.path == "/tokenize":
            result = {"tokens": list(range(200)) if body["add_special"] else [2]}
        elif not body["stream"]:
            result = {"tokens": [7], "timings": {"prompt_n": len(body["prompt"]), "prompt_ms": 0}}
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(body["n_predict"]):
                self.wfile.write(("data: " + json.dumps({"content": str(i), "tokens": [i+10]})+"\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b'data: {"stop": true, "timings": {"prompt_ms": 0}}\n\ndata: [DONE]\n\n')
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())


class MatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.requests = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_token_ids(self):
        for value in ([], [True], [-1], [1.5], [2**31], "12"):
            with self.assertRaises(ValueError):
                matrix.token_ids(value)
        self.assertEqual(matrix.token_ids([0, 248319]), [0, 248319])

    def test_fixture_integrity_and_exact_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.json"
            value = {"version": 1, "model_id": "test", "sizes": [64, 128],
                     "source_ids": list(range(200)), "suffix_ids": [3, 4]}
            value["sha256"] = matrix.digest(value)
            path.write_text(json.dumps(value))
            fixture = matrix.load_fixture(path)
            self.assertEqual(len(matrix.prompt_for(fixture, 128)), 128)
            self.assertEqual(matrix.prompt_for(fixture, 64)[-2:], [3, 4])
            value["source_ids"][0] = 99
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                matrix.load_fixture(path)

    def test_sse_multiline_done_and_eof(self):
        wire = [b': comment\n', b'event: message\n', b'data: {"x":\n',
                b'data: 1}\n', b'\n', b'data: [DONE]\n', b'\n', b'data: invalid\n']
        self.assertEqual(list(matrix.sse(wire)), [{"x": 1}])
        self.assertEqual(list(matrix.sse([b'data: {"x": 2}\n'])), [{"x": 2}])

    def test_real_http_probe_and_stream(self):
        prompt = list(range(128))
        prefill = matrix.measure_prefill(self.url, prompt, 5)
        stream = matrix.measure_stream(self.url, prompt, 4, 5)
        self.assertIsNone(prefill["server_prefill_seconds"])
        self.assertGreater(prefill["prefill_plus_one_token_seconds"], 0)
        self.assertEqual(prefill["server_prompt_tokens"], 128)
        self.assertTrue(stream["exact_output_length"])
        self.assertEqual(stream["streamed_tokens"], 4)
        self.assertGreater(stream["decode_tokens_per_second"], 0)
        self.assertGreater(stream["ttft_seconds"], 0)
        for path, body in self.server.requests[-2:]:
            self.assertEqual(path, "/completion")
            self.assertEqual(body["prompt"], prompt)
            self.assertFalse(body["cache_prompt"])
            self.assertEqual(body["temperature"], 0)

    def summary_rows(self):
        base = {"run_id": "run", "model_id": "test", "fixture_sha256": "fixture", "config_sha256": "cfg",
                "context": 1024, "backend": "native", "requested_output_tokens": 4}
        return [{**base, "type": "run_start", "sizes": [128], "repetitions": 1},
                {**base, "type": "prefill", "prompt_tokens": 128, "repetition": 1,
                 "prefill_plus_one_token_seconds": 2},
                {**base, "type": "decode", "prompt_tokens": 128, "repetition": 1,
                 "ttft_seconds": 2, "decode_tokens_per_second": 3, "exact_output_length": True,
                 "coherent_counting_prefix": True},
                {**base, "type": "run_complete"}]

    def summarize(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"results.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                matrix.summarize(argparse.Namespace(results=[str(path)]))
            return json.loads(stdout.getvalue())

    def test_summary_and_short_completion(self):
        rows = self.summary_rows()
        summary = self.summarize(rows)
        self.assertEqual(summary["decode_tokens_per_second"]["median"], 3)
        rows[2]["exact_output_length"] = False
        summary = self.summarize(rows)
        self.assertEqual(summary["short_or_uncounted_completions"], 1)
        self.assertIsNone(summary["decode_tokens_per_second"]["median"])
        rows[2]["exact_output_length"] = True
        rows[2]["coherent_counting_prefix"] = False
        summary = self.summarize(rows)
        self.assertEqual(summary["counting_failures_or_unchecked"], 1)
        self.assertIsNone(summary["decode_tokens_per_second"]["median"])

    def test_summary_rejects_partial_duplicate_and_mixed_config(self):
        rows = self.summary_rows()
        for bad in (rows[:2]+rows[3:], rows+rows, rows[:3], rows+[rows[2]]):
            with self.assertRaises(ValueError):
                self.summarize(bad)
        rows[2]["config_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.summarize(rows)

    def test_deterministic_corpus(self):
        self.assertEqual(matrix.corpus(1000), matrix.corpus(1000))
        self.assertGreaterEqual(len(matrix.corpus(1000)), 1000)

    def run_args(self, directory):
        directory = Path(directory)
        fixture = {"version": 1, "model_id": "test", "sizes": [64, 128],
                   "source_text": "records", "suffix_text": "count",
                   "source_ids": list(range(200)), "suffix_ids": [2]}
        fixture["sha256"] = matrix.digest(fixture)
        path = directory/"fixture.json"
        path.write_text(json.dumps(fixture))
        return argparse.Namespace(fixture=str(path), url=self.url, backend="fake",
            model_id="test", output=str(directory/"results.jsonl"), metadata=None,
            confirm_exclusive=True, context=1024, decode_tokens=4, repetitions=2,
            sizes=None, timeout=5)

    def test_complete_http_matrix_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.run_args(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                matrix.run(args)
            rows = [json.loads(line) for line in Path(args.output).read_text().splitlines()]
            self.assertEqual(rows[0]["type"], "run_start")
            self.assertEqual(rows[-1]["type"], "run_complete")
            self.assertEqual(len(rows), 10)
            measured = rows[1:-1]
            self.assertEqual({(r["type"], r["prompt_tokens"], r["repetition"]) for r in measured},
                {(kind, n, rep) for kind in ("prefill", "decode")
                 for n in (64, 128) for rep in (1, 2)})
            before = Path(args.output).read_bytes()
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                matrix.run(args)
            self.assertEqual(Path(args.output).read_bytes(), before)

    def test_failed_request_has_error_but_no_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.run_args(directory)
            with patch.object(matrix, "measure_stream", side_effect=ValueError("broken stream")):
                with self.assertRaisesRegex(ValueError, "broken stream"), contextlib.redirect_stdout(io.StringIO()):
                    matrix.run(args)
            rows = [json.loads(line) for line in Path(args.output).read_text().splitlines()]
            self.assertEqual([r["type"] for r in rows], ["run_start", "prefill", "run_error"])
            with self.assertRaisesRegex(ValueError, "no completed runs"):
                self.summarize(rows)

    def test_tokenizer_mismatch_refuses_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.run_args(directory)
            with patch.object(matrix, "tokenize", return_value=[99]):
                with self.assertRaisesRegex(ValueError, "tokenizer mismatch"):
                    matrix.run(args)
            self.assertFalse(Path(args.output).exists())

    def test_stream_requires_final_and_counts_tokens_not_events(self):
        def check(items):
            with patch.object(matrix, "post", return_value=contextlib.nullcontext(None)):
                with patch.object(matrix, "sse", return_value=iter(items)):
                    return matrix.measure_stream(self.url, [1], 3, 5)
        with self.assertRaisesRegex(ValueError, "final stop record"):
            check([{"tokens": [1], "content": "1"}])
        with self.assertRaisesRegex(ValueError, "stream error"):
            check([{"error": "failure"}])
        uncounted = check([{"content": "1, "}, {"content": "2"}, {"stop": True}])
        self.assertIsNone(uncounted["decode_tokens_per_second"])
        self.assertFalse(uncounted["exact_output_length"])
        grouped = check([{"tokens": [1, 2], "content": "1, "},
                         {"tokens": [3], "content": "2"}, {"stop": True}])
        self.assertEqual(grouped["streamed_tokens"], 3)
        self.assertEqual(grouped["events"], 2)
        self.assertTrue(grouped["exact_output_length"])
        self.assertTrue(grouped["coherent_counting_prefix"])


if __name__ == "__main__":
    unittest.main()
