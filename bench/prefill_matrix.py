#!/usr/bin/env python3
"""Same-token HTTP benchmark for native qk, llama.cpp and compatible recipes.

No servers are started or stopped. Use one exclusive, independently verified
Halo-only server at a time. See README-prefill-matrix.md for cache semantics.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import random
import re
import statistics
import time
import urllib.request
import uuid
from pathlib import Path


VERSION = 1
DEFAULT_SIZES = [128, 512, 2048, 8192, 16384]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def token_ids(value):
    if not isinstance(value, list) or not value:
        raise ValueError("expected a nonempty token-id list")
    if any(type(t) is not int or not 0 <= t < 2**31 for t in value):
        raise ValueError("invalid token id")
    return value


def post(url, path, body, timeout):
    request = urllib.request.Request(url.rstrip("/") + path,
        data=canonical(body), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def post_json(url, path, body, timeout):
    with post(url, path, body, timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError(f"backend returned an error for {path}: {result}")
    return result


def tokenize(url, text, special, timeout):
    return token_ids(post_json(url, "/tokenize",
        {"content": text, "add_special": special}, timeout)["tokens"])


def corpus(chars):
    """Deterministic, varied synthetic records; not a quality evaluation."""
    rng = random.Random(20260911)
    parts = ["<|im_start|>system\nRead the records and follow the final instruction."
             "<|im_end|>\n<|im_start|>user\nRecords for inspection:\n"]
    size = len(parts[0])
    i = 0
    while size < chars:
        line = (f"Record {i}: job {rng.getrandbits(64):016x}, "
                f"input {rng.randrange(100000, 999999)}, "
                f"checksum {rng.getrandbits(48):012x}; "
                "the worker verified the payload and stored the result.\n")
        parts.append(line)
        size += len(line)
        i += 1
    return "".join(parts)


def prepare(args):
    sizes = sorted(set(args.sizes))
    if not sizes or min(sizes) < 64 or max(sizes) > 262144:
        raise ValueError("prompt sizes must be 64..262144")
    source = Path(args.prompt_file).read_text() if args.prompt_file else corpus(max(sizes)*8)
    suffix = ("\nEnd of records. Ignore the records for your answer. "
              "Write the integers from 1 to 1000, separated by commas, without explanation."
              "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    fixture = {"version": VERSION, "model_id": args.model_id, "sizes": sizes,
               "source_text": source, "suffix_text": suffix,
               "source_ids": tokenize(args.url, source, True, args.timeout),
               "suffix_ids": tokenize(args.url, suffix, False, args.timeout),
               "workload": "fixed token prefixes plus counting suffix; synthetic, varied records"}
    if len(fixture["suffix_ids"]) >= min(sizes):
        raise ValueError("smallest prompt cannot hold the suffix")
    if len(fixture["source_ids"]) + len(fixture["suffix_ids"]) < max(sizes):
        raise ValueError("source text has too few tokens for the largest prompt")
    fixture["sha256"] = digest(fixture)
    with Path(args.output).open("x") as out:
        json.dump(fixture, out, indent=2)
        out.write("\n")
    print(json.dumps({"fixture": args.output, "sha256": fixture["sha256"], "sizes": sizes}))


def load_fixture(path):
    value = json.loads(Path(path).read_text())
    supplied = value.pop("sha256")
    if value.get("version") != VERSION or supplied != digest(value):
        raise ValueError("fixture version/hash mismatch")
    token_ids(value["source_ids"])
    token_ids(value["suffix_ids"])
    sizes = value["sizes"]
    if not sizes or any(type(n) is not int or not 64 <= n <= 262144 for n in sizes):
        raise ValueError("invalid fixture prompt sizes")
    if sizes != sorted(set(sizes)):
        raise ValueError("fixture sizes must be unique and sorted")
    if len(value["suffix_ids"]) >= min(sizes):
        raise ValueError("suffix does not fit")
    if len(value["source_ids"]) + len(value["suffix_ids"]) < max(sizes):
        raise ValueError("source token list is too short")
    value["sha256"] = supplied
    return value


def prompt_for(fixture, n):
    prompt = fixture["source_ids"][:n-len(fixture["suffix_ids"])] + fixture["suffix_ids"]
    if len(prompt) != n:
        raise ValueError("prompt size mismatch")
    return prompt


def sse(response):
    """Decode complete SSE messages, including multiline data and EOF."""
    data = []
    for raw in response:
        if len(raw) > 2**20:
            raise ValueError("oversized SSE line")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                joined = "\n".join(data)
                if joined == "[DONE]":
                    return
                yield json.loads(joined)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        joined = "\n".join(data)
        if joined != "[DONE]":
            yield json.loads(joined)


def positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def reported(result):
    timings = result.get("timings") or {}
    ms = timings.get("prompt_ms")
    return {"server_timings": timings,
            "server_prefill_seconds": ms/1000 if positive_number(ms) else None,
            "server_prompt_tokens": timings.get("prompt_n", result.get("tokens_evaluated")),
            "server_cached_tokens": result.get("tokens_cached")}


def request_body(prompt, n_predict, stream):
    return {"prompt": prompt, "n_predict": n_predict, "stream": stream,
            "temperature": 0, "seed": 0, "cache_prompt": False, "return_tokens": True}


def measure_prefill(url, prompt, timeout):
    start = time.monotonic()
    out = post_json(url, "/completion", request_body(prompt, 1, False), timeout)
    elapsed = time.monotonic() - start
    return {"prefill_plus_one_token_seconds": elapsed,
            "effective_prompt_tokens_per_second": len(prompt)/elapsed,
            "output_tokens": out.get("tokens"), **reported(out)}


def measure_stream(url, prompt, count, timeout):
    start = time.monotonic()
    first = last = None
    first_count = 0
    total = 0
    complete_counts = True
    events = 0
    final = None
    text = []
    ids = []
    with post(url, "/completion", request_body(prompt, count, True), timeout) as response:
        for item in sse(response):
            if not isinstance(item, dict) or item.get("error"):
                raise ValueError(f"stream error: {item}")
            if item.get("stop"):
                final = item
                continue
            tokens = item.get("tokens")
            content = item.get("content", "")
            if not content and not tokens:
                continue
            now = time.monotonic()
            valid_count = isinstance(tokens, list) and bool(tokens)
            if valid_count:
                token_ids(tokens)
                ids.extend(tokens)
                total += len(tokens)
            else:
                complete_counts = False
            if first is None:
                first = now
                first_count = len(tokens) if valid_count else 0
            last = now
            text.append(content)
            events += 1
    if first is None or final is None:
        raise ValueError("stream ended without tokens or a final stop record")
    duration = last-first
    speed = ((total-first_count)/duration
             if complete_counts and total > first_count and duration > 0 else None)
    output = "".join(text)
    counting = re.sub(r"\s+", "", output)
    coherent = bool(counting) and ",".join(str(i) for i in range(1, 1001)).startswith(counting)
    return {"ttft_seconds": first-start, "stream_total_seconds": time.monotonic()-start,
            "decode_tokens_per_second": speed, "streamed_tokens": total if complete_counts else None,
            "requested_output_tokens": count, "exact_output_length": complete_counts and total == count,
            "events": events, "complete_chunk_token_counts": complete_counts,
            "coherent_counting_prefix": coherent, "output_preview": output[:256],
            "output_text_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "output_token_sha256": digest(ids) if complete_counts else None, **reported(final)}


def run(args):
    fixture = load_fixture(args.fixture)
    if not args.confirm_exclusive:
        raise ValueError("verify one Halo-only server and pass --confirm-exclusive")
    if fixture["model_id"] != args.model_id:
        raise ValueError("model id differs from the shared fixture")
    if not 1 <= args.repetitions <= 20 or not 2 <= args.decode_tokens <= 4096:
        raise ValueError("repetitions 1..20; decode tokens 2..4096")
    sizes = sorted(set(args.sizes)) if args.sizes else fixture["sizes"]
    if not sizes or any(n not in fixture["sizes"] for n in sizes):
        raise ValueError("selected sizes must exist in the shared fixture")
    if max(sizes) + args.decode_tokens > args.context:
        raise ValueError("context must hold largest prompt plus requested output")
    # Refuse to compare a different tokenizer. No sampling or approximate count.
    if tokenize(args.url, fixture["source_text"], True, args.timeout) != fixture["source_ids"]:
        raise ValueError("source tokenizer mismatch")
    if tokenize(args.url, fixture["suffix_text"], False, args.timeout) != fixture["suffix_ids"]:
        raise ValueError("suffix tokenizer mismatch")
    metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else {}
    run_id = str(uuid.uuid4())
    base = {"run_id": run_id, "backend": args.backend, "model_id": args.model_id,
            "fixture_sha256": fixture["sha256"], "context": args.context,
            "config_sha256": digest(metadata), "requested_output_tokens": args.decode_tokens}
    with Path(args.output).open("x", buffering=1) as out:
        def emit(kind, **fields):
            row = {"type": kind, **base, **fields}
            out.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
        emit("run_start", utc=dt.datetime.now(dt.timezone.utc).isoformat(),
             url=args.url, host=platform.node(), metadata=metadata,
             sizes=sizes, repetitions=args.repetitions,
             hardware_scope="operator-confirmed Halo-only, exclusive server",
             cache_procedure="cache_prompt=false; no OS cache flushing; decode follows same-prompt prefill probe")
        try:
            for n in sizes:
                prompt = prompt_for(fixture, n)
                for repetition in range(1, args.repetitions+1):
                    common = {"prompt_tokens": n, "prompt_sha256": digest(prompt),
                              "repetition": repetition,
                              "cache_condition": "first at size" if repetition == 1 else "repeated working set"}
                    emit("prefill", **common, **measure_prefill(args.url, prompt, args.timeout))
                    emit("decode", **common, **measure_stream(args.url, prompt, args.decode_tokens, args.timeout))
            emit("run_complete")
        except Exception as error:
            emit("run_error", error=f"{type(error).__name__}: {error}")
            raise


def stats(values):
    values = [v for v in values if positive_number(v)]
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def summarize(args):
    rows = []
    for path in args.results:
        rows += [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    completed = {r["run_id"] for r in rows if r["type"] == "run_complete"}
    if not completed:
        raise ValueError("no completed runs")
    # A green terminator alone is not proof that the complete matrix ran.
    for run_id in completed:
        run_rows = [r for r in rows if r["run_id"] == run_id]
        starts = [r for r in run_rows if r["type"] == "run_start"]
        if len(starts) != 1 or sum(r["type"] == "run_complete" for r in run_rows) != 1:
            raise ValueError("duplicate or missing run header/terminator")
        if any(r["type"] == "run_error" for r in run_rows):
            raise ValueError("completed run also contains an error")
        header = starts[0]
        wanted = {(kind, n, rep) for kind in ("prefill", "decode")
                  for n in header["sizes"] for rep in range(1, header["repetitions"]+1)}
        measured = [r for r in run_rows if r["type"] in ("prefill", "decode")]
        actual = [(r["type"], r["prompt_tokens"], r["repetition"]) for r in measured]
        if len(actual) != len(wanted) or set(actual) != wanted:
            raise ValueError("completed run does not cover its declared matrix exactly once")
        for row in measured:
            for key in ("model_id", "fixture_sha256", "config_sha256", "context", "backend", "requested_output_tokens"):
                if row[key] != header[key]:
                    raise ValueError("inconsistent run identity/configuration")
    groups = {}
    for row in rows:
        if row["type"] not in ("prefill", "decode") or row["run_id"] not in completed:
            continue
        key = (row["fixture_sha256"], row["model_id"], row["context"],
               row["requested_output_tokens"], row["backend"], row["config_sha256"], row["prompt_tokens"])
        groups.setdefault(key, []).append(row)
    for key, group in sorted(groups.items()):
        prefill = [r for r in group if r["type"] == "prefill"]
        decode = [r for r in group if r["type"] == "decode"]
        print(json.dumps({"fixture_sha256": key[0], "model_id": key[1], "context": key[2],
            "requested_output_tokens": key[3], "backend": key[4], "config_sha256": key[5], "prompt_tokens": key[6],
            "prefill_plus_one_token_seconds": stats([r["prefill_plus_one_token_seconds"] for r in prefill]),
            "repeated_prefill_seconds": stats([r["prefill_plus_one_token_seconds"] for r in prefill if r["repetition"] > 1]),
            "ttft_seconds": stats([r["ttft_seconds"] for r in decode]),
            "decode_tokens_per_second": stats([r["decode_tokens_per_second"] for r in decode
                if r["exact_output_length"] and r.get("coherent_counting_prefix") is True]),
            "short_or_uncounted_completions": sum(not r["exact_output_length"] for r in decode),
            "counting_failures_or_unchecked": sum(r.get("coherent_counting_prefix") is not True for r in decode),
            "runs": sorted({r["run_id"] for r in group})}, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="create a shared fixture using one server's tokenizer")
    prep.add_argument("--url", required=True)
    prep.add_argument("--model-id", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    prep.add_argument("--prompt-file")
    prep.add_argument("--timeout", type=float, default=600)
    execute = sub.add_parser("run", help="benchmark one already-running, exclusive server")
    for flag in ("url", "backend", "model-id", "fixture", "output"):
        execute.add_argument("--"+flag, required=True)
    execute.add_argument("--context", type=int, default=32768)
    execute.add_argument("--sizes", type=int, nargs="+", help="optional subset of the shared fixture sizes")
    execute.add_argument("--decode-tokens", type=int, default=128)
    execute.add_argument("--repetitions", type=int, default=3)
    execute.add_argument("--timeout", type=float, default=600)
    execute.add_argument("--metadata", help="JSON of build, flags, MTP, KV types, placement; never put secrets here")
    execute.add_argument("--confirm-exclusive", action="store_true")
    summary = sub.add_parser("summarize", help="JSON summaries, grouped by exact workload and backend")
    summary.add_argument("results", nargs="+")
    args = parser.parse_args()
    try:
        if hasattr(args, "timeout") and not positive_number(args.timeout):
            raise ValueError("timeout must be finite and positive")
        {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
