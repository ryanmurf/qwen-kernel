#!/usr/bin/env python3
"""Freeze reproducible numeric-token Gemma 4 parity fixtures from llama-server."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "fixtures"
LLAMA_CPP_DIR = Path("/mnt/data/llama.cpp-master")
LLAMA_SERVER = LLAMA_CPP_DIR / "build/bin/llama-server"
MODEL = Path("/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf")
LLAMA_COMMIT = "571d0d540df04f25298d0e159e520d9fc62ed121"
MODEL_SHA256 = "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d"

SERVER_COMMAND = (
    "GGML_VK_VISIBLE_DEVICES=2 rtk proxy "
    "/mnt/data/llama.cpp-master/build/bin/llama-server "
    "-m /mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf "
    "-dev Vulkan0 -sm none -mg 0 -ngl 99 -fa on -ctk f16 -ctv f16 "
    "-fit off -c 16384 -b 8192 -ub 512 --parallel 1 "
    "--host 127.0.0.1 --port 18271"
)
GENERATOR_COMMAND = (
    "rtk proxy tests/gemma4/generate_fixtures.py "
    "--server http://127.0.0.1:18271"
)

SAMPLER_BASE: dict[str, Any] = {
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "xtc_probability": 0.0,
    "samplers": ["temperature"],
    "seed": 1,
    "cache_prompt": False,
    "ignore_eos": True,
    "return_tokens": True,
    "stream": False,
}

CASES = (
    {
        "name": "ordinary_chat",
        "purpose": "ordinary chat-template decode",
        "messages": [
            {
                "role": "user",
                "content": "In one sentence, explain why leaves change color in autumn.",
            }
        ],
        "n_predict": 32,
    },
    {
        "name": "coding_prompt",
        "purpose": "coding chat-template decode",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python function named stable_unique that returns the unique "
                    "items from a list in first-seen order. Include a short example."
                ),
            }
        ],
        "n_predict": 32,
    },
)

POSITION_CASES = (
    ("swa_position_1023", 1023, "first generated token occupies SWA boundary position 1023"),
    ("swa_position_1024", 1024, "first generated token occupies first wrapped SWA position 1024"),
    ("swa_position_1025", 1025, "first generated token occupies post-wrap SWA position 1025"),
    ("global_context_8192", 8192, "8K context exercises all five global-attention layers"),
)

LONG_BODY = (
    "The observatory log records a clear northern sky, a steady west wind, "
    "and three calibration pulses before sunrise. "
)
BODY_MARKER = "QK_NUMERIC_BODY_MARKER"


def post(base_url: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc


def tokenize(base_url: str, content: str, *, add_special: bool) -> list[int]:
    response = post(
        base_url,
        "/tokenize",
        {"content": content, "add_special": add_special, "parse_special": True},
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError(f"unexpected /tokenize response: {response!r}")
    return tokens


def apply_template(base_url: str, messages: list[dict[str, str]]) -> str:
    response = post(base_url, "/apply-template", {"messages": messages})
    prompt = response.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError(f"unexpected /apply-template response: {response!r}")
    return prompt


def exact_position_ids(base_url: str, count: int) -> tuple[list[int], dict[str, Any]]:
    templated = apply_template(
        base_url,
        [{"role": "user", "content": BODY_MARKER}],
    )
    if templated.count(BODY_MARKER) != 1:
        raise RuntimeError("chat template did not preserve the numeric-body marker exactly once")
    prefix, suffix = templated.split(BODY_MARKER)
    prefix_ids = tokenize(base_url, prefix, add_special=True)
    suffix_ids = tokenize(base_url, suffix, add_special=False)
    body_ids = tokenize(base_url, LONG_BODY, add_special=False)
    body_count = count - len(prefix_ids) - len(suffix_ids)
    if body_count <= 0 or not body_ids:
        raise RuntimeError(f"target token count {count} is too small for the chat envelope")
    repeated = (body_ids * ((body_count + len(body_ids) - 1) // len(body_ids)))[:body_count]
    result = prefix_ids + repeated + suffix_ids
    if len(result) != count:
        raise AssertionError(f"constructed {len(result)} tokens, expected {count}")
    return result, {
        "method": "numeric chat envelope plus repeated tokenized body, truncated exactly",
        "first_continuation_position": count,
        "prefix_token_count": len(prefix_ids),
        "body_token_count": body_count,
        "suffix_token_count": len(suffix_ids),
        "body_text": LONG_BODY,
    }


def completion(base_url: str, input_ids: list[int], n_predict: int) -> list[int]:
    request = dict(SAMPLER_BASE)
    request.update({"prompt": input_ids, "n_predict": n_predict})
    response = post(base_url, "/completion", request)
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError(f"completion did not return numeric tokens: {response!r}")
    if len(tokens) != n_predict:
        raise RuntimeError(f"completion returned {len(tokens)} tokens, expected {n_predict}")
    return tokens


def sha256_ids(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def verify_reference() -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(LLAMA_CPP_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != LLAMA_COMMIT:
        raise RuntimeError(f"llama.cpp commit mismatch: actual={actual} expected={LLAMA_COMMIT}")
    if not LLAMA_SERVER.is_file() or not MODEL.is_file():
        raise RuntimeError("llama-server binary or Gemma 4 model is missing")
    manifest = json.loads((ROOT / "tests/gemma4/manifest.json").read_text())
    recorded = manifest["artifact"]["sha256"]
    if recorded != MODEL_SHA256:
        raise RuntimeError(f"manifest model SHA mismatch: actual={recorded} expected={MODEL_SHA256}")


def fixture_record(
    *,
    name: str,
    purpose: str,
    input_ids: list[int],
    continuation_ids: list[int],
    n_predict: int,
    construction: dict[str, Any],
) -> dict[str, Any]:
    digest = sha256_ids(continuation_ids)
    settings = dict(SAMPLER_BASE)
    settings["n_predict"] = n_predict
    return {
        "schema_version": 1,
        "name": name,
        "purpose": purpose,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "reference": {
            "implementation": "llama.cpp llama-server /completion numeric-token API",
            "llama_cpp_commit": LLAMA_COMMIT,
            "llama_cpp_worktree_note": (
                "local common/debug.cpp patch adds QK_DUMP_DIR/QK_DUMP_FILTER; "
                "it does not alter model arithmetic unless those variables are set"
            ),
            "binary": str(LLAMA_SERVER),
            "model": str(MODEL),
            "model_sha256": MODEL_SHA256,
            "server_command": SERVER_COMMAND,
            "fixture_command": GENERATOR_COMMAND,
            "request": "POST /completion twice with prompt=input_ids and sampler_settings below",
        },
        "sampler_settings": settings,
        "input_construction": construction,
        "input_ids": input_ids,
        "greedy_continuation_ids": continuation_ids,
        "reproducibility": {
            "independent_runs": 2,
            "cache_prompt": False,
            "exact_match": True,
            "run_continuation_sha256": [digest, digest],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:18271")
    args = parser.parse_args()
    verify_reference()

    prepared: list[tuple[str, str, list[int], int, dict[str, Any]]] = []
    for case in CASES:
        prompt = apply_template(args.server, case["messages"])
        ids = tokenize(args.server, prompt, add_special=True)
        prepared.append(
            (
                case["name"],
                case["purpose"],
                ids,
                case["n_predict"],
                {
                    "method": "llama-server /apply-template followed by /tokenize once",
                    "messages": case["messages"],
                    "first_continuation_position": len(ids),
                },
            )
        )
    for name, count, purpose in POSITION_CASES:
        ids, construction = exact_position_ids(args.server, count)
        prepared.append((name, purpose, ids, 16, construction))

    records: list[dict[str, Any]] = []
    for name, purpose, ids, n_predict, construction in prepared:
        print(f"{name}: input={len(ids)} continuation={n_predict} run 1", flush=True)
        first = completion(args.server, ids, n_predict)
        print(f"{name}: input={len(ids)} continuation={n_predict} run 2", flush=True)
        second = completion(args.server, ids, n_predict)
        if first != second:
            raise RuntimeError(
                f"{name}: reference runs disagree: {sha256_ids(first)} != {sha256_ids(second)}"
            )
        records.append(
            fixture_record(
                name=name,
                purpose=purpose,
                input_ids=ids,
                continuation_ids=first,
                n_predict=n_predict,
                construction=construction,
            )
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = OUT_DIR / f"{record['name']}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"wrote {path.relative_to(ROOT)} "
            f"continuation_sha256={record['reproducibility']['run_continuation_sha256'][0]}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
