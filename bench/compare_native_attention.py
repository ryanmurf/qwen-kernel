#!/usr/bin/env python3
"""Read-only, same-build serial/ordered API comparison; never starts a model.

Requires complete prefill_matrix.py runs with run-api-ab.py metadata. Refuses
different precision, model, shaders, environment, workload, or output hashes.
This audits recorded evidence, not the live machine. One sample is exploratory,
and even repeated counting outputs are not a general model-quality evaluation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    wire = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(wire.encode()).hexdigest()


def positive(value):
    import math
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def normalized_metadata(metadata, mode):
    value = copy.deepcopy(metadata)
    require(value["actual_qk_environment"]["QK_ATTN_DECODE"] == mode,
            "recorded environment does not select requested attention mode")
    require(value["actual_qk_environment"]["QK_DEVICE_NAME"] == "STRIX_HALO"
            and value["device_pci"] == "0000:c1:00.0", "not the Halo-only configuration")
    require(value["actual_decode_attention"] == ("serial" if mode == "serial" else "ordered-F32"),
            "recorded backend announcement disagrees with mode")
    require(value["kv_precision"] == "F32" and value["mtp"] is False,
            "not the gated F32/no-MTP configuration")
    require(value["shaders"] and value["model"]["shards"]
            and len(value["library_sha256"]) == 64 and len(value["long_gate_sha256"]) == 64,
            "missing build/model/gate identity")
    command = value["server_command"]
    require(command.count(f"QK_ATTN_DECODE={mode}") == 1, "launch mode mismatch")
    value["server_command"] = ["--unit=RUN" if a.startswith("--unit=") else
                               "QK_ATTN_DECODE=MODE" if a.startswith("QK_ATTN_DECODE=") else a
                               for a in command]
    value.pop("actual_pid")
    value["actual_decode_attention"] = "MODE"
    value["actual_qk_environment"]["QK_ATTN_DECODE"] = "MODE"
    return value


def audit(rows, mode, *, normalize=normalized_metadata):
    require(len(rows) >= 4 and rows[0]["type"] == "run_start"
            and rows[-1]["type"] == "run_complete", "incomplete run")
    header = rows[0]
    require(type(header["repetitions"]) is int and header["repetitions"] > 0,
            "invalid repetition count")
    require(all(type(n) is int and n > 0 for n in header["sizes"]), "invalid prompt sizes")
    require(header["config_sha256"] == digest(header["metadata"]), "metadata hash mismatch")
    normalized = normalize(header["metadata"], mode)
    wanted = [(kind, size, rep) for size in header["sizes"]
              for rep in range(1, header["repetitions"] + 1) for kind in ("prefill", "decode")]
    require(wanted and len(wanted) == len(set(wanted)), "empty or duplicate declared cells")
    observed = [(r["type"], r.get("prompt_tokens"), r.get("repetition")) for r in rows[1:-1]]
    require(observed == wanted, "missing, reordered, duplicate, or unexpected cells")
    for row in rows[1:]:
        for key in ("run_id", "backend", "model_id", "fixture_sha256", "context",
                    "config_sha256", "requested_output_tokens"):
            require(row[key] == header[key], f"inconsistent {key}")
    cells = {}
    for row in rows[1:-1]:
        if row["type"] == "prefill":
            require(positive(row["prefill_plus_one_token_seconds"]), "invalid prefill timing")
            require(len(row["output_tokens"]) == 1 and type(row["output_tokens"][0]) is int,
                    "uncounted prefill output")
        else:
            require(all(row[k] is True for k in ("exact_output_length", "complete_chunk_token_counts",
                                                 "coherent_counting_prefix")), "failed decode quality/count check")
            require(row["streamed_tokens"] == header["requested_output_tokens"], "short decode")
            require(positive(row["ttft_seconds"]) and positive(row["decode_tokens_per_second"]),
                    "invalid decode timing")
            require(all(isinstance(row[k], str) and len(row[k]) == 64
                        for k in ("output_token_sha256", "output_text_sha256")), "missing output hashes")
        require(row["server_prompt_tokens"] == row["prompt_tokens"], "server prompt count mismatch")
        require(row.get("server_cached_tokens") in (None, 0), "cached-prefix comparison")
        cells[(row["type"], row["prompt_tokens"], row["repetition"])] = row
    for size in header["sizes"]:
        hashes = {r["prompt_sha256"] for r in cells.values() if r["prompt_tokens"] == size}
        require(len(hashes) == 1, "probe/stream or repetition prompt mismatch")
    return header, normalized, cells


def compare_modes(first, second, *, modes, normalize):
    """Shared workload/output audit; caller explicitly supplies mode validation."""
    a_mode, b_mode = modes
    ah, am, ac = audit(first, a_mode, normalize=normalize)
    bh, bm, bc = audit(second, b_mode, normalize=normalize)
    require(am == bm, "server configurations differ beyond selected mode/run identity")
    for key in ("model_id", "fixture_sha256", "context", "requested_output_tokens",
                "sizes", "repetitions", "cache_procedure", "host", "url"):
        require(ah[key] == bh[key], f"paired workload differs: {key}")
    require(ac.keys() == bc.keys(), "paired cells differ")
    for key, a in ac.items():
        b = bc[key]
        fields = ("prompt_sha256", "cache_condition") + (("output_tokens",) if key[0] == "prefill"
                   else ("output_token_sha256", "output_text_sha256"))
        for field in fields:
            require(a[field] == b[field], f"paired {field} differs at {key}")
    result = []
    for size in ah["sizes"]:
        metrics = {}
        for kind, field in (("prefill", "prefill_plus_one_token_seconds"),
                            ("decode", "ttft_seconds"), ("decode", "decode_tokens_per_second")):
            av = [r[field] for key, r in ac.items() if key[:2] == (kind, size)]
            bv = [r[field] for key, r in bc.items() if key[:2] == (kind, size)]
            sa, sb = statistics.median(av), statistics.median(bv)
            metrics[field] = {a_mode: {"median": sa, "min": min(av), "max": max(av)},
                              b_mode: {"median": sb, "min": min(bv), "max": max(bv)},
                              f"{b_mode}_over_{a_mode}": sb / sa}
        result.append({"prompt_tokens": size, "samples_per_mode": ah["repetitions"], **metrics})
    return {"comparison_valid": True, "outputs_match": True,
            "scope": "matched recorded API runs; not broad quality or statistical significance",
            "exploratory": ah["repetitions"] == 1, "rows": result}


def compare(serial, ordered):
    return compare_modes(serial, ordered, modes=("serial", "ordered"),
                         normalize=normalized_metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("ordered", type=Path)
    args = parser.parse_args()
    try:
        load = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        print(json.dumps(compare(load(args.serial), load(args.ordered)), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
