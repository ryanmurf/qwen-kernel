#!/usr/bin/env python3
"""Generate and validate the frozen Gemma 4 GGUF artifact ledger.

This is deliberately stdlib-only.  It reads the GGUF v3 header and hashes the
file without importing llama.cpp or loading tensor payloads into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path


DEFAULT_MODEL = Path(
    "/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf"
)
EXPECTED_SHA256 = "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d"
EXPECTED_SIZE = 14_439_363_584
EXPECTED_ACTIVE_BYTES = 2_380_055_608
EXPECTED_GLOBAL_LAYERS = [5, 11, 17, 23, 29]
EXPECTED_TYPE_HISTOGRAM = {"F32": 392, "Q4_0": 265, "Q6_K": 1}

# ggml type id -> (name, elements per block, encoded bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}

KV_TYPE_NAMES = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32",
    5: "INT32", 6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY",
    10: "UINT64", 11: "INT64", 12: "FLOAT64",
}
SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}


def fail(label: str, actual, expected) -> None:
    raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def expect(label: str, actual, expected) -> None:
    if actual != expected:
        fail(label, actual, expected)


def read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"short GGUF read: got {len(data)} bytes, wanted {size}")
    return data


def unpack(handle, fmt: str):
    return struct.unpack(fmt, read_exact(handle, struct.calcsize(fmt)))


def read_string(handle) -> str:
    (length,) = unpack(handle, "<Q")
    return read_exact(handle, length).decode("utf-8")


# The tokenizer arrays (tokens/merges/scores/token_type) are ~18 MB of the header
# and are already stored verbatim in the GGUF. Copying them into the manifest
# would bloat the repo without adding verification power, so long arrays are
# reduced to a length plus a SHA-256 over their canonical JSON encoding. That
# still detects any change to the array while staying diffable.
INLINE_ARRAY_LIMIT = 64


def summarize_kv(entry: dict) -> dict:
    value = entry["value"]
    # GGUF arrays deserialize as {"element_type": ..., "values": [...]}.
    if isinstance(value, dict) and isinstance(value.get("values"), list):
        values = value["values"]
        if len(values) <= INLINE_ARRAY_LIMIT:
            return entry
        encoded = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return {
            "type": entry["type"],
            "value_summary": {
                "element_type": value.get("element_type"),
                "count": len(values),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "first": values[:4],
                "last": values[-4:],
            },
        }
    return entry


def read_value(handle, type_id: int):
    if type_id in SCALAR_FORMATS:
        (value,) = unpack(handle, SCALAR_FORMATS[type_id])
        return bool(value) if type_id == 7 else value
    if type_id == 8:
        return read_string(handle)
    if type_id == 9:
        (element_type,) = unpack(handle, "<I")
        (count,) = unpack(handle, "<Q")
        return {
            "element_type": KV_TYPE_NAMES.get(element_type, f"UNKNOWN_{element_type}"),
            "values": [read_value(handle, element_type) for _ in range(count)],
        }
    raise ValueError(f"unsupported GGUF metadata type id {type_id}")


def tensor_length(shape: list[int], type_id: int) -> int:
    if type_id not in GGML_TYPES:
        raise ValueError(f"unsupported ggml tensor type id {type_id}")
    _, block_elements, block_bytes = GGML_TYPES[type_id]
    elements = 1
    for dimension in shape:
        elements *= dimension
    if elements % block_elements:
        raise ValueError(
            f"tensor shape {shape} has {elements} elements, not divisible by "
            f"the type-{type_id} block size {block_elements}"
        )
    return elements // block_elements * block_bytes


def parse_header(path: Path):
    metadata = {}
    tensors = []
    with path.open("rb") as handle:  # noqa: PLR1702 - single linear header pass
        expect("GGUF magic", read_exact(handle, 4), b"GGUF")
        (version,) = unpack(handle, "<I")
        expect("GGUF version", version, 3)
        (tensor_count,) = unpack(handle, "<Q")
        (metadata_count,) = unpack(handle, "<Q")

        for _ in range(metadata_count):
            key = read_string(handle)
            (type_id,) = unpack(handle, "<I")
            metadata[key] = {
                "type": KV_TYPE_NAMES.get(type_id, f"UNKNOWN_{type_id}"),
                "value": read_value(handle, type_id),
            }

        raw_tensors = []
        for _ in range(tensor_count):
            name = read_string(handle)
            (dimension_count,) = unpack(handle, "<I")
            shape = list(unpack(handle, f"<{dimension_count}Q"))
            (type_id,) = unpack(handle, "<I")
            (relative_offset,) = unpack(handle, "<Q")
            raw_tensors.append((name, shape, type_id, relative_offset))

        alignment_entry = metadata.get("general.alignment")
        alignment = alignment_entry["value"] if alignment_entry else 32
        header_end = handle.tell()
        data_offset = (header_end + alignment - 1) // alignment * alignment

    for name, shape, type_id, relative_offset in raw_tensors:
        byte_length = tensor_length(shape, type_id)
        absolute_offset = data_offset + relative_offset
        tensors.append({
            "name": name,
            "type": GGML_TYPES[type_id][0],
            "shape": shape,
            "offset": absolute_offset,
            "byte_length": byte_length,
        })
    return version, metadata_count, tensor_count, alignment, data_offset, metadata, tensors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_value(metadata, key: str):
    return metadata[key]["value"]


def validate(path: Path, metadata_count: int, tensor_count: int, metadata, tensors):
    size = path.stat().st_size
    digest = sha256_file(path)
    expect("file size", size, EXPECTED_SIZE)
    expect("SHA-256", digest, EXPECTED_SHA256)
    expect("metadata KV count", metadata_count, 52)
    expect("serialized metadata KV count", len(metadata), 52)
    expect("tensor count", tensor_count, 658)
    expect("serialized tensor count", len(tensors), 658)

    pattern_entry = metadata.get("gemma4.attention.sliding_window_pattern")
    if not pattern_entry:
        fail("sliding_window_pattern presence", None, "present")
    pattern = pattern_entry["value"]["values"]
    globals_actual = [index for index, sliding in enumerate(pattern) if not sliding]
    expect("sliding_window_pattern length", len(pattern), 30)
    expect("sliding layer count", sum(bool(value) for value in pattern), 25)
    expect("global layer count", len(globals_actual), 5)
    expect("global layer indices", globals_actual, EXPECTED_GLOBAL_LAYERS)

    histogram = dict(sorted(Counter(tensor["type"] for tensor in tensors).items()))
    expect("tensor type histogram", histogram, EXPECTED_TYPE_HISTOGRAM)
    q6_tensors = [tensor for tensor in tensors if tensor["type"] == "Q6_K"]
    expect("Q6_K tensor count", len(q6_tensors), 1)
    expect("Q6_K tensor name", q6_tensors[0]["name"], "token_embd.weight")
    expect("Q6_K tensor shape", q6_tensors[0]["shape"], [2816, 262144])

    names = {tensor["name"] for tensor in tensors}
    for layer in range(30):
        name = f"blk.{layer}.attn_v.weight"
        if layer in EXPECTED_GLOBAL_LAYERS:
            expect(f"global layer {layer} attn_v presence", name in names, False)
        else:
            expect(f"sliding layer {layer} attn_v presence", name in names, True)

    ranges = sorted(
        (tensor["offset"], tensor["offset"] + tensor["byte_length"], tensor["name"])
        for tensor in tensors
    )
    previous_end = 0
    previous_name = None
    for start, end, name in ranges:
        if start < 0 or end > size or end < start:
            fail(f"tensor range for {name}", [start, end], f"inside [0, {size})")
        if start < previous_end:
            fail(
                f"tensor overlap for {name}",
                [start, end],
                f"start >= previous end {previous_end} ({previous_name})",
            )
        previous_end, previous_name = end, name

    expert_count = int(metadata_value(metadata, "gemma4.expert_count"))
    expert_used_count = int(metadata_value(metadata, "gemma4.expert_used_count"))
    has_output = "output.weight" in names
    active_bytes = 0
    for tensor in tensors:
        length = tensor["byte_length"]
        name = tensor["name"]
        if "_exps." in name:
            numerator = length * expert_used_count
            if numerator % expert_count:
                fail(f"integral active expert bytes for {name}", numerator % expert_count, 0)
            active_bytes += numerator // expert_count
        elif name == "token_embd.weight" and has_output:
            # Untied-head models need only one encoded embedding row per token.
            hidden = tensor["shape"][0]
            type_id = next(key for key, value in GGML_TYPES.items() if value[0] == tensor["type"])
            active_bytes += tensor_length([hidden], type_id)
        else:
            active_bytes += length
    expect("summed active bytes/token", active_bytes, EXPECTED_ACTIVE_BYTES)

    vision_tensors = [
        tensor["name"] for tensor in tensors
        if any(marker in tensor["name"].lower() for marker in ("vision", "mmproj", "image"))
    ]
    expect("vision tensor count in text GGUF", len(vision_tensors), 0)
    return digest, size, histogram, active_bytes, globals_actual, vision_tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().with_name("manifest.json"),
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    parsed = parse_header(args.model)
    version, metadata_count, tensor_count, alignment, data_offset, metadata, tensors = parsed
    digest, size, histogram, active_bytes, globals_actual, vision_tensors = validate(
        args.model, metadata_count, tensor_count, metadata, tensors
    )

    manifest = {
        "schema_version": 1,
        "artifact": {
            "local_path": str(args.model),
            "filename": args.model.name,
            "byte_size": size,
            "size_gib": size / 2**30,
            "sha256": digest,
            "official_huggingface": {
                "repository": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
                "revision": "8afd43710afbb87c711f33f7e7c11b1434a9fa1a",
                "displayed_size": "14.4 GB",
                "reconciliation": (
                    "Hugging Face's 14.4 GB is a rounded decimal display of the exact "
                    "14,439,363,584-byte object; the remote and local SHA-256 values match."
                ),
            },
            "base_model_repository": metadata_value(
                metadata, "general.base_model.0.repo_url"
            ),
        },
        "gguf": {
            "version": version,
            "alignment": alignment,
            "data_offset": data_offset,
            "metadata_kv_count": metadata_count,
            "tensor_count": tensor_count,
            "text_only": True,
            "vision_tensors_skipped": vision_tensors,
        },
        "assertions": {
            "sliding_layer_count": 25,
            "global_layer_count": 5,
            "global_layer_indices": globals_actual,
            "tensor_type_histogram": histogram,
            "active_bytes_per_token": active_bytes,
            "tensor_ranges_inside_file": True,
            "tensor_ranges_non_overlapping": True,
        },
        "metadata": {key: summarize_kv(entry) for key, entry in metadata.items()},
        "tensors": tensors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temporary, args.output)
    print(
        f"PASS: wrote {args.output} ({metadata_count} metadata KVs, "
        f"{tensor_count} tensors, {active_bytes:,} active B/token)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, EOFError, KeyError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
