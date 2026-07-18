#!/usr/bin/env python3
"""Compute per-token active-set bytes for a MoE GGUF (decode bandwidth ceiling).

Pure stdlib. Parses GGUF v3 header only (no tensor data read).

Model math (per decoded token, weights only):
  - expert tensors (*_exps.weight): read top_k/n_expert of the bytes
  - shared-expert tensors (*_shexp.weight): read fully
  - token_embd.weight: one row (negligible) unless it doubles as output head
  - output.weight: read fully (logits GEMV)
  - everything else (attention/deltanet/norms/router): read fully
"""
import struct, sys, json

# ggml type id -> (block_elems, block_bytes)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}

def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")

def read_val(f, t):
    S = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
         7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
    if t in S:
        (v,) = struct.unpack(S[t], f.read(struct.calcsize(S[t])))
        return bool(v) if t == 7 else v
    if t == 8:
        return read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (cnt,) = struct.unpack("<Q", f.read(8))
        vals = [read_val(f, et) for _ in range(cnt)]
        return vals
    raise ValueError(f"unknown kv type {t}")

def tensor_bytes(dims, tid):
    name, be, bb = GGML_TYPES[tid]
    n = 1
    for d in dims:
        n *= d
    assert n % be == 0, f"{dims} not divisible by block {be}"
    return n // be * bb

def main(path):
    kv = {}
    tensors = []
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", magic
        (ver,) = struct.unpack("<I", f.read(4))
        assert ver == 3, ver
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            k = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            v = read_val(f, t)
            kv[k] = v
        for _ in range(n_tensors):
            nm = read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            (tid,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors.append((nm, list(dims), tid))

    arch = kv.get("general.architecture")
    def akv(suffix, default=None):
        return kv.get(f"{arch}.{suffix}", default)

    print(f"arch={arch}  name={kv.get('general.name')!r}")
    interesting = [k for k in kv if not k.startswith("tokenizer.")]
    for k in sorted(interesting):
        v = kv[k]
        if isinstance(v, list) and len(v) > 16:
            v = f"[{len(v)} items: {v[:4]}...]"
        print(f"  {k} = {v}")

    n_expert = akv("expert_count", 0)
    n_top = akv("expert_used_count", 0)
    print(f"\nn_expert={n_expert} top_k={n_top}")

    has_output = any(nm == "output.weight" for nm, _, _ in tensors)
    total = 0
    active = 0.0
    by_class = {}
    by_type = {}
    for nm, dims, tid in tensors:
        b = tensor_bytes(dims, tid)
        total += b
        tname = GGML_TYPES[tid][0]
        by_type[tname] = by_type.get(tname, [0, 0])
        by_type[tname][0] += 1
        by_type[tname][1] += b
        if "_exps." in nm:
            cls, a = "routed-experts", b * n_top / n_expert
        elif "shexp" in nm:
            cls, a = "shared-expert", float(b)
        elif nm.startswith("token_embd."):
            if has_output:
                # one row lookup: dims[0] = hidden, dims[1] = vocab
                be, bb = GGML_TYPES[tid][1], GGML_TYPES[tid][2]
                cls, a = "embed(row)", dims[0] // be * bb
            else:
                cls, a = "embed=head(full)", float(b)
        else:
            cls, a = "dense(full)", float(b)
        e = by_class.setdefault(cls, [0, 0, 0.0])
        e[0] += 1
        e[1] += b
        e[2] += a
        active += a

    print(f"\nfile tensor bytes total: {total/1e9:.3f} GB ({total/2**30:.3f} GiB)")
    print(f"{'class':<18}{'#':>5}{'stored GB':>12}{'active GB/tok':>15}")
    for cls, (cnt, sb, ab) in sorted(by_class.items()):
        print(f"{cls:<18}{cnt:>5}{sb/1e9:>12.3f}{ab/1e9:>15.4f}")
    print(f"{'TOTAL':<18}{'':>5}{total/1e9:>12.3f}{active/1e9:>15.4f}")
    print(f"\nquant mix: " + ", ".join(f"{t}:{c}({b/1e9:.2f}GB)" for t, (c, b) in
                                       sorted(by_type.items(), key=lambda x: -x[1][1])))
    for bw in (546e9,):
        print(f"\nweights-only decode ceiling @ {bw/1e9:.0f} GB/s: "
              f"{bw/active:.1f} tok/s  ({active/1e9:.3f} GB/token)")

if __name__ == "__main__":
    main(sys.argv[1])
