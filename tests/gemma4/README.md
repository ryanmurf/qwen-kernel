# Gemma 4 Stage 0--4 artifacts

Stages 0--4 contain the loader, native quant kernels, dense primitives, both
attention geometries, and the shared-plus-routed sparse block. Full 30-layer
serial assembly and fixture continuation parity remain Stage 5.

## Artifact ledger

Regenerate and validate the complete ledger (SHA-256, all 52 metadata KVs, and
all 658 tensor records):

```bash
rtk proxy tests/gemma4/generate_manifest.py \
  /mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf
```

`manifest.json` is about 143 KiB (146,073 bytes), not 27 MB. Metadata arrays
longer than 64 entries are stored as count plus SHA-256 over canonical JSON;
the load-bearing 30-entry sliding-window pattern remains inline.

## Numeric parity fixtures

The reference is llama.cpp
`571d0d540df04f25298d0e159e520d9fc62ed121`, with the local
`common/debug.cpp` `QK_DUMP_DIR`/`QK_DUMP_FILTER` patch present. The model is
frozen by SHA-256
`3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.

Each fixture stores its numeric input IDs, numeric greedy continuation IDs,
exact commands, reference commit/model SHA, sampler settings, and the identical
continuation SHA from two independent runs. Text output is not authoritative.

| fixture | input IDs | continuation IDs | purpose |
|---|---:|---:|---|
| `ordinary_chat` | 28 | 32 | ordinary chat |
| `coding_prompt` | 43 | 32 | coding prompt |
| `swa_position_1023` | 1023 | 16 | first continuation at position 1023 |
| `swa_position_1024` | 1024 | 16 | first wrapped position |
| `swa_position_1025` | 1025 | 16 | post-wrap position |
| `global_context_8192` | 8192 | 16 | five global-attention layers at 8K |

The generator uses `/apply-template` and `/tokenize` once, then sends numeric
arrays to `/completion` twice with temperature zero, penalties/DRY/XTC off,
`cache_prompt=false`, `ignore_eos=true`, and `return_tokens=true`. It writes no
fixture unless both continuations agree exactly.

```bash
GGML_VK_VISIBLE_DEVICES=2 rtk proxy \
  /mnt/data/llama.cpp-master/build/bin/llama-server \
  -m /mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf \
  -dev Vulkan0 -sm none -mg 0 -ngl 99 -fa on -ctk f16 -ctv f16 \
  -fit off -c 16384 -b 8192 -ub 512 --parallel 1 \
  --host 127.0.0.1 --port 18271

rtk proxy tests/gemma4/generate_fixtures.py \
  --server http://127.0.0.1:18271
```

## Build and validation

```bash
rtk proxy /usr/bin/cmake -B build
rtk proxy /usr/bin/cmake --build build -j

for module in gemv_q4_0 gemm_q4_0 gemm_q4_0_bn32 gemv_q6_k argmax1 argmax2; do
  rtk proxy spirv-val --target-env vulkan1.2 "build/shaders/$module.spv"
done

QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf \
  rtk proxy ./build/qk gemma4-load
```

The loader assigns all 658 text tensors to explicit Gemma roles, recognizes
native 18-byte Q4_0 blocks and the tied Q6_K head, validates encoded ranges and
shapes, enforces global-layer `attn_v` absence, and maps no vision projector.

## Standalone Stage 1 kernels

Q4 GEMV and both GEMMs use the same `q4_0_dequant_pair` routine. Inner block
counts are pipeline specializations for K=704/2112/2816. BN32 and BN64 GEMM
modules both use BM128/BK64/local256.

Repeated small tensors fit in Infinity Cache, so performance runs must cycle
distinct descriptor ranges spanning at least 128 MiB:

```bash
MODEL=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf

QK_DEVICE_PCI=1a:00.0 QK_GGUF="$MODEL" QK_COLD_MIB=128 QK_TPR=32 \
  rtk proxy ./build/qk gemma4-q4-gemv 2000
QK_DEVICE_PCI=1a:00.0 QK_GGUF="$MODEL" QK_COLD_MIB=128 \
  rtk proxy ./build/qk gemma4-q4-gemm 32 1000
QK_DEVICE_PCI=1a:00.0 QK_GGUF="$MODEL" QK_COLD_MIB=128 \
  rtk proxy ./build/qk gemma4-q4-gemm 64 1000
QK_DEVICE_PCI=1a:00.0 QK_GGUF="$MODEL" QK_TPR=16 \
  rtk proxy ./build/qk gemma4-head 2000
```

Check `/sys/class/drm/card2/device/gpu_busy_percent` immediately before every
timing run. Full sweeps, workgroups, cold/hot ratios, GEMM results, and Q6_K
argmax timing are in `stage1-results.json` and `docs/GEMMA4-LOG.md`.

## Stage 2--4 graph gates

The standalone graph gates use the exact `gemma4.cpp` order. On AMD, llama.cpp
uses Q4×Q8_1 MMVQ for the 2816- and 2112-wide projections, but retains
Q4×F32 for the 704-wide expert-down projection; the harness reproduces that
split for dump parity.

```bash
MODEL=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf

QK_DEVICE_PCI=1a:00.0 QK_GGUF="$MODEL" \
  rtk proxy ./build/qk gemma4-stage2
QK_DEVICE_PCI=1a:00.0 QK_GGUF="$MODEL" \
  rtk proxy ./build/qk gemma4-stage3
QK_DEVICE_PCI=1a:00.0 QK_GGUF="$MODEL" \
  rtk proxy ./build/qk gemma4-stage4 2000
```

Set `QK_G4_ORACLE_DIR` to a llama-debug tensor-dump directory to add real
llama.cpp gates. The verified dump set used physical ubatch 1 (`-b 2 -ub 1`)
so every file's final row is the second token at position 1.

Stage 3 checks positions 0, 1, 1023, 1024, and 1025 for the 1024-entry sliding
ring and the linear global cache, across layer 0 and all five global layers.
It checks raw projections, learned Q/K norms, unweighted global V norm, RoPE,
separate K/V cache contents, probabilities, attention outputs, output
projection, and the attention residual hidden state. Stage 4 checks stable
top-8 selection (including ties), every shared and routed branch tensor, and a
complete sparse block.

The grouped benchmark cycles 16 disjoint routes across all 128 experts. Its
headline 2,000-iteration XTX result is TPR64 for both weight dispatches:
521.6 GB/s gate_up (+25.3% versus isolated 416.4) and 333.1 GB/s down (+28.3%
versus isolated 259.5), with `gpu_busy_percent=0` before every timing. The pair
reaches 451.9 GB/s, not the assumed 700 GB/s. Full gates and timing records are
in `stage2-4-results.json`.

## Q4_0 large-stream calibration

The earlier 288 MiB, 2000-iteration calibration remains the upper bound:
XTX 801.5 GB/s and XT 558.3 GB/s, both at TPR32 and both correctness PASS.
Real 1–13 MiB shapes are substantially below this when measured cache-cold.

## Qwen `serve-bench` side note

`ids2.txt` and `ids4.txt` were valid. `qk_slot_start` leaves a short serial
prefill tail (13 and 15 tokens respectively), while `serve-bench` incorrectly
treated a valid zero-output prefill step as a stop. The driver now tolerates
bounded zero-output progress. With `QK_CHUNK=1`, ids2 emits reference token 198
and ids4 emits reference token 13 instead of reporting `produced=0`.
