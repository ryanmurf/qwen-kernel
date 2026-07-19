# Gemma 4 Stage 0/1 artifacts

Stage 0 is complete and Stage 1 contains standalone loader and quant-kernel
gates. No Gemma model graph, attention, MoE scheduling, or 30-layer assembly is
executed by these harnesses.

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
