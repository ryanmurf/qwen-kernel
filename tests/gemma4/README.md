# Gemma 4 Stage 0 artifacts

Stage 0 is partially landed and currently blocked on GPU device access in the
execution sandbox used on 2026-07-18. The sandbox has no `/dev/dri`; both qk and
the supplied llama.cpp build therefore enumerate only llvmpipe/no Vulkan GPU.
Per the Stage 0 stop gate, no bandwidth or llama.cpp parity/baseline numbers
were recorded from that environment.

## Artifact ledger

Regenerate and validate the complete ledger (including SHA-256, all 52 metadata
KVs, and all 658 tensor records):

```bash
rtk proxy tests/gemma4/generate_manifest.py \
  /mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf
```

`manifest.json` is about 140 KB. Metadata arrays longer than 64 entries (the
262,144-entry tokenizer `tokens`/`merges`/`scores`/`token_type`) are stored as a
count plus a SHA-256 over their canonical JSON encoding rather than verbatim:
that still detects any change to the array, but keeps the file diffable and
avoids committing 18 MB of data already present in the GGUF. Short arrays —
including the load-bearing 30-entry `sliding_window_pattern` — stay inline. Any
failed invariant prints `actual=...` and `expected=...` and exits nonzero.

## Build and Q4_0 measurement

Build with the system CMake and validate the new measurement-only shader:

```bash
rtk proxy /usr/bin/cmake -B build
rtk proxy /usr/bin/cmake --build build -j
rtk proxy spirv-val --target-env vulkan1.2 build/shaders/gemv_q4_0.spv
```

On a host with `/dev/dri` access, first confirm both targets are idle. Then run
the 288 MiB Q4_0 matrix for 2,000 GPU-timestamped iterations in forward and
reversed card order:

```bash
rtk proxy cat /sys/class/drm/card2/device/gpu_busy_percent
rtk proxy cat /sys/class/drm/card1/device/gpu_busy_percent

QK_DEVICE_PCI=1a:00.0 rtk proxy ./build/qk q4_0 32768 16384 2000
QK_DEVICE_PCI=03:00.0 rtk proxy ./build/qk q4_0 32768 16384 2000
QK_DEVICE_PCI=03:00.0 rtk proxy ./build/qk q4_0 32768 16384 2000
QK_DEVICE_PCI=1a:00.0 rtk proxy ./build/qk q4_0 32768 16384 2000
```

`runGemv` uploads weights to a device-local buffer, verifies GPU output against
CPU dequantization, performs an unreported warm run, and reports the second
run from Vulkan GPU timestamps. The matrix contains 301,989,888 encoded weight
bytes (288 MiB), larger than the XTX's 96 MiB Infinity Cache. Do not record a
run unless the output says `PASS` and `(VRAM, ...)`.

The comparison anchors are XTX F16 917.8, Q8_0 927.8, Q6_K 800.4, and IQ4_XS
729.0 GB/s. No Q4_0 number is present yet: the only attempted target run failed
before allocation because `QK_DEVICE_PCI=1a:00.0` matched no Vulkan device.

## llama.cpp parity fixtures and baseline

The reference revision is
`a935fbffe1a3d31509c325c116454ab5d56b2eb8`; binaries are under
`/mnt/data/llama.cpp-master/build/bin`. The local source tree has a pre-existing
modification to `common/debug.cpp`, so record that dirty state along with the
revision. The model object itself is frozen by `manifest.json`.

The baseline command shape, once Vulkan devices are visible, is:

```bash
MODEL=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf
BENCH=/mnt/data/llama.cpp-master/build/bin/llama-bench

GGML_VK_VISIBLE_DEVICES=2 rtk proxy "$BENCH" -m "$MODEL" -ngl 99 \
  -dev Vulkan0 -p 512 -n 128 -d 0,4096,16384 -r 5 -o jsonl
GGML_VK_VISIBLE_DEVICES=1 rtk proxy "$BENCH" -m "$MODEL" -ngl 99 \
  -dev Vulkan0 -p 512 -n 128 -d 0,4096,16384 -r 5 -o jsonl
```

Capture clocks, junction temperature, peak VRAM, Mesa/RADV version, and device
UUID beside each run; report medians and spreads rather than silently replacing
the earlier 3505.04/139.36 XTX and 2389.09/119.34 XT results.

`fixtures/` contains no numeric fixtures yet. Creating them requires two clean,
GPU-backed llama.cpp runs per prompt and exact continuation-ID agreement. The
blocked sandbox cannot perform those runs, and a single run or text-derived
substitute would violate the fixture gate. Required cases remain ordinary chat,
coding, window positions 1023/1024/1025, and a context of at least 8K tokens if
runtime permits.

