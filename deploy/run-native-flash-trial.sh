#!/usr/bin/env bash
# Experimental native qwen4exp split. Neither replaces nor stops production.
# Start worker first, then server. Both ports bind loopback by default.
set -euo pipefail
mode=${1:?usage: run-native-flash-trial.sh worker\|server MODEL.gguf [CONTEXT]}
model=${2:?GGUF first shard required}
context=${3:-8192}
case "$mode" in worker|server) ;; *) echo 'mode must be worker or server' >&2; exit 2 ;; esac
if [[ ! "$context" =~ ^[0-9]+$ ]] || (( context<64 || context>32768 )); then
    echo 'context must be 64..32768' >&2; exit 2
fi
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
test -f "$model" || { echo 'GGUF not found' >&2; exit 1; }
export QK_NATIVE_FLASH=1 QK_GGUF="$model" QK_SHADER_DIR="$project_root/build-halo/shaders"
# Do not inherit a stale device index/PCI override that defeats unique names.
unset QK_DEVICE QK_DEVICE_PCI QK_LAYER_DUMP
export QK_REASONING_EFFORT=${QK_REASONING_EFFORT:-xhigh}
export QK_PREFILL_CHUNK=${QK_PREFILL_CHUNK:-512}
# Page-cache warming of the 35.763 GiB PLE table is OFF unless the operator
# sets QK_PLE_PREFETCH=1 (needs MemoryHigh>=52G on the server unit and no
# other GPU loads; see docs/STRIX-HALO.md, memory plan and incident notes).
export QK_PLE_PREFETCH=${QK_PLE_PREFETCH:-0}
# The cooperative-matrix prefill tier stays opt-in (QK_FLASH_COOPMAT=1).
export QK_FLASH_COOPMAT=${QK_FLASH_COOPMAT:-0}
if [[ "$mode" == worker ]]; then
    export QK_DEVICE_NAME=NAVI31 QK_PIPE_HOST=127.0.0.1
    exec "$project_root/build-halo/qk" pipe-worker 8195 37:48 "$context" 1
fi
export QK_DEVICE_NAME=STRIX_HALO QK_LAYERS=0:37
exec "$project_root/build-halo/rust/release/server" \
    --model "$model" --engine-lib "$project_root/build-halo/libqk.so" \
    --host 127.0.0.1 --port 8194 --slots 1 --ctx "$context" --chunk 1 \
    --queue 4 --split-next 127.0.0.1:8195 --chat-template auto
