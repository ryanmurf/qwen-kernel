#!/usr/bin/env bash
# Max-only comparison launcher. Refuses overlap; never stops existing services.
# Usage: bash deploy/start-llama-prefill-halo.sh fast|f32|mtp [CONTEXT=32768]
# fast: F16 KV + flash attention, no MTP. f32: conservative F32, no MTP.
# mtp: F16 KV + flash attention + shared MTP draft and image projector.
set -euo pipefail
mode=${1:?usage: start-llama-prefill-halo.sh fast\|f32\|mtp [CONTEXT]}
context=${2:-32768}
case "$mode" in fast|f32|mtp) ;; *) echo 'mode must be fast, f32, or mtp' >&2; exit 2;; esac
if ! [[ "$context" =~ ^[1-9][0-9]{2,4}$ ]] || ((context < 512 || context > 32768)); then
    echo 'comparison context must be 512..32768' >&2; exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
binary=/home/ryan/llama-qwen-next-b10685/build-vulkan/bin/llama-server
target=/home/ryan/models/Qwen3.8-Flash-Next-Uncensored-Q5_K_M
model="$target/Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf"
draft=/home/ryan/models/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
projector="$target/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"
halo=/sys/bus/pci/devices/0000:c1:00.0
unit="qk-prefill-llama-$mode.service"
test -x "$binary"
test "$(< "$halo/vendor")" = 0x1002
test "$(< "$halo/device")" = 0x1586
for entry in \
    'Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf:44537339136' \
    'Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00002-of-00003.gguf:44714628064' \
    'Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00003-of-00003.gguf:44854700960'; do
    test "$(stat -c %s "$target/${entry%:*}")" = "${entry##*:}"
done
if [[ "$mode" == mtp ]]; then
    test "$(stat -c %s "$draft")" = 2786568256
    test "$(stat -c %s "$projector")" = 907543296
fi
for name in qwen-native-flash-router qwen-native-flash-server32 qwen-native-flash-worker32 \
    qwen-kernel-next qwen-kernel-next-router qwen-kernel-prefill-router \
    qwen-kernel-prefill-xtx qwen-kernel-decode-halo \
    qk-prefill-llama-fast qk-prefill-llama-f32 qk-prefill-llama-mtp; do
    if systemctl --user is-active --quiet "$name"; then
        echo "$name is active; stop and drain competing model services first" >&2; exit 1
    fi
done
for p in /proc/[0-9]*; do
    exe=$(readlink "$p/exe" 2>/dev/null) || continue
    case "$exe" in
        */llama-server|*/llama-server\ \(deleted\)|"$root/build-halo/qk"|"$root/build-halo/rust/release/server"|"$root/build-halo/qk (deleted)"|"$root/build-halo/rust/release/server (deleted)")
            echo "model process ${p#/proc/} is still running: $exe" >&2; exit 1;;
    esac
done
if (( $(< "$halo/mem_info_gtt_used") > 2*1024*1024*1024 )); then
    echo 'Halo GTT is not drained below 2 GiB; do not load another model' >&2; exit 1
fi
if ss -H -ltn 'sport = :8193' | grep -q .; then
    echo 'port 8193 is already in use' >&2; exit 1
fi
python3 "$root/deploy/release-model-cache.py" "$model"
python3 "$root/deploy/check-model-headroom.py" \
    --minimum-available-gib 24 --maximum-other-memory-gib 12

# No inherited model/device/profiler knobs or credentials in this process.
# Keep the normal Mesa disk cache; no OS-wide cache flushing is performed.
clean_env=(/usr/bin/env -i PATH=/usr/bin:/bin XDG_RUNTIME_DIR="/run/user/$(id -u)"
    XDG_CACHE_HOME=/home/ryan/.cache GGML_VK_ALLOW_SYSMEM_FALLBACK=1)
devices=$("${clean_env[@]}" "$binary" --list-devices 2>&1)
mapfile -t matches < <(printf '%s\n' "$devices" | sed -n '/STRIX_HALO/s/^[[:space:]]*\(Vulkan[0-9][0-9]*\):.*/\1/p')
if (( ${#matches[@]} != 1 )); then
    echo 'expected exactly one STRIX_HALO Vulkan device' >&2; exit 1
fi
device=${matches[0]}
printf '%s\n' "$devices"
# Limit ggml's visible backends too: its default host-staging allocator can
# otherwise choose the enumerated XTX even when all model ops target Halo.
# Validate the physical-index interpretation before loading any weights.
clean_env+=(GGML_VK_VISIBLE_DEVICES="${device#Vulkan}")
visible=$("${clean_env[@]}" "$binary" --list-devices 2>&1)
mapfile -t selected < <(printf '%s\n' "$visible" | sed -n 's/^[[:space:]]*\(Vulkan[0-9][0-9]*:.*\)$/\1/p')
if (( ${#selected[@]} != 1 )) || [[ "${selected[0]}" != *STRIX_HALO* ]]; then
    echo 'filtered physical device is not uniquely Halo; refusing model load' >&2; exit 1
fi
device=${selected[0]%%:*}
printf '%s\n' "$visible"
kv=f16; flash=on
extra=(--spec-type none)
if [[ "$mode" == f32 ]]; then
    kv=f32; flash=off
    clean_env+=(GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_MMVQ=1)
elif [[ "$mode" == mtp ]]; then
    extra=(--mmproj "$projector" -md "$draft" --spec-draft-device "$device" --spec-draft-ngl 99
        --image-min-tokens 1024 --spec-type draft-mtp --spec-draft-adaptive
        --spec-draft-n-min 0 --spec-draft-n-max 5 --spec-draft-p-min 0.75 -ctkd f16 -ctvd f16)
fi
args=(-m "$model" --device "$device" --split-mode none -ngl 99 --n-cpu-moe 0
    --fit off -c "$context" --parallel 1 -b 2048 -ub 1024 -fa "$flash" -ctk "$kv" -ctv "$kv"
    --load-mode mmap --tensor-read-lazy auto --no-repack --no-host
    --jinja --reasoning-effort xhigh --reasoning-preserve
    --temp 1 --top-k 20 --top-p 0.95 --min-p 0 --repeat-penalty 1 --presence-penalty 0
    --cache-ram 0 --ctx-checkpoints 0 --no-cache-idle-slots
    --host 127.0.0.1 --port 8193 --alias qwen3.8-flash-next-llama-bench
    --cors-origins localhost --no-cors-credentials --no-webui --metrics "${extra[@]}")
printf 'launch unit=%s mode=%s device=%s context=%s\n' "$unit" "$mode" "$device" "$context"
printf 'command:'; printf ' %q' "${clean_env[@]}" "$binary" "${args[@]}"; printf '\n'

# These cap the process/page cache, NOT all GPU GTT. Monitor host memory too.
systemd-run --user --unit="$unit" --collect -p MemoryHigh=8G -p MemoryMax=12G \
    -p MemorySwapMax=256M -p NoNewPrivileges=yes -p LimitCORE=0 -p TasksMax=256 \
    -p TimeoutStopSec=30 -p WorkingDirectory="$root" \
    "${clean_env[@]}" "$binary" "${args[@]}"
echo "started $unit; inspect its journal, placement and /health before benchmarking"
echo "stop only this comparison with: systemctl --user stop $unit"
