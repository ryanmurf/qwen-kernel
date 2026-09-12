#!/usr/bin/env bash
# Safe (re)start of the native Flash Next split on Max as transient user units.
#
#   deploy/restore-native-flash.sh MODEL.gguf [CONTEXT=32768] [WARM=0|1] [MODE=split|single]
#
# MODE=single runs every layer and the head on the Strix Halo iGPU alone
# (two-heap placement inside libqk, --local-driver); no XTX unit is started
# and WARM must stay 0 there (the 36 GiB table warm does not fit beside the
# 90 GiB single-device model).
#
# Order and checks (see docs/STRIX-HALO.md, memory plan and incident notes):
#   1. refuse while any qk/server process runs or a GPU still holds memory;
#   2. release the model shards' page cache (targeted fadvise) so the driver
#      does not have to allocate next to tens of GiB of resident cache;
#   3. XTX worker (layers 37:48 + head, loopback 8195), wait for "listening";
#   4. Halo server (layers 0:37, HTTP loopback 8194), wait for /health;
#      with WARM=1 the unit gets 52G/58G limits and warms the 36 GiB of
#      lookup tables only after both stages are loaded (nothing else may
#      load on a GPU while they stay resident);
#   5. trusted-LAN router on 8091 from the node's existing prefill-router.py.
# Stop order is the reverse: router, server, worker. Nothing is enabled at boot.
set -euo pipefail
model=${1:?usage: restore-native-flash.sh MODEL.gguf [CONTEXT] [WARM]}
context=${2:-32768}
warm=${3:-0}
mode=${4:-split}
[[ "$mode" == split || "$mode" == single ]] || { echo 'MODE must be split or single' >&2; exit 2; }
if [[ "$mode" == single && "$warm" == 1 ]]; then echo 'WARM=1 is not supported in single mode' >&2; exit 2; fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
router=/home/ryan/IdeaProjects/qwen-kernel/deploy/prefill-router.py
halo=/sys/bus/pci/devices/0000:c1:00.0
xtx=/sys/bus/pci/devices/0000:68:00.0
mib() { echo $(( $(cat "$1") / 1048576 )); }
kb() { awk -v k="$1" '$1==k":" {print $2}' /proc/meminfo; }
state() {
    echo "MemFree=$(( $(kb MemFree)/1048576 ))G MemAvailable=$(( $(kb MemAvailable)/1048576 ))G Cached=$(( $(kb Cached)/1048576 ))G" \
         "halo_gtt=$(mib $halo/mem_info_gtt_used)MiB halo_vram=$(mib $halo/mem_info_vram_used)MiB xtx_vram=$(mib $xtx/mem_info_vram_used)MiB"
}
test -f "$model" || { echo "GGUF not found: $model" >&2; exit 1; }
test -f "$router" || { echo "router script not found: $router" >&2; exit 1; }
[[ "$warm" == 0 || "$warm" == 1 ]] || { echo 'WARM must be 0 or 1' >&2; exit 2; }
for unit in qwen-native-flash-router qwen-native-flash-server32 qwen-native-flash-worker32; do
    if systemctl --user is-active --quiet "$unit"; then echo "$unit is already active; stop it first" >&2; exit 1; fi
done
if pgrep -f 'build-halo/(qk|rust/release/server)' >/dev/null; then echo "a qk/server process is still running" >&2; exit 1; fi
if (( $(mib $halo/mem_info_gtt_used) > 2048 || $(mib $xtx/mem_info_vram_used) > 2048 )); then
    echo "GPU memory not drained: $(state)" >&2; exit 1
fi
if ps -eo stat,comm | awk '$1 ~ /D/ && $2 ~ /qk|server|python3/' | grep -q .; then
    echo "a GPU process is stuck in uninterruptible sleep; do not load" >&2; exit 1
fi
echo "pre-load: $(state)"
python3 "$root/deploy/release-model-cache.py" "$model" | tail -1
echo "after cache release: $(state)"
if (( $(kb MemFree) < 30*1048576 )); then
    echo "warning: MemFree below 30 GiB before the Halo load; the driver reuses its retained pool but this is not guaranteed" >&2
fi

if [[ "$mode" == split ]]; then
    systemd-run --user --unit=qwen-native-flash-worker32 -p MemoryHigh=12G -p MemoryMax=20G -p MemorySwapMax=512M \
        -p NoNewPrivileges=yes -p LimitCORE=0 /usr/bin/bash "$root/deploy/run-native-flash-trial.sh" worker "$model" "$context"
    for _ in $(seq 1 60); do
        sleep 3
        journalctl --user -u qwen-native-flash-worker32 --no-pager -n 5 | grep -q 'listening on' && break
        systemctl --user is-active --quiet qwen-native-flash-worker32 || { echo 'worker failed' >&2; exit 1; }
    done
    echo "worker up: $(state)"
fi

# Single mode: the process itself stays small (uploaded weight pages are
# dropped as they go); the limit mostly bounds the on-demand PLE row page
# cache charged to the unit, so it is a working-set cap, not a weight budget.
if [[ "$mode" == single ]]; then high=24G; max=32G;
elif [[ "$warm" == 1 ]]; then high=52G; max=58G; else high=16G; max=24G; fi
server_started=$(date +%s)
systemd-run --user --unit=qwen-native-flash-server32 -p MemoryHigh=$high -p MemoryMax=$max -p MemorySwapMax=512M \
    -p NoNewPrivileges=yes -p LimitCORE=0 --setenv=QK_PLE_PREFETCH="$warm" \
    /usr/bin/bash "$root/deploy/run-native-flash-trial.sh" "$([[ "$mode" == single ]] && echo single || echo server)" "$model" "$context"
for _ in $(seq 1 100); do
    sleep 3
    curl -s -m 2 http://127.0.0.1:8194/health | grep -q ok && break
    systemctl --user is-active --quiet qwen-native-flash-server32 || { echo 'server failed' >&2; exit 1; }
done
echo "server up: $(state)"
if [[ "$warm" == 1 ]]; then
    # Only this invocation's journal counts: an earlier instance's line
    # would otherwise end the wait before the tables are resident.
    for _ in $(seq 1 80); do
        sleep 5
        journalctl --user -u qwen-native-flash-server32 --no-pager --since "@$server_started" | grep -q 'page cache:' && break
    done
    if journalctl --user -u qwen-native-flash-server32 --no-pager --since "@$server_started" | grep 'page cache:' | tail -1 | cut -c1-200 | grep .; then
        echo "warmed: $(state)"
    else
        echo "warning: warming did not report completion within 400 s; the tables may still be paging in" >&2
    fi
fi

systemd-run --user --unit=qwen-native-flash-router -p NoNewPrivileges=yes -p UMask=0077 -p LimitCORE=0 -p TasksMax=64 \
    -p MemoryHigh=768M -p MemoryMax=1G -p Requires=qwen-native-flash-server32.service -p After=qwen-native-flash-server32.service \
    /usr/bin/python3 -u "$router" --host 0.0.0.0 --allow-network 192.168.0.0/24 --port 8091 \
    --prefill-url http://127.0.0.1:8194 --decode-url http://127.0.0.1:8194 --decode-slots 1 \
    --threshold 2147483647 --max-prefill-tokens 2147483647
sleep 3
curl -s -m 5 http://127.0.0.1:8091/health; echo
echo "done: $(state)"
