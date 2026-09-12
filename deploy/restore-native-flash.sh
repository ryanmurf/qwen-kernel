#!/usr/bin/env bash
# Safe (re)start of the native Flash Next engine on Max as transient user units.
#
#   deploy/restore-native-flash.sh MODEL.gguf [CONTEXT=32768] [WARM=0|1] [MODE=single|split]
#
# MODE=single (default): every layer and the head on the Strix Halo iGPU
# alone (two-heap placement inside libqk, --local-driver); no XTX unit is
# started or probed. WARM must stay 0 (the 36 GiB table warm does not fit
# beside the ~90 GiB single-device model; libqk rejects it anyway).
# MODE=split (explicit opt-in only): XTX worker 37:48 + head, Halo server 0:37.
#
# Order and checks (see docs/STRIX-HALO.md, memory plan and incident notes):
#   1. validate arguments, then refuse while any qk/server process runs, a
#      GPU process is stuck, or a GPU still holds memory;
#   2. release the model shards' page cache (targeted fadvise) so the driver
#      does not have to allocate next to tens of GiB of resident cache;
#      require host headroom, including RAM/swap still owned by other jobs;
#   3. (split) XTX worker, wait for this invocation's "listening" line;
#   4. server on HTTP loopback 8194, wait for a 200 /health with status ok;
#      with WARM=1 (split only) the unit gets 52G/58G limits and warms the
#      tables only after both stages are loaded;
#   5. trusted-LAN router on 8091 from the node's existing prefill-router.py.
# Any readiness deadline that expires fails the script explicitly and stops
# only the units this invocation started. Stop order is router, server,
# worker. Nothing is enabled at boot.
set -euo pipefail
usage() { echo "usage: restore-native-flash.sh MODEL.gguf [CONTEXT=32768] [WARM=0|1] [MODE=single|split]" >&2; exit 2; }
model=${1:-}; [[ -n "$model" ]] || usage
context=${2:-32768}
warm=${3:-0}
mode=${4:-single}
if ! [[ "$context" =~ ^[0-9]+$ ]] || (( context < 64 || context > 32768 )); then
    echo 'CONTEXT must be an integer 64..32768' >&2; exit 2
fi
[[ "$warm" == 0 || "$warm" == 1 ]] || { echo 'WARM must be 0 or 1' >&2; exit 2; }
[[ "$mode" == split || "$mode" == single ]] || { echo 'MODE must be single or split' >&2; exit 2; }
if [[ "$mode" == single && "$warm" == 1 ]]; then echo 'WARM=1 is not supported in single mode' >&2; exit 2; fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
router=/home/ryan/IdeaProjects/qwen-kernel/deploy/prefill-router.py
halo=/sys/bus/pci/devices/0000:c1:00.0
xtx=/sys/bus/pci/devices/0000:68:00.0
test -f "$model" || { echo "GGUF not found: $model" >&2; exit 1; }
test -f "$router" || { echo "router script not found: $router" >&2; exit 1; }
test -d "$halo" || { echo "Halo device $halo not present" >&2; exit 1; }
if [[ "$mode" == split ]] && ! test -d "$xtx"; then echo "split mode needs the XTX at $xtx" >&2; exit 1; fi

mib() { echo $(( $(cat "$1") / 1048576 )); }
kb() { awk -v k="$1" '$1==k":" {print $2}' /proc/meminfo; }
state() {
    local s="MemFree=$(( $(kb MemFree)/1048576 ))G MemAvailable=$(( $(kb MemAvailable)/1048576 ))G Cached=$(( $(kb Cached)/1048576 ))G"
    s+=" halo_gtt=$(mib $halo/mem_info_gtt_used)MiB halo_vram=$(mib $halo/mem_info_vram_used)MiB"
    if [[ "$mode" == split ]]; then s+=" xtx_vram=$(mib $xtx/mem_info_vram_used)MiB"; fi
    echo "$s"
}
started=()
cleanup_on_failure() {
    local rc=$?
    if (( rc != 0 )) && (( ${#started[@]} > 0 )); then
        echo "failure (rc=$rc): stopping units started by this invocation: ${started[*]}" >&2
        for (( i=${#started[@]}-1; i>=0; i-- )); do systemctl --user stop "${started[$i]}" || true; done
    fi
    exit $rc
}
trap cleanup_on_failure EXIT
start_unit() {  # name, then systemd-run arguments
    local name=$1; shift
    systemd-run --user --unit="$name" "$@" >/dev/null
    started+=("$name")
}
health_ok() { curl -sf -m 3 "$1" 2>/dev/null | grep -q '"status":"ok"'; }

for unit in qwen-native-flash-router qwen-native-flash-server32 qwen-native-flash-worker32; do
    if systemctl --user is-active --quiet "$unit"; then echo "$unit is already active; stop it first" >&2; exit 1; fi
done
# Live native processes are identified by executable identity (/proc/PID/exe),
# never by command-line text, so a shell whose arguments mention the binaries
# (a test chain, an editor) cannot trip this guard.
live=""
for p in /proc/[0-9]*; do
    exe=$(readlink "$p/exe" 2>/dev/null) || continue
    case "$exe" in
        "$root/build-halo/qk"|"$root/build-halo/qk (deleted)"|"$root/build-halo/rust/release/server"|"$root/build-halo/rust/release/server (deleted)")
            live+="${p#/proc/} ";;
    esac
done
if [[ -n "$live" ]]; then echo "a native qk/server process is still running (pid $live); stop it first" >&2; exit 1; fi
# Only OUR GPU processes count as stuck (a python3 anywhere in D state for
# ordinary I/O must not block a restart): match by command line.
stuck=$(ps -eo pid,stat,args | awk '$2 ~ /D/ && ($0 ~ /build-halo\/(qk|rust\/release\/server)/ || $0 ~ /tests\/gpu_qwen4_/) {print $1}')
if [[ -n "$stuck" ]]; then
    echo "a native GPU process is in uninterruptible sleep (pid $stuck): still loading or stuck; do not load beside it" >&2; exit 1
fi
if (( $(mib $halo/mem_info_gtt_used) > 2048 )); then echo "Halo memory not drained: $(state)" >&2; exit 1; fi
if [[ "$mode" == split ]] && (( $(mib $xtx/mem_info_vram_used) > 2048 )); then echo "XTX memory not drained: $(state)" >&2; exit 1; fi
echo "pre-load: $(state)"
python3 "$root/deploy/release-model-cache.py" "$model" | tail -1
echo "after cache release: $(state)"
# Retained, unused TTM pages may be reusable, but another job's anonymous
# RAM or swap is not free capacity for the ~90 GiB single-Halo model.
# This is a read-only admission snapshot, not a reservation against jobs
# that start allocating after launch. A rejection starts no model units.
python3 "$root/deploy/check-model-headroom.py"
if (( $(kb MemFree) < 30*1048576 )); then
    echo "warning: MemFree below 30 GiB before the Halo load; the driver reuses its retained pool but this is not guaranteed" >&2
fi

if [[ "$mode" == split ]]; then
    worker_started=$(date +%s)
    start_unit qwen-native-flash-worker32 -p MemoryHigh=12G -p MemoryMax=20G -p MemorySwapMax=512M \
        -p NoNewPrivileges=yes -p LimitCORE=0 /usr/bin/bash "$root/deploy/run-native-flash-trial.sh" worker "$model" "$context"
    ready=0
    for _ in $(seq 1 60); do
        sleep 3
        if journalctl --user -u qwen-native-flash-worker32 --no-pager --since "@$worker_started" | grep -q 'listening on'; then ready=1; break; fi
        systemctl --user is-active --quiet qwen-native-flash-worker32 || { echo 'worker unit exited' >&2; exit 1; }
    done
    (( ready )) || { echo 'worker did not become ready within 180 s' >&2; exit 1; }
    echo "worker up: $(state)"
fi

# Single mode: the process itself stays small (uploaded weight pages are
# dropped as they go); the limit mostly bounds the on-demand PLE row page
# cache charged to the unit, so it is a working-set cap, not a weight budget.
if [[ "$mode" == single ]]; then high=24G; max=32G;
elif [[ "$warm" == 1 ]]; then high=52G; max=58G; else high=16G; max=24G; fi
server_started=$(date +%s)
# Optional A/B and rollback knobs forwarded into the unit when set in the
# caller's environment (systemd units do not inherit it): QK_FLASH_BATCH (0 =
# serial prefill), QK_PLE_ROW_PREFETCH (0 = no per-row prefetch),
# QK_FLASH_COOPMAT, QK_FLASH_GEMM (compact), QK_FLASH_ATTN_BATCH (vec4),
# QK_FLASH_FUSE (0 = separate dispatches), QK_MOE_GU (v1 =
# byte-addressed expert kernels), QK_GDN_STEP (v1). Profiling variables are
# deliberately NOT forwarded: HTTP measurements run with profiling off.
extra=()
for knob in QK_FLASH_BATCH QK_PLE_ROW_PREFETCH QK_FLASH_COOPMAT QK_FLASH_GEMM QK_FLASH_ATTN_BATCH QK_FLASH_FUSE QK_MOE_GU QK_GDN_STEP QK_Q51_GEMV QK_MOE_DOWN QK_Q6K_GEMV QK_ATTN_DECODE QK_ATTN_CHUNK; do
    if [[ -n "${!knob:-}" ]]; then extra+=("--setenv=$knob=${!knob}"); fi
done
start_unit qwen-native-flash-server32 -p MemoryHigh=$high -p MemoryMax=$max -p MemorySwapMax=512M \
    -p NoNewPrivileges=yes -p LimitCORE=0 --setenv=QK_PLE_PREFETCH="$warm" "${extra[@]}" \
    /usr/bin/bash "$root/deploy/run-native-flash-trial.sh" "$([[ "$mode" == single ]] && echo single || echo server)" "$model" "$context"
ready=0
for _ in $(seq 1 100); do
    sleep 3
    if health_ok http://127.0.0.1:8194/health; then ready=1; break; fi
    systemctl --user is-active --quiet qwen-native-flash-server32 || { echo 'server unit exited' >&2; exit 1; }
done
(( ready )) || { echo 'server did not report a healthy status within 300 s' >&2; exit 1; }
echo "server up: $(state)"
# Print the knobs that actually reached the unit (systemd-run --setenv), so a
# configuration label can be checked against reality.
server_pid=$(systemctl --user show qwen-native-flash-server32 -p MainPID --value)
echo "server pid $server_pid environment: $(tr '\0' '\n' < "/proc/$server_pid/environ" 2>/dev/null | grep -E '^QK_' | tr '\n' ' ')"
if [[ "$warm" == 1 ]]; then
    # Only this invocation's journal counts: an earlier instance's line
    # would otherwise end the wait before the tables are resident.
    warmed=0
    for _ in $(seq 1 80); do
        sleep 5
        if journalctl --user -u qwen-native-flash-server32 --no-pager --since "@$server_started" | grep -q 'page cache:'; then warmed=1; break; fi
    done
    if (( warmed )); then
        journalctl --user -u qwen-native-flash-server32 --no-pager --since "@$server_started" | grep 'page cache:' | tail -1 | cut -c1-200
        echo "warmed: $(state)"
    else
        echo "warning: warming did not report completion within 400 s; the tables may still be paging in" >&2
    fi
fi

start_unit qwen-native-flash-router -p NoNewPrivileges=yes -p UMask=0077 -p LimitCORE=0 -p TasksMax=64 \
    -p MemoryHigh=768M -p MemoryMax=1G -p Requires=qwen-native-flash-server32.service -p After=qwen-native-flash-server32.service \
    /usr/bin/python3 -u "$router" --host 0.0.0.0 --allow-network 192.168.0.0/24 --port 8091 \
    --prefill-url http://127.0.0.1:8194 --decode-url http://127.0.0.1:8194 --decode-slots 1 \
    --threshold 2147483647 --max-prefill-tokens 2147483647
ready=0
for _ in $(seq 1 10); do
    sleep 2
    if health_ok http://127.0.0.1:8091/health; then ready=1; break; fi
done
(( ready )) || { echo 'router did not report a healthy status within 20 s' >&2; exit 1; }
echo "router up on 8091"
echo "done: $(state)"
