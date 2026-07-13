#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-quick}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${2:-/tmp/qk-perf-ledger-${STAMP}.log}"
MODEL_DEFAULT="/Users/ryan/IdeaProjects/nes-experimental/qk-metal/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
export QK_GGUF="${QK_GGUF:-$MODEL_DEFAULT}"
export QK_MLOCK="${QK_MLOCK:-1}"

case "$PROFILE" in
    probe|quick|full|trace) ;;
    *)
        echo "usage: $0 [probe|quick|full|trace] [output.log]" >&2
        exit 2
        ;;
esac

if [[ ! -x "$ROOT/build-perf/qk" ]]; then
    echo "missing $ROOT/build-perf/qk; build only with: cmake --build build-perf -j" >&2
    exit 1
fi
if [[ ! -f "$QK_GGUF" ]]; then
    echo "missing QK_GGUF=$QK_GGUF" >&2
    exit 1
fi

WORKER_STATUS="$(tr -d '[:space:]' </tmp/qk80-worker-status.txt 2>/dev/null || true)"
if [[ "$WORKER_STATUS" != "DOWN" && "${QK_PERF_ALLOW_BUSY:-0}" != "1" ]]; then
    echo "refusing a contaminated GPU run: /tmp/qk80-worker-status.txt is '${WORKER_STATUS:-missing}'" >&2
    echo "stop qk80-worker and record DOWN, or explicitly set QK_PERF_ALLOW_BUSY=1" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
: >"$OUT"

log() {
    printf '%s\n' "$*" | tee -a "$OUT"
}

run() {
    local label="$1"
    shift
    {
        printf '\n===== %s =====\n' "$label"
        printf 'command:'
        printf ' %q' "$@"
        printf '\n'
    } | tee -a "$OUT"
    "$@" 2>&1 | tee -a "$OUT"
}

cd "$ROOT"
log "qk performance ledger"
log "utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "profile: $PROFILE"
log "repo: $ROOT"
log "git: $(git rev-parse HEAD)"
log "branch: $(git branch --show-current)"
log "binary_sha256: $(shasum -a 256 build-perf/qk | awk '{print $1}')"
log "worker_status_file: ${WORKER_STATUS:-missing}"
log "worker_pane: $(tmux list-panes -t qk80-worker -F '#{pane_current_command} dead=#{pane_dead}' 2>/dev/null || echo missing)"
log "os: $(sw_vers -productVersion) ($(uname -m))"
log "model: $QK_GGUF"
log "model_bytes: $(stat -f %z "$QK_GGUF")"
log "git_status_begin"
git status --short | tee -a "$OUT"
log "git_status_end"
log "qk_env_begin"
env | LC_ALL=C sort | awk -F= '/^QK_/ {print}' | tee -a "$OUT"
log "qk_env_end"

run "metal counter capabilities" ./build-perf/qk counters
run "cool-state f16 bandwidth probe" \
    env -u QK_COUNTERS -u QK_COUNTER_RAW ./build-perf/qk f16 8192 8192 100

if [[ "$PROFILE" == "probe" ]]; then
    log "ledger: $OUT"
    exit 0
fi

if [[ "$PROFILE" == "trace" ]]; then
    TRACE_MOE_GROUPED="${QK_TRACE_MOE_GROUPED:-5}"
    run "pp512 grouped-${TRACE_MOE_GROUPED} stage trace (perturbed, timestamp-only)" \
        env QK_COUNTERS=1 QK_COUNTER_RAW=0 QK_MAXB=512 \
        QK_MOE_GROUPED="$TRACE_MOE_GROUPED" QK_MOE_GROUP_N=192 \
        QK_PB_ONLY=512 QK_PB_NOSERIAL=1 \
        ./build-perf/qk prefillbench 2048
    log "ledger: $OUT"
    exit 0
fi

run "single-stream short-context decode" \
    env -u QK_COUNTERS -u QK_COUNTER_RAW QK_SPEC=0 \
    ./build-perf/qk token tests/ids3.txt 40 512
run "pp512 exact-class v4" \
    env -u QK_COUNTERS -u QK_COUNTER_RAW QK_MAXB=512 QK_MOE_GROUPED=4 \
    QK_MOE_GROUP_N=192 QK_PB_ONLY=512 QK_PB_NOSERIAL=1 \
    ./build-perf/qk prefillbench 2048

if [[ "$PROFILE" == "full" ]]; then
    run "pp512 packed-f16 record tier" \
        env -u QK_COUNTERS -u QK_COUNTER_RAW QK_MAXB=512 QK_MOE_GROUPED=5 \
        QK_PB_ONLY=512 QK_PB_NOSERIAL=1 ./build-perf/qk prefillbench 2048
    run "eight-slot aggregate and determinism" \
        env -u QK_COUNTERS -u QK_COUNTER_RAW QK_SPEC=0 \
        ./build-perf/qk serve-test tests/ids3.txt 200 8 512
    run "one-stream context-8k decode" \
        env -u QK_COUNTERS -u QK_COUNTER_RAW QK_SPEC=0 QK_FA_SPLIT=1 \
        ./build-perf/qk fadecode 1 8000 64 8192
    run "post-run f16 bandwidth probe" \
        env -u QK_COUNTERS -u QK_COUNTER_RAW ./build-perf/qk f16 8192 8192 100
fi

log "ledger: $OUT"
