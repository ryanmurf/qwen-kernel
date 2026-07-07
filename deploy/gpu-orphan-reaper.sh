#!/usr/bin/env bash
# gpu-orphan-reaper — kill orphaned LOCAL dev-server processes that hold the GPU.
#
# Why: on 2026-07-07 a stale `server/target/release/server` (the standalone dev
# binary, run for local testing and never cleaned up) kept running for ~13h
# holding /dev/dri/renderD128 and ~13.6 GB VRAM, starving the in-cluster
# qk-server pod so it could not load its model. This reaper prevents a repeat.
#
# It is deliberately conservative — it only kills a process when ALL hold:
#   * argv[0] path matches */target/release/server   (the LOCAL dev binary;
#     the container runs /app/server, which is never matched)
#   * it holds /dev/dri/renderD128 (an active GPU consumer)
#   * it is orphaned: PPID == 1 (no shell/harness parent watching it)
#   * it is older than AGE_MIN minutes (default 15) so a fresh, deliberate
#     local run is never reaped
# It NEVER touches the container process, gnome-shell, or a parented dev run.
#
# Run via the companion systemd timer (deploy/gpu-orphan-reaper.{service,timer})
# or ad hoc: `sudo AGE_MIN=15 ./deploy/gpu-orphan-reaper.sh`
set -u
RENDER=/dev/dri/renderD128
AGE_MIN=${AGE_MIN:-15}
LOG=${LOG:-/var/log/gpu-orphan-reaper.log}
DRY=${DRY_RUN:-0}
log(){ echo "$(date -Is) $*" | tee -a "$LOG" 2>/dev/null || echo "$(date -Is) $*"; }

# PIDs currently holding the render node (fuser needs root).
holders=$(fuser "$RENDER" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true)
[ -z "$holders" ] && exit 0

for pid in $holders; do
    exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || continue
    # Only the LOCAL dev binary; the container's /app/server is excluded by path.
    case "$exe" in */target/release/server) ;; *) continue ;; esac
    ppid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null) || continue
    [ "$ppid" = "1" ] || { log "skip pid=$pid (parented ppid=$ppid, likely an active run)"; continue; }
    # age in minutes
    secs=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' '); [ -z "$secs" ] && continue
    (( secs < AGE_MIN*60 )) && { log "skip pid=$pid (age ${secs}s < ${AGE_MIN}m, fresh run)"; continue; }
    if [ "$DRY" = "1" ]; then
        log "WOULD REAP pid=$pid exe=$exe age=${secs}s (orphaned local server holding $RENDER)"
    else
        log "REAPING pid=$pid exe=$exe age=${secs}s (orphaned local server holding $RENDER)"
        kill -TERM "$pid" 2>/dev/null; sleep 5
        kill -0 "$pid" 2>/dev/null && { kill -KILL "$pid" 2>/dev/null; log "SIGKILL pid=$pid"; }
    fi
done
