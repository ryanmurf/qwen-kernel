#!/usr/bin/env bash
# ppab.sh N REPS -- back-to-back prefill timing, matching llama-bench methodology.
#
# Deliberately does NOT use prefillbench's per-rep gpu_busy gate: that gate idles
# the GPU between reps, the card drops from ~2990 MHz to ~860 MHz, and the ramp
# back costs a uniform ~9%.  llama-bench runs its repetitions back-to-back, so
# gating per rep compares a cold-clock qk against a hot-clock llama.cpp.
# Instead: verify the card is quiet and empty ONCE before the run, run the reps
# back-to-back, and verify no foreign process appeared by the end.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:-512}"; REPS="${2:-6}"
CARD=/sys/class/drm/card2/device
# Wait for a previous qk invocation to release the card. This is before the
# process starts, so it costs nothing in-run: prefillbench's own untimed warmup
# admission ramps the clocks back up before rep 1.
for _ in $(seq 1 60); do
    busy0=$(cat $CARD/gpu_busy_percent); vram0=$(( $(cat $CARD/mem_info_vram_used) / 1048576 ))
    [ "$busy0" -le 2 ] && [ "$vram0" -le 512 ] && break
    sleep 1
done
if [ "$busy0" -gt 2 ] || [ "$vram0" -gt 512 ]; then
    echo "REFUSING: card2 busy=${busy0}% vram_used=${vram0}MiB (expected <=2% and <512MiB)" >&2
    exit 3
fi
export QK_GGUF="${QK_GGUF:-/home/ryan/intellij/ggerganov/llama.cpp/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}"
export QK_DEVICE_PCI="${QK_DEVICE_PCI:-1a:00.0}"
export QK_SHADER_DIR="${QK_SHADER_DIR:-$ROOT/build/shaders}"
# THE FULL SHIPPING CONFIGURATION. Every kernel below is opt-in and defaults to
# false in the engine, so a partial set silently measures a slower engine and
# reads as a regression. With only the first four of these, main measures
# ~2199/2250/1971 (0.76x) instead of ~2957/3123/2933 (1.02-1.10x). Override any
# of them from the environment to roll a single kernel back.
export QK_PREFILL_COOPMAT="${QK_PREFILL_COOPMAT:-1}"
export QK_MOE_PREFILL_COOPMAT="${QK_MOE_PREFILL_COOPMAT:-1}"
export QK_MOE_GROUP_PREFILL="${QK_MOE_GROUP_PREFILL:-1}"
export QK_MAXB="${QK_MAXB:-1024}"
export QK_ATTN_CHUNK="${QK_ATTN_CHUNK:-64}"
export QK_ATTN_LIVE_DISPATCH="${QK_ATTN_LIVE_DISPATCH:-1}"
export QK_ATTN_GQA_AUTO="${QK_ATTN_GQA_AUTO:-1}"
export QK_MOE_SELECT_FAST="${QK_MOE_SELECT_FAST:-1}"
export QK_MOE_ROUTE_FUSED="${QK_MOE_ROUTE_FUSED:-1}"
export QK_DN_STEP_GATE_FUSED="${QK_DN_STEP_GATE_FUSED:-1}"
export QK_MOE_DOWN_128="${QK_MOE_DOWN_128:-1}"
export QK_MOE_SELECT_HIER="${QK_MOE_SELECT_HIER:-1}"
export QK_MOE_SHARED_GU_64="${QK_MOE_SHARED_GU_64:-1}"
export QK_MOE_SHARED_DOWN_32="${QK_MOE_SHARED_DOWN_32:-1}"
export QK_MOE_GU_ROWTILE="${QK_MOE_GU_ROWTILE:-1}"
# fable's coopmat rewrite — the *_FIRST_LAYER defaults already encode the
# measured precision boundaries, so do not set them here.
export QK_DN_STEP_COLS="${QK_DN_STEP_COLS:-1}"
export QK_ATTN_PREFILL_COOPMAT="${QK_ATTN_PREFILL_COOPMAT:-1}"
export QK_MOE_LOGITS_GEMM="${QK_MOE_LOGITS_GEMM:-1}"
export QK_PREFILL_COOPMAT_SPLIT="${QK_PREFILL_COOPMAT_SPLIT:-1}"
export QK_PB_ONLY="$N" QK_PB_REPS="$REPS"
CTX=$(( N + 128 )); [ "$CTX" -lt 2048 ] && CTX=2048
out=$("$ROOT/build/qk" prefillbench "$CTX" 2>&1 | grep "prefillbench sample")
echo "$out"
vram1=$(( $(cat $CARD/mem_info_vram_used) / 1048576 ))
echo "$out" | rtk proxy awk -v n="$N" -v b0="$busy0" -v v0="$vram0" -v v1="$vram1" '
    { split($0,a,"tok/s="); split(a[2],b," "); v[NR]=b[1]+0 }
    END { asort(v); m = (NR%2) ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2
          printf "  pp%s median=%.1f tok/s  min=%.1f max=%.1f spread=%.2f%%  reps=%d  gpu_busy_before=%s%%  vram %sMiB->%sMiB\n",
                 n, m, v[1], v[NR], 100.0*(v[NR]-v[1])/m, NR, b0, v0, v1 }'
