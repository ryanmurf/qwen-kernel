#!/bin/bash
# Full correctness matrix (tests/README-parity.md) for the prefill campaign.
# usage: scripts/gates_fable.sh [extra env assignments...]
#   e.g. scripts/gates_fable.sh QK_DN_STEP_COLS=1
# Runs each gate with the tuned flag set plus the given candidate flags, and
# for ids4 also runs the contemporaneous rollback (candidate flags absent).
set -u
cd "$(dirname "$0")/.."
export QK_GGUF="${QK_GGUF:-/home/ryan/intellij/ggerganov/llama.cpp/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}"
export QK_SHADER_DIR="$PWD/build/shaders"
export QK_DEVICE_PCI="${QK_DEVICE_PCI:-1a:00.0}"
QK=build/qk

TUNED=(QK_MAXB=1024 QK_ATTN_CHUNK=64 QK_ATTN_LIVE_DISPATCH=1 QK_ATTN_GQA_AUTO=1
       QK_MOE_SELECT_FAST=1 QK_MOE_ROUTE_FUSED=1 QK_DN_STEP_GATE_FUSED=1
       QK_MOE_DOWN_128=1 QK_MOE_SELECT_HIER=1 QK_MOE_SHARED_GU_64=1
       QK_MOE_SHARED_DOWN_32=1 QK_MOE_GROUP_PREFILL=1 QK_MOE_GU_ROWTILE=1
       QK_PREFILL_COOPMAT=1 QK_MOE_PREFILL_COOPMAT=1)
CAND=("$@")

gen() {  # gen <ids> <n> <tmax> [env...]
    local ids=$1 n=$2 tmax=$3; shift 3
    env "$@" $QK serve-test "$ids" "$n" 1 "$tmax" 2>/dev/null \
        | grep GEN | head -1 | sed 's/GEN: //; s/ *$//; s/ /,/g'
}

echo "== prefillcmp (candidate) =="
env "${TUNED[@]}" "${CAND[@]}" $QK prefillcmp 2>/dev/null | tail -2

echo "== dncmp =="
env "${TUNED[@]}" "${CAND[@]}" $QK dncmp 2>/dev/null | tail -1

for i in 1 2 3; do
    ref=$(cat tests/ref$i.txt)
    n=$(tr ',' '\n' < tests/ref$i.txt | grep -c .)
    out=$(gen tests/ids$i.txt "$n" 2048 "${TUNED[@]}" "${CAND[@]}")
    [ "$out" = "$ref" ] && echo "ids$i: PASS ($n/$n)" || { echo "ids$i: FAIL"; echo "  got: $out"; }
done

for MB in 1024 512; do
    cand=$(gen tests/ids4.txt 64 4096 "${TUNED[@]}" "${CAND[@]}" QK_MAXB=$MB)
    roll=$(gen tests/ids4.txt 64 4096 "${TUNED[@]}" QK_MAXB=$MB)
    [ "$cand" = "$roll" ] && echo "ids4 maxB=$MB: candidate==rollback PASS" \
                          || { echo "ids4 maxB=$MB: candidate!=rollback FAIL"; echo "  cand: $cand"; echo "  roll: $roll"; }
    [ "$cand" = "$(cat tests/ref4.txt)" ] && echo "  (ref4: exact)" || echo "  (ref4: diverges — pre-existing at idx 32 expected)"
done

echo "== 8 slots x 200 =="
env "${TUNED[@]}" "${CAND[@]}" $QK serve-test tests/ids3.txt 200 8 2048 2>/dev/null | grep -iE "match|identical|slot|tok/s" | tail -4

echo "== N=128 handoff (prefilldecode) =="
env "${TUNED[@]}" "${CAND[@]}" $QK prefilldecode 128 24 2048 2>/dev/null | tail -4
