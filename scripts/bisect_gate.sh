#!/bin/bash
# git bisect run helper for the Metal port.
#
#   git bisect start <bad> <good>
#   git bisect run scripts/bisect_gate.sh [gate] [arg]
#
# Gates:
#   parity            (default) build + serve-test ids1 vs llama ref — token-exact or bad
#   parity-long       ids4 through QK_MAXB=512 (grouped-MoE 512-chunks)
#   decode <tok/s>    serving decode >= threshold (default 85). QK_MLOCK=1 is set to
#                     dodge the mmap-rewire pathology (PORT.md B3); still thermally
#                     fragile — sanity-check a borderline verdict with an interleaved
#                     A/B/A run before trusting it.
#   prefill <tok/s>   prefillbench N=512 batched >= threshold (default 250), QK_MAXB=512
#
# Exit: 0 good, 1 bad, 125 skip (build failure — bisect skips the commit).
set -u
cd "$(dirname "$0")/.."
export QK_SHADER_DIR="$PWD/shaders/metal"
export QK_GGUF="${QK_GGUF:-$PWD/../models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}"
[ -f "$QK_GGUF" ] || { echo "bisect_gate: set QK_GGUF" >&2; exit 125; }

cmake --build build -j 12 >/dev/null 2>&1 || exit 125
QK=build/qk

gen() {  # gen <ids> <n> [extra env...]
    local ids=$1 n=$2; shift 2
    env "$@" caffeinate -dims $QK serve-test "$ids" "$n" 2>/dev/null \
        | grep GEN | head -1 | sed 's/GEN: //; s/ /,/g'
}

case "${1:-parity}" in
parity)
    n=$(tr ',' '\n' < tests/ref1.txt | wc -l | tr -d ' ')
    [ "$(gen tests/ids1.txt "$n")" = "$(cat tests/ref1.txt)" ] || exit 1
    ;;
parity-long)
    out=$(env QK_MAXB=512 caffeinate -dims $QK serve-test tests/ids4.txt 64 1 4096 2>/dev/null \
          | grep GEN | head -1 | sed 's/GEN: //; s/ /,/g')
    [ "$out" = "$(cat tests/ref4.txt)" ] || exit 1
    ;;
decode)
    tps=$(env QK_MLOCK=1 caffeinate -dims $QK serve-test tests/ids3.txt 128 1 2048 2>/dev/null \
          | grep -o '[0-9.]* tok/s' | awk '{print $1}')
    awk -v t="${tps:-0}" -v th="${2:-85}" 'BEGIN{exit !(t+0 >= th)}' || exit 1
    ;;
prefill)
    b=$(env QK_MAXB=512 QK_PB_ONLY=512 QK_PB_NOSERIAL=1 caffeinate -dims $QK prefillbench 2>/dev/null \
        | grep "  512 " | awk '{print $5}')
    awk -v t="${b:-0}" -v th="${2:-250}" 'BEGIN{exit !(t+0 >= th)}' || exit 1
    ;;
*)
    echo "bisect_gate: unknown gate '$1'" >&2; exit 125
    ;;
esac
exit 0
