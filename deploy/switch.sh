#!/usr/bin/env bash
# Backend switching for the gemma-server Service (selects `gpu-llm: server`).
# Card roles since 2026-07-13 (Ryan's call: "audio on the XT, 80B on
# XTX+midnight, don't care about the 35B"):
#   - 7900 XT (03:00.0): AUDIO ONLY (kokoro TTS + whisper STT, coord-2's
#     domain). No LLM backend may load here. DPM pin stays.
#   - 7900 XTX (1a:00.0): the LLM card — 80B split head (default) or the
#     35B (qk-server, on-demand via 35b-up), ONE at a time; both pinned
#     1a:00.0. gemma-server (llama.cpp fallback) is UNPINNED and would
#     land on the XT — do not scale it up while audio owns that card.
#   ./switch.sh split80 -> qk-server-split-80b (80B: XT head [0,12) +
#                          midnight :18200 pipe-worker, QK_PCACHE=6 BOTH sides)
#   ./switch.sh split   -> qk-server-split (35B: XT head + midnight :18100)
#   ./switch.sh gemma   -> llama.cpp fallback
#   ./switch.sh 35b-up | 35b-down -> scale the always-on 35B (XTX)
#   ./switch.sh status  -> show all
# NB: cross-deployment overlap on ONE card silently degrades the incoming
# pod to GTT (~9 tok/s at /health 200) — after switching, verify the card's
# vram_used matches the expected resident set before trusting benchmarks.
set -euo pipefail
NS=gemma
SWITCHED="gemma-server qk-server-split qk-server-split-80b qk-server"
up() {
  for d in $SWITCHED; do
    [ "$d" = "$1" ] || kubectl scale deploy "$d" -n $NS --replicas=0 2>/dev/null || true
  done
  kubectl scale deploy "$1" -n $NS --replicas=1
  kubectl rollout status deploy "$1" -n $NS --timeout=180s
}
case "${1:-status}" in
  split)    up qk-server-split ;;
  split80)  up qk-server-split-80b ;;
  gemma)    up gemma-server ;;
  qk|35b-up)
            kubectl scale deploy qk-server -n $NS --replicas=1
            kubectl rollout status deploy qk-server -n $NS --timeout=240s ;;
  35b-down) kubectl scale deploy qk-server -n $NS --replicas=0 ;;
  status)
    kubectl get deploy $SWITCHED qk-server -n $NS 2>/dev/null || kubectl get deploy -n $NS
    echo "gemma-server (switched):"; kubectl get endpoints gemma-server -n $NS
    echo "qk-35b (always-on 35B):";  kubectl get endpoints qk-35b -n $NS ;;
  *) echo "usage: $0 {split80|split|gemma|35b-up|35b-down|status}"; exit 1 ;;
esac
