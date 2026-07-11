#!/usr/bin/env bash
# Switch the single GPU between the inference backends. All are Deployments
# in namespace `gemma`; the `gemma-server` Service selects `gpu-llm: server`, so
# whichever pod is scaled up (and Ready) receives litellm/client traffic — no
# frontend change needed. Exactly ONE backend may hold the GPU at a time.
#   ./switch.sh qk      -> qk-server (single-box, 35B)
#   ./switch.sh split   -> qk-server-split (35B: tron head + midnight :18100
#                          pipe-worker; worker prereq in docs/split-serving.md)
#   ./switch.sh split80 -> qk-server-split-80b (Qwen3-Next-80B: tron head
#                          [0,12) + midnight :18200 pipe-worker)
#   ./switch.sh gemma   -> llama.cpp fallback
#   ./switch.sh status  -> show all
set -euo pipefail
NS=gemma
ALL="gemma-server qk-server qk-server-split qk-server-split-80b"
up() {
  for d in $ALL; do
    [ "$d" = "$1" ] || kubectl scale deploy "$d" -n $NS --replicas=0 2>/dev/null || true
  done
  kubectl scale deploy "$1" -n $NS --replicas=1
  kubectl rollout status deploy "$1" -n $NS --timeout=180s
}
case "${1:-status}" in
  qk)      up qk-server ;;
  split)   up qk-server-split ;;
  split80) up qk-server-split-80b ;;
  gemma)   up gemma-server ;;
  status)
    kubectl get deploy $ALL -n $NS 2>/dev/null || kubectl get deploy -n $NS
    echo "Service endpoints:"; kubectl get endpoints gemma-server -n $NS ;;
  *) echo "usage: $0 {qk|split|gemma|status}"; exit 1 ;;
esac
