#!/usr/bin/env bash
# Switch the single GPU between the two inference backends. Both are Deployments
# in namespace `gemma`; the `gemma-server` Service selects `gpu-llm: server`, so
# whichever pod is scaled up (and Ready) receives litellm/client traffic — no
# frontend change needed.
#   ./switch.sh qk      -> scale gemma-server to 0, qk-server to 1
#   ./switch.sh gemma   -> scale qk-server to 0, gemma-server to 1
#   ./switch.sh status  -> show both
set -euo pipefail
NS=gemma
case "${1:-status}" in
  qk)
    kubectl scale deploy gemma-server -n $NS --replicas=0
    kubectl scale deploy qk-server    -n $NS --replicas=1
    kubectl rollout status deploy qk-server -n $NS --timeout=180s ;;
  gemma)
    kubectl scale deploy qk-server    -n $NS --replicas=0
    kubectl scale deploy gemma-server -n $NS --replicas=1
    kubectl rollout status deploy gemma-server -n $NS --timeout=180s ;;
  status)
    kubectl get deploy gemma-server qk-server -n $NS
    echo "Service endpoints:"; kubectl get endpoints gemma-server -n $NS ;;
  *) echo "usage: $0 {qk|gemma|status}"; exit 1 ;;
esac
