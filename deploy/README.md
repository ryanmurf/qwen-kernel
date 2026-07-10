# Deploying qk-server alongside gemma-server (switchable single GPU)

Both inference backends live in the `gemma` namespace as Deployments. The one
GPU (RX 7900 XT, ~20 GB) fits only one ~16 GB model at a time, so exactly one is
scaled up. The `gemma-server` Service selects the shared label `gpu-llm: server`
(carried by both backends' pods), so litellm and all clients keep hitting the
same endpoint — whichever pod is Ready receives traffic.

- `gemma-server` — llama.cpp Vulkan, `localhost:32000/llama-server:vulkan-anthropic`
  (master `25eec6f32`, built 2026-07-08, has the Anthropic `/v1/messages`
  endpoint; the older `:vulkan` tag (b8671, OpenAI API only) is kept in the
  registry for rollback)
- `qk-server`    — this repo's engine, `localhost:32000/qk-server:vulkan`
  (safe-Rust HTTP server over `libqk.so`; privileged + `/dev/dri`, model from
  the same `/models` hostPath)

## Switch

```bash
./deploy/switch.sh qk      # gemma-server -> 0, qk-server -> 1
./deploy/switch.sh gemma   # qk-server -> 0, gemma-server -> 1
./deploy/switch.sh status
```

Anthropic clients (the `claude-qwen` wrapper → Claude CLI) hit the Service
directly at `POST /v1/messages`. Both backends speak the Anthropic Messages
API natively (qk-server since 2026-07-06; gemma-server since the
`:vulkan-anthropic` image), so the switch is transparent to the CLI as well.
litellm is therefore unused and scaled to 0 — revive with
`kubectl scale deploy/litellm -n gemma --replicas=1` if something needs its
OpenAI-proxy path again.

CLI-path performance comparison of the two backends: see `bench/README.md`
(2026-07-08: qk-server ~4–6× faster across single-shot, tool-call, and
multi-turn scenarios).

## Rebuild the image after code changes

```bash
./deploy/build-image.sh    # rebuild libqk + server, bake image, push
./deploy/switch.sh qk      # pulls :vulkan (imagePullPolicy: Always)
```

## One-time cluster setup (already applied)

```bash
kubectl apply -f deploy/qk-server.yaml
# shared-label routing: relabel gemma's pods + repoint its Service
kubectl patch deploy gemma-server -n gemma --type merge \
  -p '{"spec":{"template":{"metadata":{"labels":{"gpu-llm":"server"}}}}}'
kubectl patch svc gemma-server -n gemma --type json \
  -p '[{"op":"replace","path":"/spec/selector","value":{"gpu-llm":"server"}}]'
```

Note: qk-server runs `--ctx 32768` (measured fit at slots=2, see qk-server.yaml); gemma runs 262144.
