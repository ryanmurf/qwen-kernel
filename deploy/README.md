# Deploying qk-server alongside gemma-server (switchable single GPU)

Both inference backends live in the `gemma` namespace as Deployments. The one
GPU (RX 7900 XT, ~20 GB) fits only one ~16 GB model at a time, so exactly one is
scaled up. The `gemma-server` Service selects the shared label `gpu-llm: server`
(carried by both backends' pods), so litellm and all clients keep hitting the
same endpoint — whichever pod is Ready receives traffic.

- `gemma-server` — llama.cpp Vulkan, `localhost:32000/llama-server:vulkan`
- `qk-server`    — this repo's engine, `localhost:32000/qk-server:vulkan`
  (safe-Rust HTTP server over `libqk.so`; privileged + `/dev/dri`, model from
  the same `/models` hostPath)

## Switch

```bash
./deploy/switch.sh qk      # gemma-server -> 0, qk-server -> 1
./deploy/switch.sh gemma   # qk-server -> 0, gemma-server -> 1
./deploy/switch.sh status
```

The switch is transparent to litellm (`http://gemma-server.gemma.svc…:8080/v1`).

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

Note: qk-server runs `--ctx 2048` (engine cap 4096); gemma runs 262144.
