#!/usr/bin/env bash
# Rebuild the qk-server artifacts (host toolchain) and bake them into the
# container image in the microk8s registry. Run after any engine/server change,
# then `./switch.sh qk` picks up the new image (imagePullPolicy: Always).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REG=localhost:32000/qk-server:vulkan
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

/usr/bin/cmake --build "$ROOT/build" --target qklib -j
( cd "$ROOT/server" && cargo build --release )

cp "$ROOT/server/target/release/server" "$CTX/server"
cp "$ROOT/build/libqk.so"               "$CTX/libqk.so"
cp -r "$ROOT/build/shaders"             "$CTX/shaders"
cp "$ROOT/deploy/Dockerfile"            "$CTX/Dockerfile"

docker build -t "$REG" "$CTX"
docker push "$REG"
echo "pushed $REG"
