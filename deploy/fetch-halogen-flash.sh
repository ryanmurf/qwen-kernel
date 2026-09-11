#!/usr/bin/env bash
# Fetch a pinned reference checkpoint without running its inference engine.
set -euo pipefail
model_dir=${1:?usage: fetch-halogen-flash.sh MODEL_DIRECTORY}
script_dir=$(dirname "$(realpath "$0")")
hf download peonist-ai/halogen-qwen3.8-flash-next \
    qwen38-flash-next-w4b.hgn qwen38-flash-next-w4b.overlay.hgn \
    qwen38-flash-next-vision.hgn \
    tokenizer/chat_template.jinja tokenizer/generation_config.json \
    tokenizer/merges.txt tokenizer/tokenizer.json tokenizer/tokenizer_config.json \
    tokenizer/vocab.json README.md \
    --revision 214a45c7106f515faf3fb72db0cf9a1bf67bfd77 \
    --local-dir "$model_dir" --max-workers 2
cd "$model_dir"
sha256sum --check "$script_dir/halogen-flash.sha256"
