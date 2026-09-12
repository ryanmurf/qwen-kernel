# Prefill-size and decode comparison

`prefill_matrix.py` measures native qk, llama.cpp, and other servers exposing
the llama.cpp `/tokenize` and `/completion` APIs. It never launches or stops
servers. Run exactly one model server at a time on the Strix Halo; check its
actual device selection, memory headroom, and process environment first.
Do not start the legacy two-GPU units to run this benchmark.

Default prompt sizes are **128, 512, 2048, 8192, and 16384 tokens**, with
128 requested output tokens and three repetitions at each size. A 32768-token
context accommodates this matrix. Larger matrices can be prepared explicitly,
but must fit the tested backend's context and memory budget.

## What is measured

- **Prefill plus one token:** end-to-end wall time for a non-streaming request
  with one requested output token. This includes request handling, prefill,
  first-token work, and response overhead; it is not pure GPU prefill time.
- **Time to first token:** client-observed latency of a separate streaming
  request with the same prompt.
- **Decode tokens/sec:** counted output tokens after the first streamed event,
  divided by elapsed time through the last token event. It is measured at
  each actual prompt size, not extrapolated from a short-prompt benchmark.
- Server-reported timing fields are saved separately. Missing or zero
  `prompt_ms` is recorded as unavailable, never as instantaneous prefill.

The synthetic input consists of varied deterministic records plus a counting
instruction. It is a throughput workload, not a general model-quality test.
All backends receive the exact same token IDs, including framing. Each run
re-tokenizes both source strings and refuses a tokenizer mismatch. Save a
different fixture for a real-code corpus with `prepare --prompt-file PATH`.

Each size/repetition runs its one-token probe before its decode request.
Therefore the decode request follows a read of the same PLE working set.
Both requests set `cache_prompt=false`; no OS cache flushing is performed.
“First at size” does **not** mean a cold model or cold file cache: sizes share
prefixes, and prior requests warm data. Later repetitions are identified
separately. For a separate cold-start campaign, restart each backend with the
same cache procedure and record it in metadata; do not mix it into these rows.

## Run

Prepare the fixture once against an already-running verified native server:

```bash
python3 bench/prefill_matrix.py prepare \
  --url http://127.0.0.1:8194 \
  --model-id Qwen3.8-Flash-Next-Uncensored-Q5_K_M \
  --output /path/to/shared-prefill-fixture.json
```

Then benchmark that server:

```bash
python3 bench/prefill_matrix.py run \
  --url http://127.0.0.1:8194 --backend native-0649101 \
  --model-id Qwen3.8-Flash-Next-Uncensored-Q5_K_M \
  --fixture /path/to/shared-prefill-fixture.json \
  --context 32768 --decode-tokens 128 --repetitions 3 \
  --metadata /path/to/native-build-and-flags.json \
  --confirm-exclusive --output /path/to/native-prefill-results.jsonl
```

After stopping and draining the native model, start the comparison backend
in its verified **single-Halo** configuration and repeat `run` with the same
fixture, context and output count, a distinct backend label, its direct URL,
and a new output filename. The “recipe” backend must be identified before
its result can be labeled accurately. No result is fabricated for it.

Record the exact model shards/quantization, backend revision, launch flags,
device placement, KV types, flash-attention mode, prefill batch sizes, MTP
state and relevant environment in the metadata JSON. Never put credentials
in this file. Use separate rows for llama.cpp with and without MTP or for
different precision tiers; do not silently compare different settings.
Metadata gets its own hash so different configurations are not pooled.

Output and fixture files are created exclusively: existing files are never
overwritten. An interrupted run retains its raw rows and an error record
when possible, but has no valid completion marker. Summaries only accept
completed matrices with exactly one measurement per declared cell:

```bash
python3 bench/prefill_matrix.py summarize \
  /path/to/native-prefill-results.jsonl /path/to/llama-prefill-results.jsonl
```

Summaries are JSON rows with medians and min/max. They preserve workload,
model, context, output count and configuration hashes. Short/EOS-terminated
or uncounted completions, and outputs that fail the counting-prefix check,
are flagged and excluded from fixed-output decode medians. Their raw speeds
remain in the result file, not silently presented as validated performance.
Raw token/text hashes allow output consistency checks across
backends; matching hashes are not a substitute for a quality evaluation.

For a small live smoke test, select a fixture subset with `run --sizes 128
--decode-tokens 16 --repetitions 1`, a separate backend label and a new output
filename. Smoke results must not be pooled with the full matrix.

CPU-only harness tests:

```bash
python3 -m unittest discover -s tests -p test_prefill_matrix.py -v
```
