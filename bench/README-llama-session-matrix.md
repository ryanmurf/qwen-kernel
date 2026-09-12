# Cache-aware llama.cpp session matrix

`llama_session_matrix.py` measures client-observed first-token latency and
streaming decode rate on an already running, exclusive server. It does not
start models, stop services, or change GPU settings. Use a loopback test port;
do not run it alongside production requests or another GPU model.

Prepare a deterministic token fixture using `prefill_matrix.py prepare` (see
[the prefill guide](README-prefill-matrix.md)). The server tokenizer must
match the fixture exactly. Record the binary, model, driver, actual device,
launch arguments and environment in a JSON metadata file. For example:

```sh
python3 -E bench/llama_session_matrix.py run \
  --url http://127.0.0.1:8193 --backend my-pinned-build \
  --fixture /absolute/path/fixture.json \
  --metadata /absolute/path/metadata.json \
  --output /absolute/path/new-results.jsonl \
  --context 65536 --repetitions 3 --decode-tokens 128 \
  --confirm-exclusive
python3 -E bench/llama_session_matrix.py summarize /absolute/path/new-results.jsonl
```

The output must not exist. Unless `--sizes` selects a subset, every fixture
size runs three conditions in order, repeated three times:

| Condition | Input | Requested KV reuse |
| --- | --- | --- |
| `fresh` | Original fixed token IDs | Disabled |
| `repeat` | Identical original token IDs | Enabled |
| `followup` | Original + repeat's generated IDs + a new user turn | Enabled |

Fresh means fresh **KV**, not cold model weights, disk or OS page cache.
Exact repeats are a best-case cache experiment, not an agent workload.
Follow-up inputs may differ between configurations if generated IDs differ;
check prompt hashes before comparing them as identical workloads.

Requests use slot 0, greedy decoding and token-aware SSE parsing. A burst
containing several speculative tokens counts as several tokens, not one
event. Decode rate excludes the first event's token count and time before
that event. Final server timings are retained separately from client wall
times. No log-probability collection or synthetic speculation is requested.

The audit requires complete ordered cells, unchanged run identity, exact
output counts, valid counting prefixes, positive timings, and actual
`timings.cache_n` / `prompt_n` accounting for every input token. The
`tokens_cached` response field is retained sequence length, not a hit count.
A reuse request with `cache_n=0` remains in the results as a cache miss;
recurrent checkpoints can restrict reuse. Partial and failed runs cannot
produce a successful summary.

This is a synthetic counting benchmark. Counting can favor speculation;
speed and a coherent prefix do not establish general quality, coding
performance, tool-call correctness or suitability as the serving default.
Test API compatibility and representative tasks separately. Always retain
failed trials and restore the known-good serving configuration afterward.

CPU-only regression tests:

```sh
python3 -m unittest discover -s tests -p 'test_*matrix.py' -v
```
