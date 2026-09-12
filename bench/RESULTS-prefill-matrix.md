# Max: same-token prefill matrix

Native baseline completed on 2026-09-11 (03:24:48 to approximately 03:55 UTC
on September 12). This is the full Qwen3.8-Flash-Next-Uncensored-Q5_K_M model
on the Strix Halo only, 32768 context capacity, one sequence, F32 native
math/KV, 512-token prefill chunks, no MTP, no cooperative-matrix tier, no
full PLE warming; row prefetch and the verified decode fusions are enabled.
Kernel revision: `0649101`. No native binary or shader changed during the run.

Median of three requests per cell, 128 requested/count-verified output tokens:

| Prompt tokens | Prefill + one token (s) | First token (s) | Decode (tokens/s) | Decode min..max |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.209 | 1.205 | 32.76 | 32.50..32.82 |
| 512 | 3.144 | 3.145 | 30.56 | 30.54..30.60 |
| 2048 | 13.530 | 13.591 | 23.32 | 23.13..23.43 |
| 8192 | 69.956 | 70.076 | 12.62 | 12.33..12.74 |
| 16384 | 194.483 | 193.133 | 8.05 | 7.33..8.62 |

All 15 decode requests produced the same valid counting prefix, all 128
tokens, with identical token/text hashes. Every one-token probe returned
token 16 (`1`). This sanity check is not a general quality evaluation.
The 16K decode spread is material; do not present its median as an exact rate.

These are client-observed wall times. The one-token probe includes request
handling and first-token work; it is not pure GPU prefill. Native
`prompt_ms` is zero/unavailable, and its reported predicted time includes
prefill, so the table uses measured streaming timestamps and token counts.
Each decode request follows a separate same-prompt one-token probe.
`cache_prompt=false` throughout; native request logs report `reuse=0`.
No OS-wide cache flushing. First at size is not cold: sizes share prefixes,
the model was already running, and file-cache working sets warm over time.
The LAN router logged no requests from the start of the run until root
paused it at about 03:29:51 UTC; it remained paused for the rest of the run.
Host memory stayed near 26 GiB available, server cgroup swap was zero,
and the XTX stayed idle (enumeration handles only, no model weights).

Raw measurements: [results-halo-prefill-matrix-native-0649101.jsonl](results-halo-prefill-matrix-native-0649101.jsonl).
Harness: `f2463a1`; CPU regression tests expanded in `73fdaba`.
Fixture SHA256: `a37dfebb993cab5cf754a0d63a31875df93d106e61f6351273ad91d427942fc7`.
Run ID: `a9032062-9409-4b27-af9a-a80883ae0505`.
The deterministic default `prepare` workload reproduces this fixture with
the same tokenizer. Full local fixture and metadata are retained under
`/home/ryan/qk-prefill-matrix-eyUgzZ`.

Served native artifact SHA256:

- `libqk.so`: `0782c0ff527a78c02ca52d1872a755835ec02a73b6229fcfd5f485d76cb2d3a8`
- Rust server: `aa497bdb0e6b8a70c8d9abe34f6956c91c04cc1ecd5bace53d9e493ce273ef9c`

## Comparison status

llama.cpp comparison is pending. The installed local fork is
`2dff8596dcb7bdf765d24d44e1155d51f04c82b7`, with a local grammar fix;
the launcher in `deploy/start-llama-prefill-halo.sh` explicitly selects Halo.
F16/flash-attention, F32/non-flash, and MTP configurations must be labeled
separately and pass the output check before a validated speed is reported.
The identity of the user's “recipe” backend is still awaiting clarification.
No speedup against either reference is claimed yet.

The drop in native decode speed with longer prompts makes attention a
candidate for the next experiment. Split-K source preparation is opt-in and
unbuilt; its attribution and speedup are not yet measured.
