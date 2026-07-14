# Greedy-parity fixtures (Metal port)

`idsN.txt` = token ids of a prompt (comma-separated); `refN.txt` = llama.cpp's
greedy continuation for that prompt, token-for-token. These are the parity
anchors used throughout PORT.md and by `scripts/bisect_gate.sh`.

Provenance: llama.cpp master `f2d1c2f` (2026-07-08), Metal backend,
llama-server `/tokenize` + `/completion` with plain `{"temperature": 0}`
(NOT `"samplers": []` — that strips temperature and made random refs once;
see PORT.md M4). Model: `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`.

| prompt | tokens | ref tokens | note |
|---|---|---|---|
| ids1 | 5 | 100 | short natural prompt |
| ids2 | 14 | 100 | short natural prompt |
| ids3 | 21 | 44 | the M4 parity prompt |
| ids4 | 1040 | 64 | long prompt — exercises grouped-MoE 512-chunks at QK_MAXB=512 |

The engine matches all four byte-for-byte in the default config (and ids4
additionally under QK_MAXB=512 with grouped MoE v3/v4 — PORT.md Phase B).

`ids-spec-echo.txt` is a performance/correctness fixture for prompt-lookup
speculative decoding, not a parity reference. It asks the model to repeat two
copies of a fixed passage. With `QK_NO_EOS=1`, compare:

```sh
QK_NO_EOS=1 ./build-perf/qk serve-test tests/ids-spec-echo.txt 100 1 512
QK_NO_EOS=1 QK_SPEC=1 QK_SPEC_LOG=1 ./build-perf/qk serve-test tests/ids-spec-echo.txt 100 1 512
```

The `GEN:` lines must match exactly. The 2026-07-12 Metal baseline accepted
7.67 tokens/round at K=8 and improved the 100-token run by 1.37x.

## 80B fixtures (Qwen3-Next-80B-A3B-Instruct-IQ4_XS-qk.gguf, added 2026-07-13)

| prompt | tokens | ref tokens | note |
|---|---|---|---|
| ids-80b-a | 39 | 128 | short natural prompt; decode anchor |
| ids-80b-long | 2103 | 64 | long prompt — exercises grouped-MoE batched prefill (seqN >= 192); run serve-test with ctx >= 4096 |

Provenance: refs are the ENGINE's greedy output (commits ed53ed1+cd119d2),
certified against llama.cpp same-file via teacher-forced /completion
(temperature 0, n_probs 3): ids-80b-long is 64/64 llama-exact;
ids-80b-a is 126/128 exact with 2 CERTIFIED near-ties at positions 9
and 112 (llama top-2 gaps 0.054/0.022 logprob, qk picks llama's #2 —
llama's own Q8 activation-quant noise). They are therefore QK ANCHORS:
a byte-identical GEN means the engine is unchanged; a diff means
re-certify against llama.cpp before accepting.

    QK_GGUF=<80B-qk.gguf> ./build/qk serve-test tests/ids-80b-a.txt 128 1 2048
    QK_GGUF=<80B-qk.gguf> QK_MAXB=512 ./build/qk serve-test tests/ids-80b-long.txt 64 1 4096
