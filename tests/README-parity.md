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
