/* qk_stub.c — deterministic fake engine implementing qk.h, no GPU.
 *
 * Purpose: lets the Rust server build and run its full test suite without
 * the Vulkan engine. Behavior is exactly specified so tests can assert ids.
 *
 *   - qk_open ignores gguf_path contents (must be non-NULL); vocab=248320,
 *     eos=248046, bos=248044.
 *   - Prefill: a slot emits nothing for the first n_prompt steps.
 *   - Generation: seed = last prompt token; each step emits
 *         next = (prev * 1103515245u + 12345u) % 248000u
 *     (never collides with special ids >= 248044).
 *   - EOS path: if the FIRST prompt token == 7, the slot samples EOS as its
 *     4th generated token (3 tokens emitted, then finished via EOS).
 *   - qk_step_chunk sleeps ~1 ms when any slot is active (emulates GPU).
 *
 * Build:  cc -shared -fPIC -O2 stub/qk_stub.c -o libqk.so
 */
#include "../include/qk.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define QK_VOCAB 248320u
#define QK_EOS   248046u
#define QK_BOS   248044u

typedef struct {
    int      active;
    uint32_t prefill_left;
    uint32_t prev;
    uint32_t gen;
    uint32_t max_gen;
    int      eos_at_4;
} slot_t;

struct qk_engine {
    qk_config cfg;
    slot_t    slots[16];
};

qk_engine *qk_open(const char *gguf_path, const qk_config *cfg,
                   char *err, size_t err_len) {
    if (!gguf_path || !cfg || cfg->n_slots < 1 || cfg->n_slots > 16 ||
        cfg->n_ctx < 64 || cfg->n_ctx > 4096 || cfg->chunk < 1 || cfg->chunk > 32) {
        if (err && err_len) snprintf(err, err_len, "qk_stub: bad config");
        return NULL;
    }
    qk_engine *e = calloc(1, sizeof(*e));
    if (!e) {
        if (err && err_len) snprintf(err, err_len, "qk_stub: oom");
        return NULL;
    }
    e->cfg = *cfg;
    return e;
}

void qk_close(qk_engine *e) { free(e); }

uint32_t qk_n_vocab(const qk_engine *e)  { (void)e; return QK_VOCAB; }
uint32_t qk_n_ctx(const qk_engine *e)    { return e->cfg.n_ctx; }
uint32_t qk_n_slots(const qk_engine *e)  { return e->cfg.n_slots; }
uint32_t qk_chunk(const qk_engine *e)    { return e->cfg.chunk; }
uint32_t qk_eos_token(const qk_engine *e){ (void)e; return QK_EOS; }
uint32_t qk_bos_token(const qk_engine *e){ (void)e; return QK_BOS; }

int qk_slot_start(qk_engine *e, uint32_t slot,
                  const uint32_t *prompt, uint32_t n_prompt, uint32_t max_gen) {
    if (!e || slot >= e->cfg.n_slots) return -1;
    if (e->slots[slot].active) return -2;
    if (!prompt || n_prompt < 1 || n_prompt + max_gen > e->cfg.n_ctx) return -3;
    for (uint32_t i = 0; i < n_prompt; i++)
        if (prompt[i] >= QK_VOCAB) return -4;
    slot_t *s = &e->slots[slot];
    s->active = 1;
    s->prefill_left = n_prompt;
    s->prev = prompt[n_prompt - 1];
    s->gen = 0;
    s->max_gen = max_gen;
    s->eos_at_4 = prompt[0] == 7u;
    return 0;
}

void qk_slot_cancel(qk_engine *e, uint32_t slot) {
    if (e && slot < e->cfg.n_slots) e->slots[slot].active = 0;
}

int qk_step_chunk(qk_engine *e, uint32_t *out_tokens, uint32_t *out_counts,
                  uint32_t *out_finished) {
    if (!e || !out_tokens || !out_counts || !out_finished) return -1;
    memset(out_counts, 0, e->cfg.n_slots * sizeof(uint32_t));
    *out_finished = 0;
    int active_at_entry = 0;
    for (uint32_t s = 0; s < e->cfg.n_slots; s++)
        if (e->slots[s].active) active_at_entry++;
    if (!active_at_entry) return 0;

    struct timespec ts = {0, 1000000}; /* 1 ms */
    nanosleep(&ts, NULL);

    for (uint32_t step = 0; step < e->cfg.chunk; step++) {
        for (uint32_t si = 0; si < e->cfg.n_slots; si++) {
            slot_t *s = &e->slots[si];
            if (!s->active) continue;
            if (s->prefill_left > 0) { s->prefill_left--; continue; }
            uint32_t next = (s->prev * 1103515245u + 12345u) % 248000u;
            if (s->eos_at_4 && s->gen == 3) next = QK_EOS;
            if (next == QK_EOS) {
                s->active = 0;
                *out_finished |= 1u << si;
                continue;
            }
            out_tokens[si * e->cfg.chunk + out_counts[si]++] = next;
            s->prev = next;
            s->gen++;
            if (s->gen >= s->max_gen) {
                s->active = 0;
                *out_finished |= 1u << si;
            }
        }
    }
    return active_at_entry;
}
