/* qk.h — C ABI for the qwen-kernel Vulkan inference engine.
 *
 * Threading contract: ALL functions must be called from a single thread
 * (the "engine thread"). The engine owns a Vulkan device; no calls are
 * thread-safe. qk_step_chunk blocks that thread for one GPU chunk
 * (roughly cfg.chunk * 6..45 ms depending on active batch size).
 */
#ifndef QK_H
#define QK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qk_engine qk_engine;

typedef struct qk_config {
    uint32_t n_slots; /* max concurrent sequences, 1..16 */
    uint32_t n_ctx;   /* per-slot capacity (prompt + generated), 64..32768 */
    uint32_t chunk;   /* GPU steps per host sync, 1..32 */
} qk_config;

/* Load model, compile pipelines, allocate per-slot state.
 * Returns NULL on failure with a message in err (always NUL-terminated). */
qk_engine *qk_open(const char *gguf_path, const qk_config *cfg,
                   char *err, size_t err_len);
void qk_close(qk_engine *e);

uint32_t qk_n_vocab(const qk_engine *e);
uint32_t qk_n_ctx(const qk_engine *e);
uint32_t qk_n_slots(const qk_engine *e);
uint32_t qk_chunk(const qk_engine *e);
uint32_t qk_eos_token(const qk_engine *e);
uint32_t qk_bos_token(const qk_engine *e);

/* Begin a sequence in a free slot. The prompt is copied.
 * Requires: slot < n_slots and free; 1 <= n_prompt; n_prompt + max_gen <= n_ctx;
 * all prompt ids < n_vocab. The slot becomes active; prompt processing
 * ("prefill") advances inside qk_step_chunk, one token per step, emitting
 * nothing until the prompt is consumed.
 * Returns 0 on success, negative on error (slot busy / bounds). */
/* snap_prefix: conversation-history boundary (token count before the generation
 * scaffold) at which to cache the KV state for cross-turn reuse; 0 caches the
 * full prefill instead. */
int qk_slot_start(qk_engine *e, uint32_t slot,
                  const uint32_t *prompt, uint32_t n_prompt, uint32_t max_gen,
                  uint32_t snap_prefix);

/* Free a slot immediately (client disconnect, stop string hit).
 * Safe on an already-free slot. */
void qk_slot_cancel(qk_engine *e, uint32_t slot);

/* Advance every active slot by up to `chunk` GPU steps.
 *
 * out_tokens   caller buffer of n_slots*chunk u32; slot s appends its newly
 *              sampled tokens at out_tokens[s*chunk + 0..out_counts[s]).
 * out_counts   caller buffer of n_slots u32 (tokens emitted this call; 0
 *              while a slot is still prefilling).
 * out_finished bitmask (bit s): slot s finished this call — either EOS was
 *              sampled (EOS itself is NOT emitted) or max_gen was reached.
 *              Finished slots are freed on return.
 *
 * Returns the number of slots that were active at entry (0 = nothing to do;
 * caller should wait for work before calling again), negative on error. */
int qk_step_chunk(qk_engine *e, uint32_t *out_tokens, uint32_t *out_counts,
                  uint32_t *out_finished);

/* ---- Pipeline split (QK_LAYERS=a:b) ----------------------------------------
 * With QK_LAYERS=a:b in the environment, qk_open loads ONLY transformer layers
 * [a,b): the first stage (a==0) additionally owns the token embedding; the
 * last stage (b==n_layer) owns the final norm + lm head + argmax. Weights,
 * KV cache and recurrent state outside the range are never allocated, so N
 * stages on N devices together hold one model.
 *
 * A split engine is driven ONLY through qk_stage_run below — qk_slot_start /
 * qk_step_chunk return an error on it. The caller (one driver per sequence)
 * carries the ~8 KB/token hidden row between stages. An unsplit engine also
 * accepts qk_stage_run (toks in, ids out), which is the same forward pass. */

uint32_t qk_layer_first(const qk_engine *e); /* a (0 when unsplit)          */
uint32_t qk_layer_end(const qk_engine *e);   /* b (n_layer when unsplit)    */
uint32_t qk_n_layer(const qk_engine *e);     /* total model layers          */
uint32_t qk_n_embd(const qk_engine *e);      /* hidden row width (floats)   */

/* Run n positions [base, base+n) of `slot` through this stage's layers,
 * chunking internally. base==0 resets the slot's state first (fresh sequence);
 * base>0 continues it (caller guarantees positions [0,base) were already run).
 *
 * First stage:     toks      = n token ids            (hidden_in must be NULL)
 * Later stages:    hidden_in = n * n_embd floats from the previous stage
 * Non-last stage:  hidden_out = n * n_embd floats out (ids_out ignored)
 * Last stage:      ids_out    = n u32 out — ids_out[i] is the greedy argmax
 *                  AFTER position base+i, so ids_out[n-1] is the next token.
 *
 * Returns 0 on success, negative on bad args. Blocks the engine thread. */
int qk_stage_run(qk_engine *e, uint32_t slot, const uint32_t *toks,
                 const float *hidden_in, uint32_t n, uint32_t base,
                 float *hidden_out, uint32_t *ids_out);

/* After a last-stage qk_stage_run: copy out the top-k (ids, logits) of the
 * FINAL position's logit row, descending (1 <= k <= 256). This is the
 * sampling hook — the driver samples from these candidates and feeds its
 * pick as the next position, so the engine itself stays sampler-free and
 * the greedy path stays bit-identical.
 * Returns 0 on success, negative on bad args / non-last stage / no run yet. */
int qk_stage_topk(qk_engine *e, uint32_t k, uint32_t *ids, float *vals);

/* Driver-managed state snapshots (split serving's cross-turn reuse): copy a
 * slot's full recurrent state (KV + DeltaNet + conv, this stage's layers) to
 * or from host snapshot entry `idx` (0 <= idx < qk_state_n; entry count is
 * the QK_PCACHE knob, default 3). The caller owns the index space and the
 * mapping to token prefixes. ONLY meaningful on split engines — on an
 * unsplit engine the internal prefix cache uses the same entries and will
 * clobber them. Returns 0 on success, negative on bad args. */
uint32_t qk_state_n(const qk_engine *e);
/* n_tok: the snapshot live token count. Attention KV is copied only up to
 * that many positions per kv-head (recurrent state always copies whole).
 * 0 = full stripes. Pass the SAME n_tok on save and load. */
int qk_state_save(qk_engine *e, uint32_t slot, uint32_t idx, uint32_t n_tok);
int qk_state_load(qk_engine *e, uint32_t slot, uint32_t idx, uint32_t n_tok);

#ifdef __cplusplus
}
#endif

#endif /* QK_H */
