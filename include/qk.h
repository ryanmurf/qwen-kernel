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
    uint32_t n_ctx;   /* per-slot capacity (prompt + generated), 64..4096 */
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

#ifdef __cplusplus
}
#endif

#endif /* QK_H */
