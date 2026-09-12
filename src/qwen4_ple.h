#pragma once
#include "gguf.h"
#include "quants.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>

// PLE hashes follow the qwen4exp definition: uint64 wrapping products, XOR,
// per-head moduli, and an EOS-cut predecessor window. No full table upload.
struct Qwen4PleConfig {
    uint32_t ngram = 0, headsPerNgram = 0, width = 0, eos = 0;
    std::vector<uint64_t> multipliers, offsets, vocabs;

    static Qwen4PleConfig read(const Gguf& model) {
        Qwen4PleConfig p;
        const std::string prefix = "qwen4exp.ple.";
        p.ngram = model.kvInt(prefix + "ngram_size", 0);
        p.headsPerNgram = model.kvInt(prefix + "heads_per_ngram", 0);
        p.width = model.kvInt("qwen4exp.embedding_length_per_layer_input", 0);
        p.eos = model.kvInt(prefix + "eos_token_id", 0);
        p.multipliers = model.kvInts(prefix + "layer_multipliers");
        p.offsets = model.kvInts(prefix + "head_offsets");
        p.vocabs = model.kvInts(prefix + "head_vocab_sizes");
        return p;
    }

    void validate(uint64_t rows) const {
        if (ngram < 2 || ngram > 16 || !headsPerNgram || headsPerNgram > 32 ||
            width < 32 || width > 4096 || width % 32 || multipliers.size() != ngram ||
            offsets.size() != (ngram - 1) * headsPerNgram || offsets.size() != vocabs.size())
            throw std::runtime_error("invalid qwen4exp PLE metadata");
        for (size_t h = 0; h < offsets.size(); ++h)
            if (!vocabs[h] || offsets[h] >= rows || vocabs[h] > rows - offsets[h] ||
                offsets[h] + vocabs[h] > uint64_t(INT32_MAX))
                throw std::runtime_error("PLE head range exceeds the table/I32 row index");
    }

    // history is oldest-first and excludes the current token. In a serving
    // engine it belongs to the sequence, never to a shared/global cache slot.
    std::vector<uint32_t> rows(uint32_t token, const std::vector<uint32_t>& history) const {
        std::vector<uint32_t> result(offsets.size());
        uint64_t mixed = uint64_t(token) * multipliers.at(0);
        bool cut = false;
        for (uint32_t back = 1; back < ngram; ++back) {
            uint32_t previous = back <= history.size() ? history[history.size() - back] : eos;
            cut = cut || previous == eos;
            mixed ^= uint64_t(cut ? eos : previous) * multipliers.at(back);
            for (uint32_t g = 0; g < headsPerNgram; ++g) {
                size_t h = (back - 1) * headsPerNgram + g;
                result[h] = uint32_t(mixed % vocabs.at(h) + offsets.at(h));
            }
        }
        return result;
    }
};

class Qwen4PleLookup {
    const GgufTensor& table_;
    Qwen4PleConfig config_;
    size_t entries_, sets_;
    uint64_t clock_ = 0;
    std::vector<uint64_t> tags_, ages_;
    std::vector<float> cache_;
public:
    uint64_t hits = 0, misses = 0;
    Qwen4PleLookup(const GgufTensor& table, Qwen4PleConfig config, size_t entries = 4096)
        : table_(table), config_(std::move(config)), entries_(entries), sets_(entries / 4) {
        config_.validate(table.ne[1]);
        if (table.type != GGML_Q5_1 || table.nDims != 2 || table.ne[0] != config_.width ||
            table.nbytes != ggmlRowBytes(table.type, table.ne[0]) * table.ne[1] || !table.data ||
            entries < 4 || entries > 65536 || entries % 4)
            throw std::runtime_error("unsupported PLE table or invalid cache size");
        tags_.assign(entries, UINT64_MAX);
        ages_.assign(entries, 0);
        cache_.resize(entries * config_.width);
        configureMapping();
    }
    const Qwen4PleConfig& config() const { return config_; }
    size_t cacheBytes() const { return cache_.size() * sizeof(float) +
                                     (tags_.size() + ages_.size()) * sizeof(uint64_t); }
    // Rows are hash-random across a 35.8 GiB disk-backed table, so every
    // uncached row is an NVMe page fault. Readahead is disabled on the table
    // once (MADV_RANDOM: a fault reads one page, not the readahead window),
    // and prefetch() issues asynchronous MADV_WILLNEED reads for every row a
    // request will touch before the serial gather, so the faults overlap in
    // the NVMe queue instead of serializing. Bounded: one page per row, no
    // table warming. QK_PLE_ROW_PREFETCH=0 disables both for A/B checks.
    bool rowPrefetch_ = true;
    void configureMapping() {
        const char* v = getenv("QK_PLE_ROW_PREFETCH");
        rowPrefetch_ = !v || strcmp(v, "0") != 0;
        if (!rowPrefetch_) return;
        const size_t page = (size_t)sysconf(_SC_PAGESIZE);
        const uintptr_t start = (uintptr_t)table_.data & ~(uintptr_t)(page - 1);
        const size_t bytes = (((uintptr_t)table_.data + table_.nbytes + page - 1) & ~(uintptr_t)(page - 1)) - start;
        if (madvise((void*)start, bytes, MADV_RANDOM) != 0) rowPrefetch_ = false;
    }
    size_t prefetched = 0;
    void prefetch(const std::vector<uint32_t>& rows) {
        if (!rowPrefetch_) return;
        const size_t page = (size_t)sysconf(_SC_PAGESIZE);
        const size_t rowBytes = ggmlRowBytes(GGML_Q5_1, config_.width);
        for (uint32_t row : rows) {
            if (row >= table_.ne[1]) throw std::runtime_error("PLE row out of range");
            const size_t base = (uint64_t(row) * 2654435761u % sets_) * 4;
            bool cached = false;
            for (size_t way = base; way < base + 4; ++way) if (tags_[way] == row) { cached = true; break; }
            if (cached) continue;
            const uintptr_t addr = (uintptr_t)(table_.data + size_t(row) * rowBytes);
            const uintptr_t start = addr & ~(uintptr_t)(page - 1);
            const size_t bytes = ((addr + rowBytes + page - 1) & ~(uintptr_t)(page - 1)) - start;
            madvise((void*)start, bytes, MADV_WILLNEED);
            ++prefetched;
        }
    }
    void gather(const std::vector<uint32_t>& rows, float* out) {
        for (uint32_t row : rows) {
            if (row >= table_.ne[1]) throw std::runtime_error("PLE row out of range");
            size_t base = (uint64_t(row) * 2654435761u % sets_) * 4;
            size_t slot = base;
            bool found = false;
            for (size_t way = base; way < base + 4; ++way) {
                if (tags_[way] == row) { slot = way; found = true; break; }
                if (ages_[way] < ages_[slot]) slot = way;
            }
            float* decoded = cache_.data() + slot * config_.width;
            if (found) ++hits;
            else {
                ++misses;
                auto* block = reinterpret_cast<const block_q5_1*>(table_.data +
                    size_t(row) * ggmlRowBytes(GGML_Q5_1, config_.width));
                dequant_row_q5_1(block, decoded, config_.width);
                tags_[slot] = row;
            }
            ages_[slot] = ++clock_;
            memcpy(out, decoded, config_.width * sizeof(float));
            out += config_.width;
        }
    }
};
