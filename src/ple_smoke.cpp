#include "qwen4_ple.h"
#include <chrono>
#include <cmath>

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: qk-ple-smoke MODEL.gguf\n"); return 2; }
    try {
        Gguf model;
        if (!model.open(argv[1])) return 1;
        const auto* table = model.find("per_layer_token_embd.weight");
        if (!table) throw std::runtime_error("missing PLE table");
        Qwen4PleLookup lookup(*table, Qwen4PleConfig::read(model));
        const auto& config = lookup.config();
        std::vector<float> first(config.offsets.size() * config.width), second(first.size());
        std::vector<uint32_t> history;
        double checksum = 0;
        auto start = std::chrono::steady_clock::now();
        for (uint32_t token : {config.eos, 198u, 1234u, 5678u, config.eos, 198u}) {
            auto rows = config.rows(token, history);
            lookup.gather(rows, first.data()); lookup.gather(rows, second.data());
            if (first != second) throw std::runtime_error("cached lookup differs");
            for (float x : first) {
                if (!std::isfinite(x)) throw std::runtime_error("non-finite PLE embedding");
                checksum += x;
            }
            std::printf("token=%u rows=", token);
            for (uint32_t r : rows) std::printf(" %u", r);
            std::puts("");
            history.push_back(token);
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        std::printf("PLE sparse lookup: PASS; cache_bytes=%zu hits=%llu misses=%llu checksum=%.9g elapsed_ms=%.3f\n",
            lookup.cacheBytes(), (unsigned long long)lookup.hits, (unsigned long long)lookup.misses, checksum, ms);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
