// Read tensor metadata only. Do not initialize Vulkan or read tensor payloads.
#include "gguf.h"
#include <cstdio>
#include <map>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: qk-model-audit MODEL.gguf\n");
        return 2;
    }
    Gguf model;
    if (!model.open(argv[1])) return 1;
    const auto arch = model.kvStr("general.architecture", "unknown");
    std::printf("architecture: %s\n", arch.c_str());
    for (const char* key : {"block_count", "embedding_length", "context_length",
            "expert_count", "expert_used_count", "expert_feed_forward_length",
            "hyper_connection.count", "hyper_connection.low_rank",
            "attention.indexer.top_k", "embedding_length_per_layer_input"}) {
        std::printf("%s: %llu\n", key,
                    (unsigned long long)model.kvInt(arch + "." + key, 0));
    }
    std::map<uint32_t, uint64_t> counts, bytes;
    uint64_t total = 0;
    for (const auto& [name, tensor] : model.tensors()) {
        ++counts[tensor.type];
        bytes[tensor.type] += tensor.nbytes;
        total += tensor.nbytes;
        if (name == "per_layer_token_embd.weight") {
            std::printf("PLE table: %s, %llu x %llu, %.3f GiB (keep disk-backed)\n",
                ggmlTypeName(tensor.type), (unsigned long long)tensor.ne[0],
                (unsigned long long)tensor.ne[1], tensor.nbytes / double(1ull << 30));
        }
    }
    for (const auto& [type, count] : counts) {
        std::printf("type %u (%s): %llu tensors, %.3f GiB\n", type, ggmlTypeName(type),
                    (unsigned long long)count, bytes[type] / double(1ull << 30));
    }
    std::printf("recognized payload: %.3f GiB\n", total / double(1ull << 30));
    if (arch == "qwen4exp") {
        const auto& ratios = model.kvInts(arch + ".attention.compress_ratios");
        uint32_t active = 0;
        for (uint64_t r : ratios) active += r != 0;
        std::printf("nonzero attention compression ratios: %u of %zu (zero bypasses QSA indexer)\n",
                    active, ratios.size());
        std::puts("native qwen4exp serving: NOT IMPLEMENTED; Q5 kernel tests are not a full model port");
    }
    return 0;
}
