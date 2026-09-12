// Focused diagnostic: replay the unchanged GPU router on captured layer-41
// FFN inputs. Only its ~5 MiB F32 router weights are read, not the whole model.
// Reuse the isolated test's Vulkan scaffolding; no serving binary is changed.
#define main unused_attention_test_main
#include "halo_attn_operator.cpp"
#undef main

static void routerDispatch(Kernel& kernel, const uint32_t* pc, uint32_t groups) {
    vkCmdBindPipeline(kernel.c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.p.p);
    vkCmdBindDescriptorSets(kernel.c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.p.pl,
                           0, 1, &kernel.set, 0, nullptr);
    vkCmdPushConstants(kernel.c.cb, kernel.p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(kernel.c.cb, groups, 1, 1);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: halo-router-replay MODEL SERIAL_FFN_MIXED SPLIT_FFN_MIXED\n");
        return 2;
    }
    Gguf model;
    if (!model.open(argv[1])) return 2;
    const auto* weights = model.find("blk.41.ffn_gate_inp.weight");
    const auto* shared = model.find("blk.41.ffn_gate_inp_shexp.weight");
    if (!weights || weights->type != GGML_F32 || weights->ne[0] != 2560 || weights->ne[1] != 512
            || weights->nbytes != 512 * 2560 * 4 || !shared || shared->type != GGML_F32
            || shared->nbytes != 2560 * 4) return 2;
    unsetenv("QK_DEVICE"); unsetenv("QK_DEVICE_PCI"); setenv("QK_DEVICE_NAME", "STRIX_HALO", 1);
    VkCtx c; initVk(c, argv[0]);
    if (c.props.vendorID != 0x1002 || c.props.deviceID != 0x1586 || pciBdf(c.phys) != "0000:c1:00.0") return 2;
    bool valid = true;
    std::vector<uint32_t> selections[2];
    {
        MappedBuffer w(c, 512 * 2560), s(c, 2560), x(c, 2560), logits(c, 512, true), selected(c, 40, true);
        memcpy(w.ptr, weights->data, weights->nbytes); memcpy(s.ptr, shared->data, shared->nbytes);
        Kernel router(c, "moe_logits.spv", {&w, &x, &logits}, 16);
        Kernel select(c, "moe_select_256.spv", {&logits, &s, &x, &selected}, 16);
        const uint32_t pc[4] = {2560, 640, 512, 10};
        for (int input = 0; input < 2; ++input) {
            FILE* f = fopen(argv[2 + input], "rb");
            if (!f) return 2;
            bool read = fread(x.ptr, 4, 2560, f) == 2560 && fgetc(f) == EOF;
            fclose(f);
            if (!read || !std::all_of(x.ptr, x.ptr + 2560, [](float v) { return std::isfinite(v); })) return 2;
            timed(c, 1, [&] {
                routerDispatch(router, pc, 512);
                memoryBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
                routerDispatch(select, pc, 1);
            });
            if (!std::all_of(logits.ptr, logits.ptr + 512, [](float v) { return std::isfinite(v); })) return 1;
            std::vector<uint32_t> rank(512);
            for (uint32_t i = 0; i < 512; ++i) rank[i] = i;
            std::sort(rank.begin(), rank.end(), [&](uint32_t a, uint32_t b) {
                return logits.ptr[a] != logits.ptr[b] ? logits.ptr[a] > logits.ptr[b] : a < b;
            });
            selections[input].resize(10);
            memcpy(selections[input].data(), selected.ptr, 40);
            for (uint32_t i = 0; i < 10; ++i) valid &= selections[input][i] == rank[i];
            printf("{\"input\":\"%s\",\"actual_gpu_top10\":[", input ? "split" : "serial");
            for (uint32_t i = 0; i < 10; ++i) printf("%s%u", i ? "," : "", selections[input][i]);
            printf("],\"top12_logits\":[");
            for (uint32_t i = 0; i < 12; ++i)
                printf("%s{\"expert\":%u,\"logit\":%.9g}", i ? "," : "", rank[i], logits.ptr[rank[i]]);
            printf("],\"boundary_margin\":%.9g}\n", logits.ptr[rank[9]] - logits.ptr[rank[10]]);
        }
    }
    printf("{\"gpu_selection_matches_sorted_logits\":%s,\"same_top10\":%s}\n",
           valid ? "true" : "false", selections[0] == selections[1] ? "true" : "false");
    VK_CHECK(vkDeviceWaitIdle(c.dev));
    vkDestroyCommandPool(c.dev, c.pool, nullptr); vkDestroyDevice(c.dev, nullptr); vkDestroyInstance(c.inst, nullptr);
    return valid ? 0 : 1;
}
