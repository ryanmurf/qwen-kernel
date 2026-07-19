#pragma once

#include <stdexcept>

// Standalone Stage 2--4 graph gates. This header is included by main.cpp after
// the shared Vulkan helpers, so it can reuse VkCtx/Buf without coupling Gemma
// semantics to qk_engine's Qwen graph.

struct G4Invocation {
    VkCtx& c;
    Pipe pipe{};
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::vector<Buf> buffers;
    std::vector<void*> mapped;

    G4Invocation(VkCtx& context, const char* shader, uint32_t pushBytes,
                 const std::vector<size_t>& sizes,
                 const std::vector<uint32_t>& specializations = {}) : c(context) {
        pipe.nBind = (uint32_t)sizes.size();
        std::vector<VkDescriptorSetLayoutBinding> bindings(sizes.size());
        for (uint32_t i = 0; i < bindings.size(); ++i)
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                           VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dsInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsInfo.bindingCount = (uint32_t)bindings.size();
        dsInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(c.dev, &dsInfo, nullptr, &pipe.dsl));
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &pipe.dsl;
        layoutInfo.pushConstantRangeCount = pushBytes ? 1u : 0u;
        layoutInfo.pPushConstantRanges = pushBytes ? &range : nullptr;
        VK_CHECK(vkCreatePipelineLayout(c.dev, &layoutInfo, nullptr, &pipe.pl));
        auto code = loadSpv(c.argv0, shader);
        VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderInfo.codeSize = code.size()*sizeof(uint32_t);
        shaderInfo.pCode = code.data();
        VK_CHECK(vkCreateShaderModule(c.dev, &shaderInfo, nullptr, &pipe.sm));
        std::vector<VkSpecializationMapEntry> maps(specializations.size());
        for (uint32_t i = 0; i < maps.size(); ++i) maps[i] = {i, i*4u, 4};
        VkSpecializationInfo specInfo{(uint32_t)maps.size(), maps.data(),
                                      specializations.size()*sizeof(uint32_t),
                                      specializations.data()};
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = pipe.sm;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.stage.pSpecializationInfo = specializations.empty() ? nullptr : &specInfo;
        pipelineInfo.layout = pipe.pl;
        VK_CHECK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &pipelineInfo,
                                          nullptr, &pipe.p));

        buffers.reserve(sizes.size());
        mapped.resize(sizes.size());
        for (uint32_t i = 0; i < sizes.size(); ++i) {
            buffers.push_back(createBuf(c, std::max<size_t>(sizes[i], 4),
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false));
            VK_CHECK(vkMapMemory(c.dev, buffers.back().mem, 0, VK_WHOLE_SIZE, 0, &mapped[i]));
            memset(mapped[i], 0, std::max<size_t>(sizes[i], 4));
        }
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      (uint32_t)sizes.size()};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(c.dev, &poolInfo, nullptr, &descriptorPool));
        VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocateInfo.descriptorPool = descriptorPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &pipe.dsl;
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &allocateInfo, &descriptorSet));
        std::vector<VkDescriptorBufferInfo> bufferInfo(sizes.size());
        std::vector<VkWriteDescriptorSet> writes(sizes.size());
        for (uint32_t i = 0; i < sizes.size(); ++i) {
            bufferInfo[i] = {buffers[i].buf, 0, VK_WHOLE_SIZE};
            writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfo[i];
        }
        vkUpdateDescriptorSets(c.dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }

    ~G4Invocation() {
        vkDeviceWaitIdle(c.dev);
        for (uint32_t i = 0; i < buffers.size(); ++i) {
            if (mapped[i]) vkUnmapMemory(c.dev, buffers[i].mem);
            destroyBuf(c, buffers[i]);
        }
        if (descriptorPool) vkDestroyDescriptorPool(c.dev, descriptorPool, nullptr);
        destroyPipe(c, pipe);
    }

    template<class T> T* data(uint32_t binding) { return (T*)mapped[binding]; }

    void aliasBinding(uint32_t binding, uint32_t target) {
        VkDescriptorBufferInfo info{buffers[target].buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = descriptorSet;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        vkUpdateDescriptorSets(c.dev, 1, &write, 0, nullptr);
    }

    void dispatch(const void* push, uint32_t pushBytes,
                  uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        VK_CHECK(vkResetCommandBuffer(c.cb, 0));
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cb, &beginInfo));
        VkMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        hostBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &hostBarrier, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pl,
                                0, 1, &descriptorSet, 0, nullptr);
        if (pushBytes)
            vkCmdPushConstants(c.cb, pipe.pl, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, pushBytes, push);
        vkCmdDispatch(c.cb, x, y, z);
        VkMemoryBarrier readBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        readBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        readBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0,
                             1, &readBarrier, 0, nullptr, 0, nullptr);
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    }
};

static bool g4Compare(const char* name, const std::vector<float>& reference,
                      const float* actual, double limit) {
    double sumSquares = 0.0, maxAbsolute = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double error = std::fabs((double)actual[i] - reference[i]);
        if (error > maxAbsolute) { maxAbsolute = error; worst = i; }
        sumSquares += (double)reference[i]*reference[i];
    }
    const double rms = std::sqrt(sumSquares/std::max<size_t>(1, reference.size()));
    const double relative = maxAbsolute/std::max(1.0e-8, rms);
    const bool ok = relative <= limit;
    printf("  %-24s max_abs/rms=%9.3g worst=%zu -> %s\n",
           name, relative, worst, ok ? "PASS" : "FAIL");
    return ok;
}

static bool g4CompareAbsolute(const char* name, const std::vector<float>& reference,
                              const float* actual, double limit) {
    double maxAbsolute = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double error = std::fabs((double)actual[i] - reference[i]);
        if (error > maxAbsolute) { maxAbsolute = error; worst = i; }
    }
    const bool ok = maxAbsolute <= limit;
    printf("  %-24s max_abs=%9.3g worst=%zu -> %s\n",
           name, maxAbsolute, worst, ok ? "PASS" : "FAIL");
    return ok;
}

static bool g4LoadLastFloatRow(const std::string& path, size_t values,
                               std::vector<float>& output) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) return false;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return false; }
    long bytes = ftell(file);
    const long wanted = (long)(values*sizeof(float));
    if (bytes < wanted || fseek(file, bytes - wanted, SEEK_SET) != 0) {
        fclose(file); return false;
    }
    output.resize(values);
    const bool ok = fread(output.data(), sizeof(float), values, file) == values;
    fclose(file);
    return ok;
}

static bool g4CompareLlamaDump(const char* directory, const char* fileName,
                               const char* label, const std::vector<float>& actual,
                               double limit) {
    if (!directory) return true;
    std::vector<float> oracle;
    const std::string path = std::string(directory) + "/" + fileName;
    if (!g4LoadLastFloatRow(path, actual.size(), oracle)) {
        fprintf(stderr, "missing/invalid llama.cpp oracle tensor: %s\n", path.c_str());
        return false;
    }
    std::string fullLabel = std::string("llama ") + label;
    return g4Compare(fullLabel.c_str(), oracle, actual.data(), limit);
}

static std::vector<float> g4MatvecQ4Cpu(const GgufTensor* tensor,
                                        const std::vector<float>& input,
                                        uint32_t expert = 0) {
    const uint32_t K = (uint32_t)tensor->ne[0];
    const uint32_t M = (uint32_t)tensor->ne[1];
    const size_t rowBytes = ggmlRowBytes(GGML_Q4_0, K);
    const uint8_t* slice = tensor->data + (size_t)expert*M*rowBytes;
    std::vector<float> result(M), row(K);
    for (uint32_t m = 0; m < M; ++m) {
        dequant_row_q4_0((const block_q4_0*)(slice + (size_t)m*rowBytes), row.data(), K);
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; ++k) sum += row[k]*input[k];
        result[m] = sum;
    }
    return result;
}

// llama.cpp's Vulkan decode path quantizes contiguous F32 activations to
// Q8_1 before every Gemma Q4_0 matvec on AMD. Keep that backend-numerical
// reference separate from the F32 semantic reference above.
struct G4Q8_1 {
    uint16_t d;
    uint16_t s;
    int8_t qs[32];
};
static_assert(sizeof(G4Q8_1) == 36, "Q8_1 block ABI");

static std::vector<G4Q8_1> g4QuantizeQ8Cpu(const std::vector<float>& input) {
    if (input.size() % 32 != 0) throw std::runtime_error("Q8_1 input alignment");
    std::vector<G4Q8_1> output(input.size()/32);
    for (size_t block = 0; block < output.size(); ++block) {
        const float* values = input.data() + block*32;
        float maximum = 0.0f;
        for (uint32_t i = 0; i < 32; ++i)
            maximum = std::max(maximum, std::fabs(values[i]));
        const float d = maximum/127.0f;
        const float inverse = d != 0.0f ? 1.0f/d : 0.0f;
        int32_t sum = 0;
        for (uint32_t i = 0; i < 32; ++i) {
            const int32_t q = (int32_t)std::round(values[i]*inverse);
            output[block].qs[i] = (int8_t)q;
            sum += q;
        }
        output[block].d = qk_f32_to_f16(d);
        output[block].s = qk_f32_to_f16(sum*d);
    }
    return output;
}

static std::vector<float> g4MatvecQ4Q8Cpu(const GgufTensor* tensor,
                                           const std::vector<float>& input,
                                           uint32_t expert = 0) {
    const uint32_t K = (uint32_t)tensor->ne[0];
    const uint32_t M = (uint32_t)tensor->ne[1];
    const uint32_t blocksPerRow = K/32;
    const size_t rowBytes = ggmlRowBytes(GGML_Q4_0, K);
    const auto q8 = g4QuantizeQ8Cpu(input);
    const auto* rows = (const block_q4_0*)(tensor->data + (size_t)expert*M*rowBytes);
    std::vector<float> result(M, 0.0f);
    for (uint32_t row = 0; row < M; ++row) {
        float sum = 0.0f;
        for (uint32_t block = 0; block < blocksPerRow; ++block) {
            const block_q4_0& q4 = rows[(size_t)row*blocksPerRow + block];
            const float da = qk_f16_to_f32(q4.d);
            const float db = qk_f16_to_f32(q8[block].d);
            const float sb = qk_f16_to_f32(q8[block].s);
            for (uint32_t fragment = 0; fragment < 4; ++fragment) {
                int32_t integerDot = 0;
                const uint32_t offset = fragment*4;
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint8_t packed = q4.qs[offset + j];
                    integerDot += (packed & 15)*q8[block].qs[offset + j];
                    integerDot += (packed >> 4)*q8[block].qs[offset + j + 16];
                }
                sum += da*(float(integerDot)*db - 2.0f*sb);
            }
        }
        result[row] = sum;
    }
    return result;
}

static std::vector<float> g4MatvecQ4Gpu(VkCtx& c, const GgufTensor* tensor,
                                        const std::vector<float>& input,
                                        uint32_t expert = 0, uint32_t tpr = 256) {
    const uint32_t K = (uint32_t)tensor->ne[0];
    const uint32_t M = (uint32_t)tensor->ne[1];
    const size_t weightBytes = (size_t)M*ggmlRowBytes(GGML_Q4_0, K);
    G4Invocation run(c, "gemv_q4_0.spv", 8,
                     {weightBytes, (size_t)K*4, (size_t)M*4}, {tpr, K/32});
    memcpy(run.data<uint8_t>(0), tensor->data + (size_t)expert*weightBytes, weightBytes);
    memcpy(run.data<float>(1), input.data(), (size_t)K*4);
    struct { uint32_t M, K; } push{M, K};
    const uint32_t rowsPerGroup = 256/tpr;
    run.dispatch(&push, sizeof(push), (M + rowsPerGroup - 1)/rowsPerGroup);
    return std::vector<float>(run.data<float>(2), run.data<float>(2) + M);
}

static std::vector<G4Q8_1> g4QuantizeQ8Gpu(VkCtx& c,
                                           const std::vector<float>& input) {
    const size_t blocks = (input.size() + 31)/32;
    G4Invocation run(c, "gemma4_quant_q8.spv", 4,
                     {input.size()*4, blocks*sizeof(G4Q8_1)});
    memcpy(run.data<float>(0), input.data(), input.size()*4);
    const uint32_t n = (uint32_t)input.size();
    run.dispatch(&n, sizeof(n), (uint32_t)blocks);
    return std::vector<G4Q8_1>(run.data<G4Q8_1>(1),
                               run.data<G4Q8_1>(1) + blocks);
}

static std::vector<float> g4MatvecQ4Q8Gpu(VkCtx& c, const GgufTensor* tensor,
                                          const std::vector<float>& input,
                                          uint32_t expert = 0, uint32_t tpr = 64) {
    const uint32_t K = (uint32_t)tensor->ne[0];
    const uint32_t M = (uint32_t)tensor->ne[1];
    const size_t weightBytes = (size_t)M*ggmlRowBytes(GGML_Q4_0, K);
    const auto q8 = g4QuantizeQ8Gpu(c, input);
    G4Invocation run(c, "gemma4_gemv_q4_q8.spv", 8,
                     {weightBytes, q8.size()*sizeof(G4Q8_1), (size_t)M*4}, {tpr});
    memcpy(run.data<uint8_t>(0), tensor->data + (size_t)expert*weightBytes, weightBytes);
    memcpy(run.data<G4Q8_1>(1), q8.data(), q8.size()*sizeof(G4Q8_1));
    struct { uint32_t M, K; } push{M, K};
    const uint32_t rowsPerGroup = 256/tpr;
    run.dispatch(&push, sizeof(push), (M + rowsPerGroup - 1)/rowsPerGroup);
    return std::vector<float>(run.data<float>(2), run.data<float>(2) + M);
}

static std::vector<float> g4RmsGpu(VkCtx& c, const std::vector<float>& input,
                                   const float* weight, uint32_t vectorSize,
                                   uint32_t vectors = 1, float postScale = 1.0f) {
    const size_t n = (size_t)vectorSize*vectors;
    G4Invocation run(c, "gemma4_rms.spv", 16,
                     {n*4, (size_t)vectorSize*4, n*4});
    memcpy(run.data<float>(0), input.data(), n*4);
    if (weight) memcpy(run.data<float>(1), weight, (size_t)vectorSize*4);
    else std::fill(run.data<float>(1), run.data<float>(1) + vectorSize, 1.0f);
    struct { uint32_t n, weighted; float eps, postScale; }
        push{vectorSize, weight ? 1u : 0u, gemma4::kRmsEpsilon, postScale};
    run.dispatch(&push, sizeof(push), vectors);
    return std::vector<float>(run.data<float>(2), run.data<float>(2) + n);
}

static std::vector<float> g4ElementwiseGpu(VkCtx& c,
                                           const std::vector<float>& a,
                                           const std::vector<float>& b,
                                           uint32_t mode, float scale = 1.0f) {
    const size_t n = a.size();
    G4Invocation run(c, "gemma4_elementwise.spv", 20, {n*4, n*4, n*4});
    memcpy(run.data<float>(0), a.data(), n*4);
    if (!b.empty()) memcpy(run.data<float>(1), b.data(), std::min(n, b.size())*4);
    struct { uint32_t n, mode; float scale; uint32_t lo, hi; }
        push{(uint32_t)n, mode, scale, 0, 0};
    run.dispatch(&push, sizeof(push), ((uint32_t)n + 255)/256);
    return std::vector<float>(run.data<float>(2), run.data<float>(2) + n);
}

static bool caseGemma4Stage2(VkCtx& c) {
    printf("\n== Gemma 4 Stage 2: dense graph primitives ==\n");
    Gemma4Stage1Weights model;
    if (!loadGemma4Stage1(model)) return false;
    bool ok = true;

    // Embedding dequant plus sqrt(2816), using a descriptor-local view of one
    // tied Q6_K row so the gate does not duplicate the 605 MiB tensor.
    const uint32_t token = 198;
    const size_t embeddingBytes = ggmlRowBytes(GGML_Q6_K, gemma4::kEmbedding);
    std::vector<float> embeddingRef(gemma4::kEmbedding);
    dequant_row_q6_K((const block_q6_K*)(model.tokenEmbedding->data + token*embeddingBytes),
                     embeddingRef.data(), gemma4::kEmbedding);
    for (float& value : embeddingRef) value *= std::sqrt((float)gemma4::kEmbedding);
    G4Invocation embedding(c, "gemma4_embed.spv", 8,
                           {embeddingBytes, 4, gemma4::kEmbedding*4});
    memcpy(embedding.data<uint8_t>(0), model.tokenEmbedding->data + token*embeddingBytes,
           embeddingBytes);
    embedding.data<uint32_t>(1)[0] = 0;
    struct { uint32_t kdim, tokenIndex; } embedPush{gemma4::kEmbedding, 0};
    embedding.dispatch(&embedPush, sizeof(embedPush), 1);
    ok &= g4Compare("embedding * sqrt(2816)", embeddingRef,
                    embedding.data<float>(2), 3.0e-5);

    // Weighted and unweighted norm variants.
    auto input = randomX(gemma4::kEmbedding);
    std::vector<float> rmsRef(gemma4::kEmbedding);
    gemma4::rmsNorm(input.data(), (const float*)model.layers[0].ffnNorm->data,
                    rmsRef.data(), gemma4::kEmbedding);
    auto rmsGpu = g4RmsGpu(c, input, (const float*)model.layers[0].ffnNorm->data,
                           gemma4::kEmbedding);
    ok &= g4Compare("weighted RMS", rmsRef, rmsGpu.data(), 2.0e-5);
    gemma4::rmsNorm(input.data(), nullptr, rmsRef.data(), gemma4::kEmbedding);
    rmsGpu = g4RmsGpu(c, input, nullptr, gemma4::kEmbedding);
    ok &= g4Compare("unweighted RMS", rmsRef, rmsGpu.data(), 2.0e-5);

    // Complete shared/dense branch in the exact Gemma order:
    // ffn_norm -> GELU(gate)*up -> down -> post_ffw_norm_1.
    gemma4::rmsNorm(input.data(), (const float*)model.layers[0].ffnNorm->data,
                    rmsRef.data(), gemma4::kEmbedding);
    auto gateRef = g4MatvecQ4Q8Cpu(model.layers[0].ffnGate, rmsRef);
    auto upRef = g4MatvecQ4Q8Cpu(model.layers[0].ffnUp, rmsRef);
    std::vector<float> hiddenRef(gemma4::kSharedFf);
    for (uint32_t i = 0; i < gemma4::kSharedFf; ++i)
        hiddenRef[i] = gemma4::gelu(gateRef[i])*upRef[i];
    auto downRef = g4MatvecQ4Q8Cpu(model.layers[0].ffnDown, hiddenRef);
    std::vector<float> denseRef(gemma4::kEmbedding);
    gemma4::rmsNorm(downRef.data(), (const float*)model.layers[0].postFfwNorm1->data,
                    denseRef.data(), gemma4::kEmbedding);

    auto denseInGpu = g4RmsGpu(c, input, (const float*)model.layers[0].ffnNorm->data,
                               gemma4::kEmbedding);
    auto gateGpu = g4MatvecQ4Q8Gpu(c, model.layers[0].ffnGate, denseInGpu);
    auto upGpu = g4MatvecQ4Q8Gpu(c, model.layers[0].ffnUp, denseInGpu);
    auto hiddenGpu = g4ElementwiseGpu(c, gateGpu, upGpu, 0);
    auto downGpu = g4MatvecQ4Q8Gpu(c, model.layers[0].ffnDown, hiddenGpu);
    auto denseGpu = g4RmsGpu(c, downGpu, (const float*)model.layers[0].postFfwNorm1->data,
                             gemma4::kEmbedding);
    ok &= g4Compare("dense gate", gateRef, gateGpu.data(), 2.5e-3);
    ok &= g4Compare("dense up", upRef, upGpu.data(), 2.5e-3);
    ok &= g4Compare("dense GELU product", hiddenRef, hiddenGpu.data(), 4.0e-3);
    ok &= g4Compare("dense down", downRef, downGpu.data(), 6.0e-3);
    ok &= g4Compare("dense post norm", denseRef, denseGpu.data(), 6.0e-3);

    // The output head is the same Q6_K storage as token_embd. Exercise a
    // contiguous vocabulary slice with the native 2816-wide specialization;
    // the full 262144-row dispatch differs only in row count.
    constexpr uint32_t headFirstRow = 4096;
    constexpr uint32_t headRows = 256;
    const size_t q6RowBytes = ggmlRowBytes(GGML_Q6_K, gemma4::kEmbedding);
    std::vector<float> headInput = randomX(gemma4::kEmbedding);
    std::vector<float> headRef(headRows), headRow(gemma4::kEmbedding);
    for (uint32_t row = 0; row < headRows; ++row) {
        const auto* packed = (const block_q6_K*)(model.tokenEmbedding->data +
            (size_t)(headFirstRow + row)*q6RowBytes);
        dequant_row_q6_K(packed, headRow.data(), gemma4::kEmbedding);
        float sum = 0.0f;
        for (uint32_t i = 0; i < gemma4::kEmbedding; ++i)
            sum += headRow[i]*headInput[i];
        headRef[row] = sum;
    }
    G4Invocation head(c, "gemv_q6_k.spv", 8,
        {headRows*q6RowBytes, gemma4::kEmbedding*4, headRows*4}, {128, 11});
    memcpy(head.data<uint8_t>(0), model.tokenEmbedding->data +
           (size_t)headFirstRow*q6RowBytes, headRows*q6RowBytes);
    memcpy(head.data<float>(1), headInput.data(), gemma4::kEmbedding*4);
    struct { uint32_t M, K; } headPush{headRows, gemma4::kEmbedding};
    head.dispatch(&headPush, sizeof(headPush), headRows/2);
    std::vector<float> headGpu(head.data<float>(2), head.data<float>(2) + headRows);
    ok &= g4Compare("tied Q6_K head slice", headRef, headGpu.data(), 2.5e-3);
    std::vector<float> softcapRef(headRows);
    for (uint32_t i = 0; i < headRows; ++i)
        softcapRef[i] = gemma4::kSoftcap*std::tanh(headGpu[i]/gemma4::kSoftcap);
    auto softcapGpu = g4ElementwiseGpu(c, headGpu, {}, 3, gemma4::kSoftcap);
    ok &= g4Compare("30*tanh(logit/30)", softcapRef, softcapGpu.data(), 2.0e-5);

    // Residual/output scaling and softcap are materialized primitives.
    std::vector<float> residualRef(gemma4::kEmbedding);
    const float layerScale = *(const float*)model.layers[0].layerOutputScale->data;
    for (uint32_t i = 0; i < gemma4::kEmbedding; ++i)
        residualRef[i] = (denseGpu[i] + input[i])*layerScale;
    auto residualGpu = g4ElementwiseGpu(c, denseGpu, input, 1, layerScale);
    ok &= g4Compare("residual/output scale", residualRef, residualGpu.data(), 2.0e-7);

    std::vector<float> logits(1024, -20.0f);
    logits[17] = 4.0f;
    logits[23] = std::nextafter(4.0f, -INFINITY);
    logits[41] = 4.0f;  // exact tie: lower id 17 must win
    G4Invocation argmax(c, "gemma4_argmax.spv", 16, {logits.size()*4, 4});
    memcpy(argmax.data<float>(0), logits.data(), logits.size()*4);
    struct { uint32_t n; float cap; uint32_t lo, hi; }
        argPush{(uint32_t)logits.size(), gemma4::kSoftcap, 0, 0};
    argmax.dispatch(&argPush, sizeof(argPush), 1);
    const uint32_t winner = argmax.data<uint32_t>(1)[0];
    printf("  %-24s winner=%u expected=17 -> %s\n", "softcap near-tie argmax",
           winner, winner == 17 ? "PASS" : "FAIL");
    ok &= winner == 17;

    printf("Stage 2 gate: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static float g4FixtureValue(uint32_t seed, uint32_t position, uint32_t index) {
    uint32_t value = seed ^ (position + 1u)*0x9e3779b9u ^ (index + 3u)*0x85ebca6bu;
    value ^= value >> 16; value *= 0x7feb352du;
    value ^= value >> 15; value *= 0x846ca68bu;
    value ^= value >> 16;
    return ((int32_t)(value & 0xffffu) - 32768) / 16384.0f;
}

static void g4AttentionPrepCpu(const gemma4::AttentionConfig& cfg, uint32_t cacheLength,
                               uint32_t position, const std::vector<float>& rawQ,
                               const std::vector<float>& rawK,
                               const std::vector<float>& rawV, const float* qWeight,
                               const float* kWeight, const float* ropeFactors,
                               std::vector<float>& query, std::vector<float>& keyCache,
                               std::vector<float>& valueCache) {
    std::vector<float> norm(cfg.headDim);
    const uint32_t slot = cfg.sliding ? position % cacheLength : position;
    auto prepareRotary = [&](const float* source, const float* weight, float* destination) {
        gemma4::rmsNorm(source, weight, norm.data(), cfg.headDim);
        const uint32_t half = cfg.ropeDim/2;
        for (uint32_t pair = 0; pair < half; ++pair) {
            const float angle = position*gemma4::ropeFrequency(pair, cfg.ropeDim,
                                                               cfg.ropeBase, ropeFactors);
            const float cosine = std::cos(angle), sine = std::sin(angle);
            const float first = norm[pair], second = norm[pair + half];
            destination[pair] = first*cosine - second*sine;
            destination[pair + half] = first*sine + second*cosine;
        }
        for (uint32_t dim = cfg.ropeDim; dim < cfg.headDim; ++dim)
            destination[dim] = norm[dim];
    };
    for (uint32_t head = 0; head < cfg.queryHeads; ++head)
        prepareRotary(rawQ.data() + (size_t)head*cfg.headDim, qWeight,
                      query.data() + (size_t)head*cfg.headDim);
    for (uint32_t head = 0; head < cfg.kvHeads; ++head) {
        float* keyOut = keyCache.data() + ((size_t)head*cacheLength + slot)*cfg.headDim;
        prepareRotary(rawK.data() + (size_t)head*cfg.headDim, kWeight, keyOut);
        float* valueOut = valueCache.data() + ((size_t)head*cacheLength + slot)*cfg.headDim;
        gemma4::rmsNorm(rawV.data() + (size_t)head*cfg.headDim, nullptr, valueOut,
                        cfg.headDim);
    }
}

static void g4AttentionCpu(const gemma4::AttentionConfig& cfg, uint32_t cacheLength,
                           uint32_t position, const std::vector<float>& query,
                           const std::vector<float>& keyCache,
                           const std::vector<float>& valueCache, uint32_t probabilityStride,
                           std::vector<float>& probabilities, std::vector<float>& output) {
    const uint32_t start = cfg.sliding && position + 1 > cacheLength
                         ? position + 1 - cacheLength : 0;
    const uint32_t count = position + 1 - start;
    const uint32_t group = cfg.queryHeads/cfg.kvHeads;
    for (uint32_t qHead = 0; qHead < cfg.queryHeads; ++qHead) {
        const uint32_t kvHead = qHead/group;
        float maximum = -INFINITY;
        for (uint32_t local = 0; local < count; ++local) {
            const uint32_t absolute = start + local;
            const uint32_t slot = cfg.sliding ? absolute % cacheLength : absolute;
            const float* key = keyCache.data() + ((size_t)kvHead*cacheLength + slot)*cfg.headDim;
            float score = 0.0f;  // f_attention_scale is deliberately 1.0
            for (uint32_t dim = 0; dim < cfg.headDim; ++dim)
                score += query[(size_t)qHead*cfg.headDim + dim]*key[dim];
            probabilities[(size_t)qHead*probabilityStride + local] = score;
            maximum = std::max(maximum, score);
        }
        float sum = 0.0f;
        for (uint32_t local = 0; local < count; ++local) {
            float& value = probabilities[(size_t)qHead*probabilityStride + local];
            value = std::exp(value - maximum);
            sum += value;
        }
        for (uint32_t local = 0; local < count; ++local)
            probabilities[(size_t)qHead*probabilityStride + local] /= sum;
        for (uint32_t dim = 0; dim < cfg.headDim; ++dim) {
            float accumulated = 0.0f;
            for (uint32_t local = 0; local < count; ++local) {
                const uint32_t absolute = start + local;
                const uint32_t slot = cfg.sliding ? absolute % cacheLength : absolute;
                const float* value = valueCache.data() +
                    ((size_t)kvHead*cacheLength + slot)*cfg.headDim;
                accumulated += probabilities[(size_t)qHead*probabilityStride + local]*value[dim];
            }
            output[(size_t)qHead*cfg.headDim + dim] = accumulated;
        }
    }
}

static bool g4RunAttentionGeometry(VkCtx& c, const Gemma4Stage1Weights& model,
                                   uint32_t layerIndex) {
    const auto& layer = model.layers[layerIndex];
    const auto cfg = gemma4::attentionConfig(layerIndex);
    const uint32_t lastPosition = 1025;
    const uint32_t cacheLength = cfg.sliding ? gemma4::kSlidingWindow : lastPosition + 1;
    const uint32_t probabilityStride = cacheLength;
    const size_t qValues = (size_t)cfg.queryHeads*cfg.headDim;
    const size_t kvValues = (size_t)cfg.kvHeads*cfg.headDim;
    const size_t cacheValues = (size_t)cfg.kvHeads*cacheLength*cfg.headDim;
    std::vector<float> rawQ(qValues), rawK(kvValues), rawV(kvValues), query(qValues);
    std::vector<float> keyCache(cacheValues), valueCache(cacheValues);
    const float* qWeight = (const float*)layer.attnQNorm->data;
    const float* kWeight = (const float*)layer.attnKNorm->data;
    const float* factors = cfg.sliding ? nullptr : (const float*)model.ropeFreqs->data;
    std::vector<float> factorBuffer(cfg.headDim/2, 1.0f);
    if (factors) memcpy(factorBuffer.data(), factors, factorBuffer.size()*4);

    G4Invocation prep(c, "gemma4_attn_prep.spv", 40,
        {qValues*4, kvValues*4, kvValues*4, (size_t)cfg.headDim*4,
         (size_t)cfg.headDim*4, factorBuffer.size()*4, qValues*4,
         cacheValues*4, cacheValues*4});
    memcpy(prep.data<float>(3), qWeight, (size_t)cfg.headDim*4);
    memcpy(prep.data<float>(4), kWeight, (size_t)cfg.headDim*4);
    memcpy(prep.data<float>(5), factorBuffer.data(), factorBuffer.size()*4);
    if (!cfg.sliding) prep.aliasBinding(2, 1);  // global raw K is the raw V input

    G4Invocation attention(c, "gemma4_attn.spv", 32,
        {qValues*4, cacheValues*4, cacheValues*4,
         (size_t)cfg.queryHeads*probabilityStride*4, qValues*4});
    bool ok = true;
    static constexpr uint32_t checkpoints[] = {0, 1, 1023, 1024, 1025};
    size_t checkpoint = 0;
    for (uint32_t position = 0; position <= lastPosition; ++position) {
        for (uint32_t i = 0; i < qValues; ++i)
            rawQ[i] = g4FixtureValue(0x1234u + layerIndex, position, i);
        for (uint32_t i = 0; i < kvValues; ++i)
            rawK[i] = g4FixtureValue(0x5678u + layerIndex, position, i);
        if (cfg.sliding) {
            for (uint32_t i = 0; i < kvValues; ++i)
                rawV[i] = g4FixtureValue(0x9abcu + layerIndex, position, i);
        } else {
            rawV = rawK;
        }
        g4AttentionPrepCpu(cfg, cacheLength, position, rawQ, rawK, rawV,
                           qWeight, kWeight, factors, query, keyCache, valueCache);
        memcpy(prep.data<float>(0), rawQ.data(), qValues*4);
        memcpy(prep.data<float>(1), rawK.data(), kvValues*4);
        if (cfg.sliding) memcpy(prep.data<float>(2), rawV.data(), kvValues*4);
        struct { uint32_t position, headDim, queryHeads, kvHeads, ropeDim,
                          cacheLength, sliding, useFactors; float eps, ropeBase; }
            prepPush{position, cfg.headDim, cfg.queryHeads, cfg.kvHeads, cfg.ropeDim,
                     cacheLength, cfg.sliding ? 1u : 0u, factors ? 1u : 0u,
                     gemma4::kRmsEpsilon, cfg.ropeBase};
        prep.dispatch(&prepPush, sizeof(prepPush), cfg.queryHeads + 2*cfg.kvHeads);

        if (checkpoint >= std::size(checkpoints) || position != checkpoints[checkpoint]) continue;
        char label[96];
        snprintf(label, sizeof(label), "L%u p%u Q post-RoPE", layerIndex, position);
        // RADV native sin/cos and libc sinf/cosf differ by roughly 1e-4 at
        // absolute positions around 1K. The bound remains tighter than the
        // accepted Stage-1 Q4 dequant/GEMV corpus.
        ok &= g4Compare(label, query, prep.data<float>(6), 2.0e-4);
        const uint32_t slot = cfg.sliding ? position % cacheLength : position;
        std::vector<float> currentK(kvValues), currentV(kvValues);
        std::vector<float> gpuK(kvValues), gpuV(kvValues);
        for (uint32_t head = 0; head < cfg.kvHeads; ++head) {
            size_t cacheOffset = ((size_t)head*cacheLength + slot)*cfg.headDim;
            memcpy(currentK.data() + (size_t)head*cfg.headDim,
                   keyCache.data() + cacheOffset, cfg.headDim*4);
            memcpy(currentV.data() + (size_t)head*cfg.headDim,
                   valueCache.data() + cacheOffset, cfg.headDim*4);
            memcpy(gpuK.data() + (size_t)head*cfg.headDim,
                   prep.data<float>(7) + cacheOffset, cfg.headDim*4);
            memcpy(gpuV.data() + (size_t)head*cfg.headDim,
                   prep.data<float>(8) + cacheOffset, cfg.headDim*4);
        }
        snprintf(label, sizeof(label), "L%u p%u K cache slot %u", layerIndex, position, slot);
        ok &= g4Compare(label, currentK, gpuK.data(), 2.0e-4);
        snprintf(label, sizeof(label), "L%u p%u V cache slot %u", layerIndex, position, slot);
        ok &= g4Compare(label, currentV, gpuV.data(), 8.0e-5);
        if (!cfg.sliding) {
            bool distinct = memcmp(gpuK.data(), gpuV.data(), kvValues*4) != 0;
            printf("  L%u p%u global K/V caches distinct -> %s\n", layerIndex, position,
                   distinct ? "PASS" : "FAIL");
            ok &= distinct;
        }

        std::vector<float> probabilityRef((size_t)cfg.queryHeads*probabilityStride, 0.0f);
        std::vector<float> outputRef(qValues), gpuProb(probabilityRef.size());
        g4AttentionCpu(cfg, cacheLength, position, query, keyCache, valueCache,
                       probabilityStride, probabilityRef, outputRef);
        memcpy(attention.data<float>(0), prep.data<float>(6), qValues*4);
        memcpy(attention.data<float>(1), prep.data<float>(7), cacheValues*4);
        memcpy(attention.data<float>(2), prep.data<float>(8), cacheValues*4);
        struct { uint32_t position, headDim, queryHeads, kvHeads, cacheLength,
                          sliding, probabilityStride, unused; }
            attnPush{position, cfg.headDim, cfg.queryHeads, cfg.kvHeads, cacheLength,
                     cfg.sliding ? 1u : 0u, probabilityStride, 0};
        attention.dispatch(&attnPush, sizeof(attnPush), cfg.kvHeads);
        snprintf(label, sizeof(label), "L%u p%u probabilities", layerIndex, position);
        // Probability RMS is close to zero for the deliberately sharp random
        // score corpus; an absolute bound is the meaningful parity metric.
        ok &= g4CompareAbsolute(label, probabilityRef, attention.data<float>(3), 1.0e-4);
        snprintf(label, sizeof(label), "L%u p%u attention output", layerIndex, position);
        ok &= g4Compare(label, outputRef, attention.data<float>(4), 4.0e-4);
        if (layerIndex == 0 || layerIndex == 5) {
            std::vector<float> attentionGpu(attention.data<float>(4),
                                            attention.data<float>(4) + qValues);
            // Gate the projection as its own primitive using the exact GPU
            // attention input; the preceding attention output is independently
            // checked above, avoiding Q8_1 threshold amplification at p1025.
            auto projectedRef = g4MatvecQ4Q8Cpu(layer.attnOut, attentionGpu);
            auto projectedGpu = g4MatvecQ4Q8Gpu(c, layer.attnOut, attentionGpu);
            snprintf(label, sizeof(label), "L%u p%u attention output projection",
                     layerIndex, position);
            ok &= g4Compare(label, projectedRef, projectedGpu.data(), 5.0e-3);
            std::vector<float> postNormRef(gemma4::kEmbedding);
            gemma4::rmsNorm(projectedGpu.data(),
                            (const float*)layer.postAttentionNorm->data,
                            postNormRef.data(), gemma4::kEmbedding);
            auto postNormGpu = g4RmsGpu(c, projectedGpu,
                (const float*)layer.postAttentionNorm->data, gemma4::kEmbedding);
            std::vector<float> residual(gemma4::kEmbedding);
            for (uint32_t i = 0; i < gemma4::kEmbedding; ++i)
                residual[i] = g4FixtureValue(0xc001u + layerIndex, position, i);
            std::vector<float> hiddenRef(gemma4::kEmbedding);
            for (uint32_t i = 0; i < gemma4::kEmbedding; ++i)
                hiddenRef[i] = postNormRef[i] + residual[i];
            auto hiddenGpu = g4ElementwiseGpu(c, postNormGpu, residual, 2);
            snprintf(label, sizeof(label), "L%u p%u attention hidden state",
                     layerIndex, position);
            ok &= g4Compare(label, hiddenRef, hiddenGpu.data(), 2.0e-5);
        }
        ++checkpoint;
    }
    return ok;
}

static bool g4CheckLlamaAttentionDump(VkCtx& c, const Gemma4Stage1Weights& model,
                                      const char* directory) {
    if (!directory) return true;
    bool ok = true;
    for (uint32_t layerIndex : {0u, 5u}) {
        const auto cfgForLayer = gemma4::attentionConfig(layerIndex);
        const auto& weights = model.layers[layerIndex];
        std::vector<float> projectionInput, rawQOracle, rawKOracle, rawVOracle;
        const std::string suffix = "-" + std::to_string(layerIndex) + ".bin";
        const std::string root(directory);
        bool loadedProjection =
            g4LoadLastFloatRow(root + "/attn_norm" + suffix,
                               gemma4::kEmbedding, projectionInput) &&
            g4LoadLastFloatRow(root + "/Qcur" + suffix,
                               (size_t)cfgForLayer.queryHeads*cfgForLayer.headDim,
                               rawQOracle);
        if (cfgForLayer.sliding) {
            loadedProjection &=
                g4LoadLastFloatRow(root + "/Kcur" + suffix,
                                   (size_t)cfgForLayer.kvHeads*cfgForLayer.headDim,
                                   rawKOracle) &&
                g4LoadLastFloatRow(root + "/Vcur" + suffix,
                                   (size_t)cfgForLayer.kvHeads*cfgForLayer.headDim,
                                   rawVOracle);
        } else {
            // gemma4.cpp names the single shared raw K/V projection Vcur.
            loadedProjection &=
                g4LoadLastFloatRow(root + "/Vcur" + suffix,
                                   (size_t)cfgForLayer.kvHeads*cfgForLayer.headDim,
                                   rawKOracle);
        }
        if (!loadedProjection) {
            fprintf(stderr, "Gemma Stage 3 llama.cpp projection dump set is incomplete for L%u in %s\n",
                    layerIndex, directory);
            return false;
        }
        auto rawQGpu = g4MatvecQ4Q8Gpu(c, weights.attnQ, projectionInput);
        auto rawKGpu = g4MatvecQ4Q8Gpu(c, weights.attnK, projectionInput);
        char label[72];
        snprintf(label, sizeof(label), "llama L%u raw Q projection", layerIndex);
        ok &= g4Compare(label, rawQOracle, rawQGpu.data(), 2.5e-3);
        snprintf(label, sizeof(label), "llama L%u raw K projection", layerIndex);
        ok &= g4Compare(label, rawKOracle, rawKGpu.data(), 2.5e-3);
        if (cfgForLayer.sliding) {
            auto rawVGpu = g4MatvecQ4Q8Gpu(c, weights.attnV, projectionInput);
            snprintf(label, sizeof(label), "llama L%u raw V projection", layerIndex);
            ok &= g4Compare(label, rawVOracle, rawVGpu.data(), 2.5e-3);
        } else {
            printf("  llama L%u raw V aliases raw K input -> PASS\n", layerIndex);
        }
    }
    const auto cfg = gemma4::kSlidingAttention;
    const auto& layer = model.layers[0];
    const size_t qValues = (size_t)cfg.queryHeads*cfg.headDim;
    const size_t kvValues = (size_t)cfg.kvHeads*cfg.headDim;
    std::vector<float> rawQ, rawK, rawV, qOracle, kOracle, vOracle;
    const std::string base(directory);
    bool loaded = g4LoadLastFloatRow(base + "/Qcur-0.bin", qValues, rawQ) &&
                  g4LoadLastFloatRow(base + "/Kcur-0.bin", kvValues, rawK) &&
                  g4LoadLastFloatRow(base + "/Vcur-0.bin", kvValues, rawV) &&
                  g4LoadLastFloatRow(base + "/Qcur_pos-0.bin", qValues, qOracle) &&
                  g4LoadLastFloatRow(base + "/Kcur_pos-0.bin", kvValues, kOracle) &&
                  g4LoadLastFloatRow(base + "/Vcur_normed-0.bin", kvValues, vOracle);
    if (!loaded) {
        fprintf(stderr, "Gemma Stage 3 llama.cpp dump set is incomplete in %s\n", directory);
        return false;
    }
    const uint32_t cacheLength = 2;
    const size_t cacheValues = (size_t)cfg.kvHeads*cacheLength*cfg.headDim;
    std::vector<float> factors(cfg.headDim/2, 1.0f);
    G4Invocation prep(c, "gemma4_attn_prep.spv", 40,
        {qValues*4, kvValues*4, kvValues*4, (size_t)cfg.headDim*4,
         (size_t)cfg.headDim*4, factors.size()*4, qValues*4,
         cacheValues*4, cacheValues*4});
    memcpy(prep.data<float>(0), rawQ.data(), qValues*4);
    memcpy(prep.data<float>(1), rawK.data(), kvValues*4);
    memcpy(prep.data<float>(2), rawV.data(), kvValues*4);
    memcpy(prep.data<float>(3), layer.attnQNorm->data, cfg.headDim*4);
    memcpy(prep.data<float>(4), layer.attnKNorm->data, cfg.headDim*4);
    memcpy(prep.data<float>(5), factors.data(), factors.size()*4);
    struct { uint32_t position, headDim, queryHeads, kvHeads, ropeDim,
                      cacheLength, sliding, useFactors; float eps, ropeBase; }
        push{1, cfg.headDim, cfg.queryHeads, cfg.kvHeads, cfg.ropeDim,
             cacheLength, 1, 0, gemma4::kRmsEpsilon, cfg.ropeBase};
    prep.dispatch(&push, sizeof(push), cfg.queryHeads + 2*cfg.kvHeads);
    ok &= g4Compare("llama Q post-RoPE L0/p1", qOracle, prep.data<float>(6), 2.0e-4);
    std::vector<float> gpuK(kvValues), gpuV(kvValues);
    for (uint32_t head = 0; head < cfg.kvHeads; ++head) {
        const size_t source = ((size_t)head*cacheLength + 1)*cfg.headDim;
        memcpy(gpuK.data() + (size_t)head*cfg.headDim,
               prep.data<float>(7) + source, cfg.headDim*4);
        memcpy(gpuV.data() + (size_t)head*cfg.headDim,
               prep.data<float>(8) + source, cfg.headDim*4);
    }
    ok &= g4Compare("llama K post-RoPE L0/p1", kOracle, gpuK.data(), 2.0e-4);
    ok &= g4Compare("llama normalized V L0/p1", vOracle, gpuV.data(), 2.0e-5);
    return ok;
}

static bool caseGemma4Stage3(VkCtx& c) {
    printf("\n== Gemma 4 Stage 3: sliding/global attention ==\n");
    Gemma4Stage1Weights model;
    if (!loadGemma4Stage1(model)) return false;
    bool ok = true;
    constexpr uint32_t layers[] = {0, 5, 11, 17, 23, 29};
    for (uint32_t layer : layers) {
        const auto cfg = gemma4::attentionConfig(layer);
        const auto& weights = model.layers[layer];
        bool shapeOk = weights.sliding == cfg.sliding && weights.attnQ->ne[1] ==
            (uint64_t)cfg.queryHeads*cfg.headDim && weights.attnK->ne[1] ==
            (uint64_t)cfg.kvHeads*cfg.headDim &&
            (cfg.sliding ? weights.attnV != nullptr : weights.attnV == nullptr);
        printf("  layer %u geometry hd=%u q=%u kv=%u rawV=%s -> %s\n", layer,
               cfg.headDim, cfg.queryHeads, cfg.kvHeads,
               weights.attnV ? "tensor" : "reuse-K", shapeOk ? "PASS" : "FAIL");
        ok &= shapeOk;

        // Actual Q/K/V projections at the first token. Global V is deliberately
        // not dispatched: its input is the K projection result.
        auto projectedInput = randomX(gemma4::kEmbedding);
        auto qRef = g4MatvecQ4Q8Cpu(weights.attnQ, projectedInput);
        auto qGpu = g4MatvecQ4Q8Gpu(c, weights.attnQ, projectedInput);
        auto kRef = g4MatvecQ4Q8Cpu(weights.attnK, projectedInput);
        auto kGpu = g4MatvecQ4Q8Gpu(c, weights.attnK, projectedInput);
        char label[64];
        snprintf(label, sizeof(label), "L%u Q projection", layer);
        ok &= g4Compare(label, qRef, qGpu.data(), 2.5e-3);
        snprintf(label, sizeof(label), "L%u K projection", layer);
        ok &= g4Compare(label, kRef, kGpu.data(), 2.5e-3);
        if (cfg.sliding) {
            auto vRef = g4MatvecQ4Q8Cpu(weights.attnV, projectedInput);
            auto vGpu = g4MatvecQ4Q8Gpu(c, weights.attnV, projectedInput);
            snprintf(label, sizeof(label), "L%u V projection", layer);
            ok &= g4Compare(label, vRef, vGpu.data(), 2.5e-3);
        } else {
            printf("  L%u V projection reuses raw K buffer -> PASS\n", layer);
        }
        ok &= g4RunAttentionGeometry(c, model, layer);
    }
    ok &= g4CheckLlamaAttentionDump(c, model, getenv("QK_G4_ORACLE_DIR"));
    printf("Stage 3 gate: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static std::vector<float> g4RouterCpu(const Gemma4LayerWeights& layer,
                                      const std::vector<float>& attnOut) {
    std::vector<float> normalized(gemma4::kEmbedding), logits(gemma4::kExperts);
    gemma4::rmsNorm(attnOut.data(), nullptr, normalized.data(), gemma4::kEmbedding,
                    gemma4::kRmsEpsilon, 1.0f/std::sqrt((float)gemma4::kEmbedding));
    const float* scales = (const float*)layer.routerScale->data;
    const float* weights = (const float*)layer.router->data;
    for (uint32_t expert = 0; expert < gemma4::kExperts; ++expert) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < gemma4::kEmbedding; ++k)
            sum += weights[(size_t)expert*gemma4::kEmbedding + k] * scales[k] * normalized[k];
        logits[expert] = sum;
    }
    return logits;
}

static std::vector<float> g4RouterGpu(VkCtx& c, const Gemma4LayerWeights& layer,
                                      const std::vector<float>& attnOut) {
    auto normalized = g4RmsGpu(c, attnOut, nullptr, gemma4::kEmbedding, 1,
                               1.0f/std::sqrt((float)gemma4::kEmbedding));
    const size_t routerBytes = (size_t)gemma4::kEmbedding*gemma4::kExperts*4;
    G4Invocation run(c, "gemma4_router.spv", 8,
        {gemma4::kEmbedding*4, gemma4::kEmbedding*4, routerBytes,
         gemma4::kExperts*4});
    memcpy(run.data<float>(0), normalized.data(), normalized.size()*4);
    memcpy(run.data<float>(1), layer.routerScale->data, gemma4::kEmbedding*4);
    memcpy(run.data<float>(2), layer.router->data, routerBytes);
    struct { uint32_t n, experts; } push{gemma4::kEmbedding, gemma4::kExperts};
    run.dispatch(&push, sizeof(push), gemma4::kExperts);
    return std::vector<float>(run.data<float>(3),
                              run.data<float>(3) + gemma4::kExperts);
}

static gemma4::Selection g4SelectGpu(VkCtx& c, const std::vector<float>& logits) {
    G4Invocation run(c, "gemma4_select.spv", 0,
                     {gemma4::kExperts*4, sizeof(gemma4::Selection)});
    memcpy(run.data<float>(0), logits.data(), gemma4::kExperts*4);
    run.dispatch(nullptr, 0, 1);
    return *run.data<gemma4::Selection>(1);
}

static std::vector<float> g4ExpertGateUpCpu(const GgufTensor* tensor,
                                            const std::vector<float>& input,
                                            const gemma4::Selection& selection) {
    std::vector<float> hidden((size_t)gemma4::kExpertsUsed*gemma4::kExpertFf);
    for (uint32_t rank = 0; rank < gemma4::kExpertsUsed; ++rank) {
        auto gateUp = g4MatvecQ4Q8Cpu(tensor, input, selection.ids[rank]);
        for (uint32_t row = 0; row < gemma4::kExpertFf; ++row)
            hidden[(size_t)rank*gemma4::kExpertFf + row] =
                gemma4::gelu(gateUp[row])*gateUp[gemma4::kExpertFf + row];
    }
    return hidden;
}

static std::vector<float> g4ExpertGateUpGpu(VkCtx& c, const GgufTensor* tensor,
                                            const std::vector<float>& input,
                                            const gemma4::Selection& selection) {
    const size_t hiddenValues = (size_t)gemma4::kExpertsUsed*gemma4::kExpertFf;
    const auto q8 = g4QuantizeQ8Gpu(c, input);
    G4Invocation run(c, "gemma4_moe_gateup.spv", 0,
        {(size_t)tensor->nbytes, q8.size()*sizeof(G4Q8_1), sizeof(selection),
         hiddenValues*4}, {64});
    memcpy(run.data<uint8_t>(0), tensor->data, tensor->nbytes);
    memcpy(run.data<G4Q8_1>(1), q8.data(), q8.size()*sizeof(G4Q8_1));
    memcpy(run.data<gemma4::Selection>(2), &selection, sizeof(selection));
    run.dispatch(nullptr, 0, gemma4::kExpertFf, 2);
    return std::vector<float>(run.data<float>(3), run.data<float>(3) + hiddenValues);
}

static std::vector<float> g4ExpertDownCpu(const Gemma4LayerWeights& layer,
                                          const std::vector<float>& hidden,
                                          const gemma4::Selection& selection) {
    std::vector<float> result(gemma4::kEmbedding, 0.0f), one(gemma4::kExpertFf);
    const float* scales = (const float*)layer.expertDownScale->data;
    for (uint32_t rank = 0; rank < gemma4::kExpertsUsed; ++rank) {
        memcpy(one.data(), hidden.data() + (size_t)rank*gemma4::kExpertFf,
               gemma4::kExpertFf*4);
        auto contribution = g4MatvecQ4Cpu(layer.expertDown, one, selection.ids[rank]);
        const float scale = selection.weights[rank]*scales[selection.ids[rank]];
        for (uint32_t i = 0; i < gemma4::kEmbedding; ++i)
            result[i] += contribution[i]*scale;
    }
    return result;
}

static std::vector<float> g4ExpertDownGpu(VkCtx& c, const Gemma4LayerWeights& layer,
                                          const std::vector<float>& hidden,
                                          const gemma4::Selection& selection) {
    G4Invocation run(c, "gemma4_moe_down.spv", 0,
        {(size_t)layer.expertDown->nbytes, hidden.size()*4,
         sizeof(selection), gemma4::kExperts*4, gemma4::kEmbedding*4}, {64});
    memcpy(run.data<uint8_t>(0), layer.expertDown->data, layer.expertDown->nbytes);
    memcpy(run.data<float>(1), hidden.data(), hidden.size()*4);
    memcpy(run.data<gemma4::Selection>(2), &selection, sizeof(selection));
    memcpy(run.data<float>(3), layer.expertDownScale->data, gemma4::kExperts*4);
    run.dispatch(nullptr, 0, gemma4::kEmbedding/4);
    return std::vector<float>(run.data<float>(4),
                              run.data<float>(4) + gemma4::kEmbedding);
}

static bool g4BenchmarkGroupedMoe(VkCtx& c, const Gemma4LayerWeights& layer,
                                  uint32_t iterations);

static bool caseGemma4Stage4(VkCtx& c, uint32_t benchmarkIterations) {
    printf("\n== Gemma 4 Stage 4: shared FFN + grouped routed MoE ==\n");
    Gemma4Stage1Weights model;
    if (!loadGemma4Stage1(model)) return false;
    const auto& layer = model.layers[0];
    bool ok = true;
    auto attnOut = randomX(gemma4::kEmbedding);
    const char* oracleDirectory = getenv("QK_G4_ORACLE_DIR");
    if (oracleDirectory) {
        const std::string path = std::string(oracleDirectory) + "/attn_out-0.bin";
        if (!g4LoadLastFloatRow(path, gemma4::kEmbedding, attnOut)) {
            fprintf(stderr, "cannot load llama.cpp attn_out oracle: %s\n", path.c_str());
            return false;
        }
        printf("  oracle input: llama.cpp attn_out-0 final token\n");
    }

    // Router path is intentionally independent from the routed FFN pre-norm.
    auto routerRef = g4RouterCpu(layer, attnOut);
    auto routerGpu = g4RouterGpu(c, layer, attnOut);
    ok &= g4Compare("router logits", routerRef, routerGpu.data(), 3.0e-5);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_moe_logits-0.bin", "router logits",
                             routerGpu, 2.0e-4);
    const auto selectedRef = gemma4::stableTop8(routerRef.data());
    const auto selectedGpu = g4SelectGpu(c, routerGpu);
    bool idsOk = true;
    std::vector<float> selectedWeightRef(gemma4::kExpertsUsed);
    for (uint32_t rank = 0; rank < gemma4::kExpertsUsed; ++rank) {
        idsOk &= selectedGpu.ids[rank] == selectedRef.ids[rank];
        selectedWeightRef[rank] = selectedRef.weights[rank];
    }
    printf("  router top-8 ids        ");
    for (uint32_t id : selectedGpu.ids) printf("%u ", id);
    printf("-> %s\n", idsOk ? "PASS" : "FAIL");
    ok &= idsOk;
    ok &= g4CompareAbsolute("router top-8 weights", selectedWeightRef,
                            selectedGpu.weights, 2.0e-6);
    if (oracleDirectory) {
        std::vector<float> selectedWeights(selectedGpu.weights,
                                           selectedGpu.weights + gemma4::kExpertsUsed);
        ok &= g4CompareLlamaDump(oracleDirectory, "ffn_moe_weights_norm-0.bin",
                                 "top-8 weights", selectedWeights, 2.0e-5);
    }

    // Stable adversarial ties, including a near-tie immediately below them.
    std::vector<float> tied(gemma4::kExperts, -10.0f);
    const uint32_t tieIds[8] = {3, 9, 12, 17, 20, 50, 90, 127};
    for (uint32_t id : tieIds) tied[id] = 5.0f;
    tied[2] = std::nextafter(5.0f, -INFINITY);
    const auto tiedGpu = g4SelectGpu(c, tied);
    bool tiesOk = true;
    for (uint32_t rank = 0; rank < 8; ++rank) tiesOk &= tiedGpu.ids[rank] == tieIds[rank];
    printf("  adversarial stable ties ids=[3,9,12,17,20,50,90,127] -> %s\n",
           tiesOk ? "PASS" : "FAIL");
    ok &= tiesOk;
    if (!ok) {
        printf("Stage 4 gate: FAIL before expert dispatch; stopping\n");
        return false;
    }

    // Shared branch, in parallel off the unmodified attn_out.
    std::vector<float> sharedInputRef(gemma4::kEmbedding);
    gemma4::rmsNorm(attnOut.data(), (const float*)layer.ffnNorm->data,
                    sharedInputRef.data(), gemma4::kEmbedding);
    auto sharedGateRef = g4MatvecQ4Q8Cpu(layer.ffnGate, sharedInputRef);
    auto sharedUpRef = g4MatvecQ4Q8Cpu(layer.ffnUp, sharedInputRef);
    std::vector<float> sharedHiddenRef(gemma4::kSharedFf);
    for (uint32_t i = 0; i < gemma4::kSharedFf; ++i)
        sharedHiddenRef[i] = gemma4::gelu(sharedGateRef[i])*sharedUpRef[i];
    auto sharedDownRef = g4MatvecQ4Q8Cpu(layer.ffnDown, sharedHiddenRef);
    std::vector<float> sharedBranchRef(gemma4::kEmbedding);
    gemma4::rmsNorm(sharedDownRef.data(), (const float*)layer.postFfwNorm1->data,
                    sharedBranchRef.data(), gemma4::kEmbedding);

    auto sharedInputGpu = g4RmsGpu(c, attnOut, (const float*)layer.ffnNorm->data,
                                   gemma4::kEmbedding);
    auto sharedGateGpu = g4MatvecQ4Q8Gpu(c, layer.ffnGate, sharedInputGpu);
    auto sharedUpGpu = g4MatvecQ4Q8Gpu(c, layer.ffnUp, sharedInputGpu);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_gate-0.bin", "shared gate projection",
                             sharedGateGpu, 3.0e-3);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_up-0.bin", "shared up projection",
                             sharedUpGpu, 3.0e-3);
    auto sharedHiddenGpu = g4ElementwiseGpu(c, sharedGateGpu, sharedUpGpu, 0);
    auto sharedDownGpu = g4MatvecQ4Q8Gpu(c, layer.ffnDown, sharedHiddenGpu);
    std::vector<float> sharedPostPrimitiveRef(gemma4::kEmbedding);
    gemma4::rmsNorm(sharedDownGpu.data(), (const float*)layer.postFfwNorm1->data,
                    sharedPostPrimitiveRef.data(), gemma4::kEmbedding);
    auto sharedBranchGpu = g4RmsGpu(c, sharedDownGpu,
                                    (const float*)layer.postFfwNorm1->data,
                                    gemma4::kEmbedding);
    ok &= g4Compare("shared pre-norm", sharedInputRef, sharedInputGpu.data(), 2.0e-5);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_norm_1-0.bin", "shared pre-norm",
                             sharedInputGpu, 2.0e-4);
    ok &= g4Compare("shared GELU tensor", sharedHiddenRef, sharedHiddenGpu.data(), 4.0e-3);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_geglu-0.bin", "shared GELU tensor",
                             sharedHiddenGpu, 3.0e-3);
    ok &= g4Compare("shared down tensor", sharedDownRef, sharedDownGpu.data(), 6.0e-3);
    ok &= g4Compare("shared post-norm", sharedPostPrimitiveRef,
                    sharedBranchGpu.data(), 6.0e-3);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_mlp-0.bin", "shared post-norm",
                             sharedBranchGpu, 3.0e-3);

    // Routed branch has its own learned pre/post norms, but consumes the
    // selection produced from the distinct unweighted router path above.
    std::vector<float> routedInputRef(gemma4::kEmbedding);
    gemma4::rmsNorm(attnOut.data(), (const float*)layer.preFfwNorm2->data,
                    routedInputRef.data(), gemma4::kEmbedding);
    auto routedInputGpu = g4RmsGpu(c, attnOut,
                                   (const float*)layer.preFfwNorm2->data,
                                   gemma4::kEmbedding);
    ok &= g4Compare("routed pre-norm", routedInputRef, routedInputGpu.data(), 2.0e-5);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_norm_2-0.bin", "routed pre-norm",
                             routedInputGpu, 2.0e-4);
    auto expertHiddenRef = g4ExpertGateUpCpu(layer.expertGateUp, routedInputRef,
                                             selectedRef);
    auto expertHiddenGpu = g4ExpertGateUpGpu(c, layer.expertGateUp, routedInputGpu,
                                             selectedGpu);
    ok &= g4Compare("8-expert GELU gate/up", expertHiddenRef,
                    expertHiddenGpu.data(), 5.0e-3);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_moe_geglu-0.bin",
                             "8-expert GELU gate/up", expertHiddenGpu, 4.0e-3);
    auto routedDownRef = g4ExpertDownCpu(layer, expertHiddenRef, selectedRef);
    auto routedDownGpu = g4ExpertDownGpu(c, layer, expertHiddenGpu, selectedGpu);
    ok &= g4Compare("weighted expert down", routedDownRef, routedDownGpu.data(), 8.0e-3);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_moe_out-0.bin",
                             "weighted expert down", routedDownGpu, 5.0e-3);
    std::vector<float> routedBranchRef(gemma4::kEmbedding);
    gemma4::rmsNorm(routedDownGpu.data(), (const float*)layer.postFfwNorm2->data,
                    routedBranchRef.data(), gemma4::kEmbedding);
    auto routedBranchGpu = g4RmsGpu(c, routedDownGpu,
                                    (const float*)layer.postFfwNorm2->data,
                                    gemma4::kEmbedding);
    ok &= g4Compare("routed post-norm", routedBranchRef,
                    routedBranchGpu.data(), 8.0e-3);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_moe-0.bin", "routed post-norm",
                             routedBranchGpu, 5.0e-3);

    // Parallel branch sum -> shared post-FFW norm -> attention residual ->
    // scalar layer output scale. This is the complete sparse block epilogue.
    std::vector<float> branchSumRef(gemma4::kEmbedding);
    for (uint32_t i = 0; i < gemma4::kEmbedding; ++i)
        branchSumRef[i] = sharedBranchGpu[i] + routedBranchGpu[i];
    auto branchSumGpu = g4ElementwiseGpu(c, sharedBranchGpu, routedBranchGpu, 2);
    ok &= g4Compare("parallel branch sum", branchSumRef, branchSumGpu.data(), 8.0e-3);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_moe_combined-0.bin",
                             "parallel branch sum", branchSumGpu, 5.0e-3);
    std::vector<float> postFfwRef(gemma4::kEmbedding);
    gemma4::rmsNorm(branchSumGpu.data(), (const float*)layer.postFfwNorm->data,
                    postFfwRef.data(), gemma4::kEmbedding);
    auto postFfwGpu = g4RmsGpu(c, branchSumGpu, (const float*)layer.postFfwNorm->data,
                               gemma4::kEmbedding);
    ok &= g4Compare("post_ffw_norm", postFfwRef, postFfwGpu.data(), 8.0e-3);
    ok &= g4CompareLlamaDump(oracleDirectory, "ffn_post_norm-0.bin", "post_ffw_norm",
                             postFfwGpu, 5.0e-3);
    const float outputScale = *(const float*)layer.layerOutputScale->data;
    std::vector<float> blockRef(gemma4::kEmbedding);
    for (uint32_t i = 0; i < gemma4::kEmbedding; ++i)
        blockRef[i] = (postFfwGpu[i] + attnOut[i])*outputScale;
    auto blockGpu = g4ElementwiseGpu(c, postFfwGpu, attnOut, 1, outputScale);
    ok &= g4Compare("complete sparse block", blockRef, blockGpu.data(), 8.0e-3);

    if (ok && benchmarkIterations) ok &= g4BenchmarkGroupedMoe(c, layer, benchmarkIterations);
    printf("Stage 4 gate: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static uint32_t g4WaitForIdleBusy() {
    const char* path = getenv("QK_GPU_BUSY_PATH");
    if (!path) path = "/sys/class/drm/card2/device/gpu_busy_percent";
    uint32_t value = UINT32_MAX;
    for (uint32_t attempt = 0; attempt < 40; ++attempt) {
        if (FILE* file = fopen(path, "r")) {
            if (fscanf(file, "%u", &value) != 1) value = UINT32_MAX;
            fclose(file);
        }
        if (value == 0 || value == UINT32_MAX) break;
        usleep(50000);
    }
    return value;
}

static bool g4BenchmarkGroupedMoe(VkCtx& c, const Gemma4LayerWeights& layer,
                                  uint32_t iterations) {
    if (!c.hasTimestamps || iterations == 0) {
        fprintf(stderr, "grouped MoE benchmark requires Vulkan timestamps\n");
        return false;
    }
    printf("\n== grouped top-8 cache-cold benchmark (%u iterations) ==\n", iterations);
    constexpr uint32_t routes = gemma4::kExperts/gemma4::kExpertsUsed;
    const size_t gateBytes = layer.expertGateUp->nbytes;
    const size_t downBytes = layer.expertDown->nbytes;
    const size_t xBytes = (gemma4::kEmbedding/32)*sizeof(G4Q8_1);
    const size_t hiddenBytes = (size_t)gemma4::kExpertsUsed*gemma4::kExpertFf*4;
    const size_t outputBytes = gemma4::kEmbedding*4;
    const VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    Buf gateWeights = createBuf(c, gateBytes, deviceUsage, true);
    Buf downWeights = createBuf(c, downBytes, deviceUsage, true);
    Buf inputFloat = createBuf(c, gemma4::kEmbedding*4, deviceUsage, true);
    Buf input = createBuf(c, xBytes, deviceUsage, true);
    Buf hidden = createBuf(c, hiddenBytes, deviceUsage, true);
    Buf scales = createBuf(c, gemma4::kExperts*4, deviceUsage, true);
    Buf output = createBuf(c, outputBytes, deviceUsage, true);
    Buf selections = createBuf(c, routes*sizeof(gemma4::Selection),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    const size_t stageBytes = std::max({gateBytes, downBytes, hiddenBytes,
                                       xBytes, (size_t)gemma4::kEmbedding*4,
                                       (size_t)gemma4::kExperts*4});
    Buf stage = createBuf(c, stageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
    void* stageMap = nullptr;
    void* selectionMap = nullptr;
    VK_CHECK(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &stageMap));
    VK_CHECK(vkMapMemory(c.dev, selections.mem, 0, VK_WHOLE_SIZE, 0, &selectionMap));
    for (uint32_t route = 0; route < routes; ++route) {
        auto& selection = ((gemma4::Selection*)selectionMap)[route];
        for (uint32_t rank = 0; rank < gemma4::kExpertsUsed; ++rank) {
            selection.ids[rank] = route*gemma4::kExpertsUsed + rank;
            selection.weights[rank] = 1.0f/gemma4::kExpertsUsed;
        }
    }
    auto begin = [&]() {
        VK_CHECK(vkResetCommandBuffer(c.cb, 0));
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cb, &info));
    };
    auto submit = [&]() {
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        info.commandBufferCount = 1;
        info.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &info, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    };
    auto upload = [&](Buf& destination, const void* source, size_t bytes) {
        memcpy(stageMap, source, bytes);
        begin();
        VkBufferCopy copy{0, 0, bytes};
        vkCmdCopyBuffer(c.cb, stage.buf, destination.buf, 1, &copy);
        submit();
    };
    upload(gateWeights, layer.expertGateUp->data, gateBytes);
    upload(downWeights, layer.expertDown->data, downBytes);
    auto randomInput = randomX(gemma4::kEmbedding);
    auto randomHidden = randomX((uint32_t)(hiddenBytes/4));
    auto randomInputQ8 = g4QuantizeQ8Cpu(randomInput);
    upload(inputFloat, randomInput.data(), gemma4::kEmbedding*4);
    upload(input, randomInputQ8.data(), xBytes);
    upload(hidden, randomHidden.data(), hiddenBytes);
    upload(scales, layer.expertDownScale->data, gemma4::kExperts*4);

    // Stage 1 proved that TPR32 does not generalize across shapes. Grouping
    // changes occupancy again, so sweep the complete 32/64/128/256 set rather
    // than inheriting either the isolated or large-matrix winner.
    constexpr uint32_t tprs[] = {32, 64, 128, 256};
    std::array<Pipe, 4> gatePipes, downPipes;
    Pipe quantPipe = makePipe(c, "gemma4_quant_q8.spv", 2, 4);
    for (uint32_t i = 0; i < 4; ++i) {
        gatePipes[i] = makePipe(c, "gemma4_moe_gateup.spv", 4, 4, tprs[i]);
        downPipes[i] = makePipe(c, "gemma4_moe_down.spv", 5, 4, tprs[i]);
    }
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  routes*(4u + 5u) + 2u};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = routes*2 + 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &poolInfo, nullptr, &descriptorPool));
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.insert(layouts.end(), routes, gatePipes[0].dsl);
    layouts.insert(layouts.end(), routes, downPipes[0].dsl);
    layouts.push_back(quantPipe.dsl);
    std::vector<VkDescriptorSet> sets(routes*2 + 1);
    VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = (uint32_t)sets.size();
    allocateInfo.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(c.dev, &allocateInfo, sets.data()));
    auto writeSet = [&](VkDescriptorSet set, const std::vector<Buf*>& bound,
                        uint32_t route) {
        std::vector<VkDescriptorBufferInfo> infos(bound.size());
        std::vector<VkWriteDescriptorSet> writes(bound.size());
        for (uint32_t binding = 0; binding < bound.size(); ++binding) {
            if (bound[binding] == &selections)
                infos[binding] = {selections.buf, route*sizeof(gemma4::Selection),
                                  sizeof(gemma4::Selection)};
            else
                infos[binding] = {bound[binding]->buf, 0, VK_WHOLE_SIZE};
            writes[binding] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[binding].dstSet = set;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &infos[binding];
        }
        vkUpdateDescriptorSets(c.dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    };
    for (uint32_t route = 0; route < routes; ++route) {
        writeSet(sets[route], {&gateWeights, &input, &selections, &hidden}, route);
        writeSet(sets[routes + route],
                 {&downWeights, &hidden, &selections, &scales, &output}, route);
    }
    writeSet(sets[routes*2], {&inputFloat, &input}, 0);
    VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 2;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateQueryPool(c.dev, &queryInfo, nullptr, &queryPool));
    VkMemoryBarrier shaderBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    shaderBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    shaderBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    enum class Phase { Quant, GateUp, Down, Pair, Full };
    struct Timing { double ns; uint32_t busy; };
    auto timePhase = [&](Phase phase, uint32_t gateIndex, uint32_t downIndex) {
        Timing timing{};
        timing.busy = g4WaitForIdleBusy();
        begin();
        vkCmdResetQueryPool(c.cb, queryPool, 0, 2);
        vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
            const uint32_t route = iteration % routes;
            if (phase == Phase::Quant || phase == Phase::Full) {
                vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, quantPipe.p);
                vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        quantPipe.pl, 0, 1, &sets[routes*2], 0, nullptr);
                const uint32_t n = gemma4::kEmbedding;
                vkCmdPushConstants(c.cb, quantPipe.pl, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(n), &n);
                vkCmdDispatch(c.cb, gemma4::kEmbedding/32, 1, 1);
                vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &shaderBarrier, 0, nullptr, 0, nullptr);
            }
            if (phase == Phase::GateUp || phase == Phase::Pair || phase == Phase::Full) {
                Pipe& gatePipe = gatePipes[gateIndex];
                vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, gatePipe.p);
                vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        gatePipe.pl, 0, 1, &sets[route], 0, nullptr);
                const uint32_t groups = 256/tprs[gateIndex];
                vkCmdDispatch(c.cb, gemma4::kExpertFf,
                              (gemma4::kExpertsUsed + groups - 1)/groups, 1);
                vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &shaderBarrier, 0, nullptr, 0, nullptr);
            }
            if (phase == Phase::Down || phase == Phase::Pair || phase == Phase::Full) {
                Pipe& downPipe = downPipes[downIndex];
                vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, downPipe.p);
                vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        downPipe.pl, 0, 1, &sets[routes + route], 0, nullptr);
                const uint32_t groups = 256/tprs[downIndex];
                vkCmdDispatch(c.cb, (gemma4::kEmbedding + groups - 1)/groups, 1, 1);
                vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &shaderBarrier, 0, nullptr, 0, nullptr);
            }
        }
        vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
        submit();
        uint64_t timestamps[2]{};
        VK_CHECK(vkGetQueryPoolResults(c.dev, queryPool, 0, 2, sizeof(timestamps),
                                       timestamps, sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        timing.ns = (double)(timestamps[1] - timestamps[0]) *
                    c.props.limits.timestampPeriod / iterations;
        return timing;
    };
    auto report = [&](const char* name, Phase phase, uint32_t gateIndex,
                      uint32_t downIndex, double payload, double baseline) {
        Timing first = timePhase(phase, gateIndex, downIndex);
        Timing second = timePhase(phase, gateIndex, downIndex);
        const double rate = payload/second.ns;
        printf("  %-10s cold=%7.2f us gpu_busy_percent=%s%u | "
               "repeat=%7.2f us gpu_busy_percent=%s%u | %6.1f GB/s | "
               "vs isolated %.1f GB/s: %+5.1f%% | cold/repeat %.3fx\n",
               name, first.ns/1000.0, first.busy == UINT32_MAX ? "n/a:" : "",
               first.busy == UINT32_MAX ? 0 : first.busy,
               second.ns/1000.0, second.busy == UINT32_MAX ? "n/a:" : "",
               second.busy == UINT32_MAX ? 0 : second.busy,
               rate, baseline, (rate/baseline - 1.0)*100.0, first.ns/second.ns);
        return second.ns;
    };
    const double gatePayload = 8.0*2230272.0;
    const double downPayload = 8.0*1115136.0;
    Timing quantFirst = timePhase(Phase::Quant, 0, 0);
    Timing quantRepeat = timePhase(Phase::Quant, 0, 0);
    printf("  %-10s cold=%7.2f us gpu_busy_percent=%s%u | "
           "repeat=%7.2f us gpu_busy_percent=%s%u\n",
           "quant_q8", quantFirst.ns/1000.0,
           quantFirst.busy == UINT32_MAX ? "n/a:" : "",
           quantFirst.busy == UINT32_MAX ? 0 : quantFirst.busy,
           quantRepeat.ns/1000.0,
           quantRepeat.busy == UINT32_MAX ? "n/a:" : "",
           quantRepeat.busy == UINT32_MAX ? 0 : quantRepeat.busy);
    uint32_t bestGate = 0, bestDown = 0;
    double bestGateNs = INFINITY, bestDownNs = INFINITY;
    for (uint32_t i = 0; i < 4; ++i) {
        char name[24]; snprintf(name, sizeof(name), "gate_up/%u", tprs[i]);
        double ns = report(name, Phase::GateUp, i, 0, gatePayload, 416.4);
        if (ns < bestGateNs) { bestGateNs = ns; bestGate = i; }
    }
    for (uint32_t i = 0; i < 4; ++i) {
        char name[24]; snprintf(name, sizeof(name), "down/%u", tprs[i]);
        double ns = report(name, Phase::Down, 0, i, downPayload, 259.5);
        if (ns < bestDownNs) { bestDownNs = ns; bestDown = i; }
    }
    const double isolatedPairRate = (gatePayload + downPayload) /
        (gatePayload/416.4 + downPayload/259.5);
    char pairName[32];
    snprintf(pairName, sizeof(pairName), "pair/%u+%u", tprs[bestGate], tprs[bestDown]);
    const double pairNs = report(pairName, Phase::Pair, bestGate, bestDown,
                                 gatePayload + downPayload, isolatedPairRate);
    Timing fullFirst = timePhase(Phase::Full, bestGate, bestDown);
    Timing fullRepeat = timePhase(Phase::Full, bestGate, bestDown);
    printf("  grouped winners: gate_up TPR=%u %.1f GB/s; down TPR=%u %.1f GB/s\n",
           tprs[bestGate], gatePayload/bestGateNs, tprs[bestDown], downPayload/bestDownNs);
    printf("  grouped pair effective: %.1f GB/s, %.2f us/layer (two dispatches)\n",
           (gatePayload + downPayload)/pairNs, pairNs/1000.0);
    printf("  routed full path: %.2f us/layer (Q8_1 quant + two grouped dispatches), "
           "gpu_busy_percent=%s%u; cold %.2f us, gpu_busy_percent=%s%u\n",
           fullRepeat.ns/1000.0,
           fullRepeat.busy == UINT32_MAX ? "n/a:" : "",
           fullRepeat.busy == UINT32_MAX ? 0 : fullRepeat.busy,
           fullFirst.ns/1000.0,
           fullFirst.busy == UINT32_MAX ? "n/a:" : "",
           fullFirst.busy == UINT32_MAX ? 0 : fullFirst.busy);

    vkDestroyQueryPool(c.dev, queryPool, nullptr);
    vkDestroyDescriptorPool(c.dev, descriptorPool, nullptr);
    for (uint32_t i = 0; i < 4; ++i) {
        destroyPipe(c, gatePipes[i]);
        destroyPipe(c, downPipes[i]);
    }
    destroyPipe(c, quantPipe);
    vkUnmapMemory(c.dev, stage.mem);
    vkUnmapMemory(c.dev, selections.mem);
    for (Buf* buffer : {&gateWeights, &downWeights, &inputFloat, &input, &hidden, &scales,
                        &output, &selections, &stage}) destroyBuf(c, *buffer);
    return true;
}
