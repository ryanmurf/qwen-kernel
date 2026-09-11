// Included after the Vulkan helpers by the CLI harness. These are operator
// tests, NOT a qwen4exp serving implementation or a complete HC module.
static bool caseQwen4Hc(VkCtx& c, uint32_t n, uint32_t h, uint32_t tokens) {
    if (!n || n > 16384 || !h || h > 64 || !tokens || tokens > 32) return false;
    struct { uint32_t n, h, tokens, mode; float eps; } pc{n, h, tokens, 0, 1e-6f};
    const size_t wide = (size_t)n * h * tokens;
    std::vector<float> host[5];
    host[0] = randomX((uint32_t)wide);
    host[1] = randomX(n * h, 59);
    host[2] = randomX((uint32_t)wide, 71);
    host[3] = randomX(n * tokens, 83);
    host[4].resize(wide);
    // Include distinct stream scales, zero inputs, and saturated gates.
    for (size_t i = 0; i < wide; ++i) {
        host[0][i] *= 1.0f + float(i / n % h);
        if (i < n || i % 31 == 0) host[0][i] = 0.0f;
        host[2][i] *= 100.0f;
    }
    for (auto& v : host[1]) v += 1.0f;
    Buf buf[5];
    float* ptr[5]{};
    for (unsigned i = 0; i < 5; ++i) {
        buf[i] = createBuf(c, host[i].size() * sizeof(float),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
        VK_CHECK(vkMapMemory(c.dev, buf[i].mem, 0, VK_WHOLE_SIZE, 0, (void**)&ptr[i]));
        memcpy(ptr[i], host[i].data(), host[i].size() * sizeof(float));
    }
    Pipe pipe = makePipe(c, "qwen4_hc.spv", 5, sizeof(pc));
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1; poolInfo.poolSizeCount = 1; poolInfo.pPoolSizes = &poolSize;
    VkDescriptorPool pool;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &poolInfo, nullptr, &pool));
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &pipe.dsl;
    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(c.dev, &alloc, &set));
    VkDescriptorBufferInfo info[5];
    VkWriteDescriptorSet writes[5]{};
    for (unsigned i = 0; i < 5; ++i) {
        info[i] = {buf[i].buf, 0, buf[i].size};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set; writes[i].dstBinding = i;
        writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &info[i];
    }
    vkUpdateDescriptorSets(c.dev, 5, writes, 0, nullptr);
    const auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    bool pass = true;
    for (pc.mode = 0; pc.mode < 5; ++pc.mode) {
        // NaN poison catches missed rows, tail lanes and missing token batches.
        std::fill(ptr[4], ptr[4] + wide, std::numeric_limits<float>::quiet_NaN());
        std::vector<double> reference(pc.mode == 1 ? (size_t)n * tokens : wide, 0.0);
        if (pc.mode == 0 || pc.mode == 3) {
            for (uint32_t t = 0; t < tokens; ++t) for (uint32_t s = 0; s < h; ++s) {
                size_t base = ((size_t)t * h + s) * n;
                double sq = 0.0;
                for (uint32_t i = 0; i < n; ++i) sq += double(host[0][base+i]) * host[0][base+i];
                double scale = 1.0 / std::sqrt(sq / n + pc.eps);
                for (uint32_t i = 0; i < n; ++i)
                    reference[base+i] = host[0][base+i] * scale * host[1][(pc.mode == 3 ? 0 : s*n)+i]
                        * (pc.mode == 3 ? sigmoid(host[2][base+i]) : 1.0);
            }
        } else if (pc.mode == 1) {
            for (uint32_t t = 0; t < tokens; ++t) for (uint32_t i = 0; i < n; ++i)
                for (uint32_t s = 0; s < h; ++s) {
                    size_t j = ((size_t)t * h + s) * n + i;
                    reference[(size_t)t*n+i] += host[0][j] * sigmoid(host[2][j]) / h;
                }
        } else if (pc.mode == 2) {
            for (size_t j = 0; j < wide; ++j) {
                size_t t = j / (n*h), s = j / n % h;
                reference[j] = host[0][j] + host[3][t*n + j%n] *
                    (2.0 * sigmoid(double(host[2][t*h+s]) / h));
            }
        } else {
            for (uint32_t t = 0; t < tokens; ++t) for (uint32_t s = 0; s < h; ++s) {
                size_t base = ((size_t)t * h + s) * n;
                double dot = 0;
                for (uint32_t i = 0; i < n; ++i) dot += double(host[0][base+i]) * host[2][base+i];
                double score = dot / std::sqrt(double(n));
                double mag = std::sqrt(std::max(std::fabs(score), 1e-6));
                double gate = sigmoid(score > 0 ? mag : (score < 0 ? -mag : 0));
                for (uint32_t i = 0; i < n; ++i) reference[base+i] = host[3][t*n+i] * gate;
            }
        }
        VK_CHECK(vkResetCommandBuffer(c.cb, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(c.cb, &begin));
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pl, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(c.cb, pipe.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        uint32_t gx = pc.mode == 0 || pc.mode >= 3 ? h : ((pc.mode == 1 ? n : n*h) + 255) / 256;
        vkCmdDispatch(c.cb, gx, 1, tokens);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1; submit.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &submit, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
        bool ok = true;
        double worst = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            double delta = std::fabs(ptr[4][i] - reference[i]);
            ok &= std::isfinite(ptr[4][i]) && delta <= 2e-5 * (1.0 + std::fabs(reference[i]));
            worst = std::max(worst, delta);
        }
        printf("HC mode=%u N=%u H=%u T=%u max_abs_error=%.3g: %s\n",
               pc.mode, n, h, tokens, worst, ok ? "PASS" : "FAIL");
        pass &= ok;
    }
    vkDestroyDescriptorPool(c.dev, pool, nullptr);
    destroyPipe(c, pipe);
    for (unsigned i = 0; i < 5; ++i) { vkUnmapMemory(c.dev, buf[i].mem); destroyBuf(c, buf[i]); }
    return pass;
}
