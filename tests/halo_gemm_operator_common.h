struct MappedBuffer {
    static inline std::vector<MappedBuffer*> registry;
    VkCtx& c;
    Buf b;
    Buf staging;
    float* ptr = nullptr;
    bool staged = false, readback = false;
    MappedBuffer(VkCtx& ctx, size_t floats, bool readBack = false) : c(ctx), readback(readBack) {
        staged = getenv("QK_OPERATOR_DEVICE_LOCAL") && !strcmp(getenv("QK_OPERATOR_DEVICE_LOCAL"), "1");
        const auto flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t type = findMemType(c.mp, ~0u, flags);
        if (type == UINT32_MAX) throw std::runtime_error("test needs coherent, host-visible device-local memory");
        const auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        b = createBuf(c, floats * 4, usage, true, staged ? -1 : (int)type);
        if (!b.deviceLocal) throw std::runtime_error("operator refused host-memory spill");
        if (staged) staging = createBuf(c, floats * 4, usage, false);
        VK_CHECK(vkMapMemory(c.dev, staged ? staging.mem : b.mem, 0, VK_WHOLE_SIZE, 0, (void**)&ptr));
        registry.push_back(this);
    }
    ~MappedBuffer() {
        registry.erase(std::remove(registry.begin(), registry.end(), this), registry.end());
        if (ptr) vkUnmapMemory(c.dev, staged ? staging.mem : b.mem);
        destroyBuf(c, staging); destroyBuf(c, b);
    }
    MappedBuffer(const MappedBuffer&) = delete;
};

struct Kernel {
    VkCtx& c;
    Pipe p;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t pcBytes;
    Kernel(VkCtx& ctx, const char* shader, std::initializer_list<MappedBuffer*> bufs,
           uint32_t bytes, uint32_t group = 0) : c(ctx), pcBytes(bytes) {
        p = makePipe(c, shader, bufs.size(), bytes, group);
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)bufs.size()};
        VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        info.maxSets = 1; info.poolSizeCount = 1; info.pPoolSizes = &size;
        VK_CHECK(vkCreateDescriptorPool(c.dev, &info, nullptr, &pool));
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &p.dsl;
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &alloc, &set));
        std::vector<VkDescriptorBufferInfo> bindings(bufs.size());
        std::vector<VkWriteDescriptorSet> writes(bufs.size());
        uint32_t i = 0;
        for (auto* buf : bufs) {
            bindings[i] = {buf->b.buf, 0, buf->b.size};
            writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = set; writes[i].dstBinding = i;
            writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bindings[i]; ++i;
        }
        vkUpdateDescriptorSets(c.dev, writes.size(), writes.data(), 0, nullptr);
    }
    ~Kernel() { vkDestroyDescriptorPool(c.dev, pool, nullptr); destroyPipe(c, p); }
    void dispatch(const GemmPC& pc, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(c.cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcBytes, &pc);
        vkCmdDispatch(c.cb, x, y, z);
    }
};

void memoryBarrier(VkCtx& c, VkPipelineStageFlags src, VkAccessFlags srcAccess,
                   VkPipelineStageFlags dst, VkAccessFlags dstAccess) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess; b.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(c.cb, src, dst, 0, 1, &b, 0, nullptr, 0, nullptr);
}

template<class Emit> double timed(VkCtx& c, uint32_t repetitions, Emit emit) {
    if (!c.hasTimestamps || !c.timestampValidBits) throw std::runtime_error("GPU timestamps required");
    VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qp.queryType = VK_QUERY_TYPE_TIMESTAMP; qp.queryCount = 2;
    VkQueryPool queries;
    VK_CHECK(vkCreateQueryPool(c.dev, &qp, nullptr, &queries));
    VK_CHECK(vkResetCommandBuffer(c.cb, 0));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(c.cb, &begin));
    // Native-like pure DEVICE_LOCAL placement is an optional, separate tier.
    // Transfers are OUTSIDE the timestamp interval; direct-map and staged
    // measurements must retain their actual memory type in the run metadata.
    memoryBarrier(c, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    for (auto* b : MappedBuffer::registry) if (b->staged) {
        VkBufferCopy copy{0, 0, b->b.size};
        vkCmdCopyBuffer(c.cb, b->staging.buf, b->b.buf, 1, &copy);
    }
    memoryBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdResetQueryPool(c.cb, queries, 0, 2);
    vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 0);
    for (uint32_t i = 0; i < repetitions; ++i) {
        emit();
        memoryBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
    vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
    memoryBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
    for (auto* b : MappedBuffer::registry) if (b->staged && b->readback) {
        VkBufferCopy copy{0, 0, b->b.size};
        vkCmdCopyBuffer(c.cb, b->b.buf, b->staging.buf, 1, &copy);
    }
    memoryBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    VK_CHECK(vkEndCommandBuffer(c.cb));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &c.cb;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
    uint64_t ticks[2];
    VK_CHECK(vkGetQueryPoolResults(c.dev, queries, 0, 2, sizeof(ticks), ticks, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    vkDestroyQueryPool(c.dev, queries, nullptr);
    const uint64_t mask = c.timestampValidBits >= 64 ? ~0ull : ((1ull << c.timestampValidBits) - 1);
    return double((ticks[1] - ticks[0]) & mask) * c.props.limits.timestampPeriod / 1000.0 / repetitions;
}
