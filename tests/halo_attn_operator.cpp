// Standalone, model-free diagnostic. Reuse this repository's Vulkan helpers
// without rebuilding or changing the active serving library/shaders.
// Build: c++ -O2 -std=c++17 -pthread tests/halo_attn_operator.cpp -lvulkan -o OUT
// Run only in an exclusive Halo GPU window. Argument: timed repetitions 1..100.
// Synthetic operator timing is not full-model throughput or an accuracy gate.
#define QK_LIBRARY
#include "../src/main.cpp"

#include <limits>

namespace {
constexpr uint32_t capacity = 32768, heads = 24, kvHeads = 2, dim = 256;
struct AttnPC {
    uint32_t pos, tmax, dh, nRot, hQ, hKV;
    float eps, freqBase;
    uint32_t splitMax, chunk;
};
static_assert(sizeof(AttnPC) == 40, "shader push-constant layout");

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
    void dispatch(const AttnPC& pc, uint32_t x, uint32_t y = 1) {
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(c.cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcBytes, &pc);
        vkCmdDispatch(c.cb, x, y, 1);
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

std::vector<double> reference(const float* q, const float* k, const float* v,
                              const float* gates, uint32_t n) {
    std::vector<double> out(heads * dim, 0), scores(n);
    for (uint32_t h = 0; h < heads; ++h) {
        const uint32_t kv = h / (heads / kvHeads);
        double maxScore = -std::numeric_limits<double>::infinity();
        for (uint32_t t = 0; t < n; ++t) {
            double dot = 0;
            for (uint32_t d = 0; d < dim; ++d)
                dot += double(q[h * dim + d]) * k[(size_t(kv) * capacity + t) * dim + d];
            scores[t] = dot / std::sqrt(double(dim));
            maxScore = std::max(maxScore, scores[t]);
        }
        double normalizer = 0;
        for (auto& score : scores) { score = std::exp(score - maxScore); normalizer += score; }
        for (uint32_t t = 0; t < n; ++t)
            for (uint32_t d = 0; d < dim; ++d)
                out[h * dim + d] += scores[t] * v[(size_t(kv) * capacity + t) * dim + d];
        for (uint32_t d = 0; d < dim; ++d)
            out[h * dim + d] *= 1.0 / normalizer / (1.0 + std::exp(-double(gates[h * 2 * dim + dim + d])));
    }
    return out;
}

bool check(const char* mode, uint32_t n, uint32_t chunk, const float* actual,
           const std::vector<double>& expected, double micros, const float* exactRef = nullptr) {
    double worst = 0, squared = 0, norm = 0;
    bool pass = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::abs(double(actual[i]) - expected[i]);
        pass &= std::isfinite(actual[i]) && error <= 2e-5 * (1 + std::abs(expected[i]));
        worst = std::max(worst, error); squared += error * error; norm += expected[i] * expected[i];
    }
    const bool exact = !exactRef || memcmp(actual, exactRef, expected.size() * 4) == 0;
    pass &= exact;
    printf("{\"mode\":\"%s\",\"keys\":%u,\"capacity\":%u,\"chunk\":%u,\"gpu_us\":%.6f,"
           "\"max_abs_error_fp64\":%.9g,\"relative_rms_fp64\":%.9g,\"serial_exact\":%s,\"result\":\"%s\"}\n",
           mode, n, capacity, chunk, micros, worst, std::sqrt(squared / std::max(norm, 1e-30)),
           exactRef ? (exact ? "true" : "false") : "null", pass ? "PASS" : "FAIL");
    fflush(stdout);
    return pass;
}
} // namespace

int main(int argc, char** argv) {
    uint32_t repeats = 8;
    if (argc > 2) return 2;
    if (argc == 2) {
        char* end = nullptr; long value = strtol(argv[1], &end, 10);
        if (end == argv[1] || *end || value < 1 || value > 100) return 2;
        repeats = uint32_t(value);
    }
    // Pin before creating a Vulkan device. Never accept an inherited XTX override.
    unsetenv("QK_DEVICE"); unsetenv("QK_DEVICE_PCI"); setenv("QK_DEVICE_NAME", "STRIX_HALO", 1);
    VkCtx c; initVk(c, argv[0]);
    if (c.props.vendorID != 0x1002 || c.props.deviceID != 0x1586 || pciBdf(c.phys) != "0000:c1:00.0") return 2;
    bool pass = true;
    {
        MappedBuffer q(c, heads * dim), k(c, size_t(kvHeads) * capacity * dim),
            v(c, size_t(kvHeads) * capacity * dim), gates(c, heads * 2 * dim),
            position(c, 1), partial(c, size_t(heads) * ((capacity + 31) / 32) * (dim + 2)),
            serialOut(c, heads * dim, true), splitOut(c, heads * dim, true), scores(c, size_t(heads) * capacity);
        printf("operator memory: %s; staged=%s; repetitions=%u; synthetic, not model throughput\n",
               memTypeDesc(c, k.b.memType).c_str(), k.staged ? "yes" : "no", repeats); fflush(stdout);
        Kernel serial(c, "fa_attn_srv.spv", {&q, &k, &v, &gates, &serialOut, &position}, 32);
        Kernel split(c, "fa_attn_srv_split.spv", {&q, &k, &v, &partial, &position}, 40);
        Kernel group4(c, "fa_attn_srv_split_gqa.spv", {&q, &k, &v, &partial, &position}, 40, 4);
        Kernel reduce(c, "fa_attn_srv_reduce.spv", {&partial, &gates, &splitOut, &position}, 40);
        Kernel scoreKernel(c, "fa_score_srv.spv", {&q, &k, &scores, &position}, 32);
        Kernel scoreTiled4(c, "fa_score_srv_tiled.spv", {&q, &k, &scores, &position}, 32, 4);
        Kernel scoreTiled12(c, "fa_score_srv_tiled.spv", {&q, &k, &scores, &position}, 32, 12);
        Kernel ordered64(c, "fa_value_srv_ordered.spv", {&scores, &v, &gates, &splitOut, &position}, 32, 64);
        Kernel ordered128(c, "fa_value_srv_ordered.spv", {&scores, &v, &gates, &splitOut, &position}, 32, 128);
        Kernel ordered256(c, "fa_value_srv_ordered.spv", {&scores, &v, &gates, &splitOut, &position}, 32, 256);
        std::mt19937 rng(271828); std::uniform_real_distribution<float> uniform(-1, 1);
        for (uint32_t i = 0; i < heads * dim; ++i) q.ptr[i] = uniform(rng);
        for (uint32_t i = 0; i < heads * 2 * dim; ++i) gates.ptr[i] = 20 * uniform(rng);
        std::vector<float> goodK(k.b.size / 4), goodV(v.b.size / 4);
        for (auto& value : goodK) value = uniform(rng);
        for (auto& value : goodV) value = uniform(rng);
        const float poison = std::numeric_limits<float>::quiet_NaN();
        // Decreasing final length catches accidentally reusing old live partials/KV.
        for (uint32_t n : {1u, 255u, 256u, 257u, 1025u, 16433u, 32768u, 127u}) {
            std::fill(k.ptr, k.ptr + goodK.size(), poison);
            std::fill(v.ptr, v.ptr + goodV.size(), poison);
            for (uint32_t kv = 0; kv < kvHeads; ++kv) {
                const size_t offset = size_t(kv) * capacity * dim;
                memcpy(k.ptr + offset, goodK.data() + offset, size_t(n) * dim * 4);
                memcpy(v.ptr + offset, goodV.data() + offset, size_t(n) * dim * 4);
            }
            uint32_t pos = n - 1; memcpy(position.ptr, &pos, 4);
            const auto gold = reference(q.ptr, k.ptr, v.ptr, gates.ptr, n);
            AttnPC pc{0, capacity, dim, 64, heads, kvHeads, 1e-6f, 1e7f, 0, 0};
            std::fill(serialOut.ptr, serialOut.ptr + heads * dim, poison);
            timed(c, 1, [&] { serial.dispatch(pc, heads); });
            double us = timed(c, repeats, [&] { serial.dispatch(pc, heads); });
            pass &= check("serial", n, 0, serialOut.ptr, gold, us);
            for (uint32_t chunk : {32u, 127u, 256u, 1024u}) {
                pc.chunk = chunk; pc.splitMax = (capacity + chunk - 1) / chunk;
                for (uint32_t group : {1u, 4u}) {
                    auto emit = [&] {
                        if (group == 1) split.dispatch(pc, pc.splitMax, heads);
                        else group4.dispatch(pc, pc.splitMax, heads / group);
                        memoryBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
                        reduce.dispatch(pc, heads);
                    };
                    std::fill(partial.ptr, partial.ptr + partial.b.size / 4, poison);
                    std::fill(splitOut.ptr, splitOut.ptr + heads * dim, poison);
                    timed(c, 1, emit);
                    us = timed(c, repeats, emit);
                    pass &= check(group == 1 ? "split" : "gqa4", n, chunk, splitOut.ptr, gold, us);
                }
            }
            for (uint32_t scoreGroup : {0u, 4u, 12u}) for (uint32_t stripe : {64u, 128u, 256u}) {
                auto emit = [&] {
                    if (scoreGroup == 0) scoreKernel.dispatch(pc, (capacity + 255) / 256, heads);
                    else {
                        auto& score = scoreGroup == 4 ? scoreTiled4 : scoreTiled12;
                        score.dispatch(pc, (capacity + 31) / 32, heads / scoreGroup);
                    }
                    memoryBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
                    auto& valueKernel = stripe == 64 ? ordered64 : (stripe == 128 ? ordered128 : ordered256);
                    valueKernel.dispatch(pc, heads, dim / stripe);
                };
                std::fill(scores.ptr, scores.ptr + scores.b.size / 4, poison);
                std::fill(splitOut.ptr, splitOut.ptr + heads * dim, poison);
                timed(c, 1, emit);
                us = timed(c, repeats, emit);
                std::string name = "ordered-q" + std::to_string(scoreGroup) + "-dim" + std::to_string(stripe);
                pass &= check(name.c_str(), n, 0, splitOut.ptr, gold, us, serialOut.ptr);
            }
        }
    }
    VK_CHECK(vkDeviceWaitIdle(c.dev));
    vkDestroyCommandPool(c.dev, c.pool, nullptr);
    vkDestroyDevice(c.dev, nullptr); vkDestroyInstance(c.inst, nullptr);
    return pass ? 0 : 1;
}
