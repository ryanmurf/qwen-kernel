// Single-sequence native qwen4exp graph. Prefix correctness tests use the same
// kernels as the experimental split-stage adapter, with a small weight budget.
#include "qwen4_ple.h"
#include <atomic>
#include <memory>
#include <numeric>
#include <set>
#include <thread>
#include <sys/mman.h>
#include <unistd.h>
class Qwen4Graph {
    VkCtx& c;
    Gguf& g;
    std::map<std::string, Buf> buffers;
    std::map<std::string, Pipe> pipes;
    std::map<std::string, std::string> taps;
    uint32_t firstLayer = 0, lastLayer = 0, layer = 0, position = 0, capacity = 32;
    uint64_t weightLimit = 8ull<<30;
    bool withHead = false;
    bool servingBudget = false;
    bool replayReady = false;
    std::vector<float> logits;
    std::unique_ptr<Qwen4PleLookup> ple;
    std::vector<uint32_t> tokenHistory;
    // Background page-cache prefetch of the disk-mapped PLE table (rows are
    // hash-random, so a cold table costs one NVMe page fault per row).
    std::thread pleThread;
    std::atomic<bool> pleStop{false};
    struct PageRange { uintptr_t start; size_t bytes; };
    std::vector<PageRange> pleKeep;  // table ranges made resident; paged out on close
    // Two-heap placement (single-device serving on a UMA part): tensors named
    // here are allocated from the host-visible heap (hostHeapType) instead of
    // device-local memory. Everything else, including all activations, KV and
    // recurrent state, stays device-local. systemReserveBytes is the
    // MemAvailable floor enforced after every uploaded tensor (0 = unchecked).
    std::set<std::string> hostHeapTensors;
    int hostHeapType = -1;
    size_t systemReserveBytes = 0;
    static uint64_t memTotalBytes() {
        FILE* f = fopen("/proc/meminfo","r"); if (!f) return 0;
        char line[256]; unsigned long long kb = 0; bool found = false;
        while (fgets(line,sizeof line,f)) if (sscanf(line,"MemTotal: %llu kB",&kb) == 1) { found = true; break; }
        fclose(f); return found ? (uint64_t)kb*1024 : 0;
    }
    // Drop an uploaded tensor's file pages from the page cache (the GPU copy is
    // the working copy) and enforce the system-memory reserve, failing closed
    // before the next allocation rather than after the node is exhausted.
    void releaseUploaded(const GgufTensor& t) {
        if (!servingBudget) return;
        const size_t page = (size_t)sysconf(_SC_PAGESIZE);
        const uintptr_t start = (uintptr_t)t.data & ~(uintptr_t)(page-1);
        const size_t bytes = ((uintptr_t)t.data + t.nbytes + page-1 & ~(uintptr_t)(page-1)) - start;
        madvise((void*)start, bytes, MADV_PAGEOUT);
        if (systemReserveBytes && memAvailableBytes() < systemReserveBytes)
            throw std::runtime_error("MemAvailable fell below QK_SYSTEM_RESERVE_GIB during the weight upload; load aborted (fail closed)");
    }
    // Fails CLOSED: an unreadable or unparsable /proc/meminfo reports 0 bytes
    // available, which stops the readahead rather than letting it run blind.
    static size_t memAvailableBytes() {
        FILE* f = fopen("/proc/meminfo","r"); if (!f) return 0;
        char line[256]; unsigned long long kb = 0; bool found = false;
        while (fgets(line,sizeof line,f)) if (sscanf(line,"MemAvailable: %llu kB",&kb) == 1) { found = true; break; }
        fclose(f);
        if (!found || kb > SIZE_MAX/1024) return 0;
        return (size_t)kb*1024;
    }
    // Batched prefill: every activation buffer holds batchCap rows so the
    // serial path (row 0) and the batched path share names and descriptors.
    // rowBytes keeps the single-token size for the debug taps.
    uint32_t batchCap = 0;
    std::map<std::string, size_t> rowBytes;
    static constexpr uint32_t headTile = 64;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    // QK_FLASH_PROFILE=1: per-dispatch GPU timestamps aggregated by shader.
    VkQueryPool profileQuery = VK_NULL_HANDLE;
    std::vector<std::string> profileLabels;
    std::map<std::string, std::pair<double,uint32_t>> profileMs;
    static constexpr uint32_t profileEntries = 8192;
    Buf staging;
    void* mapped = nullptr;
    static constexpr size_t stageBytes = 16 << 20;
    const uint32_t n = 2560, hc = 4, low = 320, lowPad = 384, ff = 640, experts = 512, used = 10;
    float eps = 1e-6f;
    std::string w(const std::string& suffix) const { return "blk." + std::to_string(layer) + "." + suffix; }
    std::string state(const std::string& kind) const { return "$" + kind + "." + std::to_string(layer); }
    void stamp(const char* label) {
        if (!profileQuery || profileLabels.size() >= profileEntries) return;
        vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profileQuery, (uint32_t)profileLabels.size());
        profileLabels.push_back(label);
    }
    void begin() {
        VK_CHECK(vkResetCommandBuffer(c.cb, 0));
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(c.cb, &info));
        if (profileQuery) { vkCmdResetQueryPool(c.cb, profileQuery, 0, profileEntries); profileLabels.clear(); stamp("_start"); }
    }
    void submit(bool endRecording = true) {
        if (endRecording) VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        info.commandBufferCount = 1; info.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &info, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
        if (profileQuery && endRecording && profileLabels.size() > 1) {
            std::vector<uint64_t> ts(profileLabels.size());
            VK_CHECK(vkGetQueryPoolResults(c.dev, profileQuery, 0, (uint32_t)ts.size(), ts.size()*8, ts.data(), 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const uint32_t bits = c.timestampValidBits;
            const uint64_t mask = bits >= 64 ? UINT64_MAX : ((uint64_t{1} << bits) - 1u);
            static const bool raw = [] { const char* v = getenv("QK_FLASH_PROFILE"); return v && !strcmp(v,"2"); }();
            for (size_t i = 1; i < ts.size(); ++i) {
                double ms = double((ts[i]-ts[i-1]) & mask) * c.props.limits.timestampPeriod * 1e-6;
                auto& agg = profileMs[profileLabels[i]]; agg.first += ms; agg.second++;
                if (raw) fprintf(stderr,"[flash raw] %04zu %-34s %8.1f us\n",i,profileLabels[i].c_str(),ms*1000);
            }
            profileLabels.clear();
        }
    }
    // Print and clear the aggregated per-shader GPU time.
    void printProfile(const char* what) {
        if (!profileQuery) return;
        std::vector<std::pair<std::string,std::pair<double,uint32_t>>> rows(profileMs.begin(),profileMs.end());
        std::sort(rows.begin(),rows.end(),[](const auto& a,const auto& b) { return a.second.first > b.second.first; });
        double total = 0; for (const auto& r : rows) total += r.second.first;
        fprintf(stderr,"[flash profile] %s: %.3f ms GPU%s\n",what,total,ple ? (" (PLE hits " + std::to_string(ple->hits) + ", misses " + std::to_string(ple->misses) + ")").c_str() : "");
        for (const auto& r : rows)
            fprintf(stderr,"[flash profile]   %-34s %9.3f ms %5.1f%% (%u dispatches, %.1f us each)\n",r.first.c_str(),
                    r.second.first,100*r.second.first/std::max(total,1e-9),r.second.second,1000*r.second.first/r.second.second);
        profileMs.clear();
    }
    void barrier(VkPipelineStageFlags source = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VkAccessFlags access = VK_ACCESS_SHADER_WRITE_BIT) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = access; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(c.cb, source, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }
    Buf& allocate(const std::string& name, size_t bytes) {
        auto [it, inserted] = buffers.emplace(name, Buf{});
        const bool hostHeap = hostHeapTensors.count(name) != 0;
        if (inserted) it->second = createBuf(c, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            !hostHeap, hostHeap ? hostHeapType : -1);
        // Unplanned spill (the driver falling back to host memory) is still an
        // error; planned host-heap placement must land on exactly the planned type.
        if (!hostHeap && !it->second.deviceLocal) throw std::runtime_error("native graph refuses host-memory spill: " + name);
        if (hostHeap && it->second.memType != (uint32_t)hostHeapType) throw std::runtime_error("planned host-heap placement failed: " + name);
        if (it->second.size != bytes) throw std::runtime_error("buffer shape changed: " + name);
        return it->second;
    }
    void upload(Buf& dst, const void* src, size_t bytes) {
        for (size_t offset = 0; offset < bytes; offset += stageBytes) {
            size_t chunk = std::min(stageBytes, bytes-offset);
            memcpy(mapped, (const uint8_t*)src+offset, chunk);
            begin(); VkBufferCopy copy{0, offset, chunk};
            vkCmdCopyBuffer(c.cb, staging.buf, dst.buf, 1, &copy); submit();
        }
    }
    Buf createHostCachedBuf(VkDeviceSize size) {
        Buf b; b.size = size;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(c.dev, &bci, nullptr, &b.buf));
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(c.dev, b.buf, &req);
        const VkMemoryPropertyFlags cached = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        uint32_t type = findMemType(c.mp, req.memoryTypeBits, cached);
        if (type == UINT32_MAX) type = findMemType(c.mp, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type == UINT32_MAX) throw std::runtime_error("no host-visible memory type for staging");
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size; mai.memoryTypeIndex = type;
        VK_CHECK(vkAllocateMemory(c.dev, &mai, nullptr, &b.mem));
        VK_CHECK(vkBindBufferMemory(c.dev, b.buf, b.mem, 0));
        return b;
    }
    Buf& allocateRows(const std::string& name, size_t bytesPerRow, uint32_t rows) {
        rowBytes[name] = bytesPerRow;
        return allocate(name, bytesPerRow * std::max<uint32_t>(rows, 1));
    }
    // Synchronous bounded readback through the staging buffer; only valid
    // between graph submissions.
    void download(Buf& src, size_t srcOffset, void* dst, size_t bytes) {
        for (size_t offset = 0; offset < bytes; offset += stageBytes) {
            size_t chunk = std::min(stageBytes, bytes-offset);
            begin();
            VkMemoryBarrier read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(c.cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&read,0,nullptr,0,nullptr);
            VkBufferCopy copy{srcOffset+offset, 0, chunk};
            vkCmdCopyBuffer(c.cb, src.buf, staging.buf, 1, &copy);
            read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(c.cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&read,0,nullptr,0,nullptr);
            submit();
            memcpy((uint8_t*)dst+offset, mapped, chunk);
        }
    }
    // Device-to-device copy recorded inside the graph, fenced against the
    // surrounding compute dispatches.
    void copyWithinGraph(const std::string& src, const std::string& dst, size_t bytes) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(c.cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&b,0,nullptr,0,nullptr);
        VkBufferCopy copy{0,0,bytes};
        vkCmdCopyBuffer(c.cb,buffers.at(src).buf,buffers.at(dst).buf,1,&copy);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    // fence=false skips the barrier after a dispatch whose outputs are not
    // read before the next fenced dispatch, letting independent projections
    // overlap instead of draining the GPU between them.
    template<class PC> void launch(const char* shader, std::initializer_list<std::string> refs,
                                    const PC& pc, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t spec, bool fence = true) {
        const std::string key = std::string(shader) + "/" + std::to_string(spec);
        auto it = pipes.find(key);
        if (it == pipes.end()) it = pipes.emplace(key, makePipe(c, shader, refs.size(), sizeof(pc), spec)).first;
        Pipe& pipeline = it->second;
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &pipeline.dsl;
        VkDescriptorSet set;
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &alloc, &set));
        std::vector<VkDescriptorBufferInfo> info(refs.size());
        std::vector<VkWriteDescriptorSet> writes(refs.size());
        size_t i = 0;
        for (const auto& ref : refs) {
            const auto& b = buffers.at(ref);
            info[i] = {b.buf, 0, b.size};
            writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = set; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &info[i]; ++i;
        }
        vkUpdateDescriptorSets(c.dev, writes.size(), writes.data(), 0, nullptr);
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pl, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(c.cb, pipeline.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        if (gx>c.props.limits.maxComputeWorkGroupCount[0] || gy>c.props.limits.maxComputeWorkGroupCount[1] ||
            gz>c.props.limits.maxComputeWorkGroupCount[2]) throw std::runtime_error("dispatch exceeds device limits");
        vkCmdDispatch(c.cb, gx, gy, gz);
        if (fence) { barrier(); stamp(shader); }
    }
    // Folded one-dimensional launch (kernels index wg = y*numX + x) with an
    // optional z batch of tokens.
    template<class PC> void emit(const char* shader, std::initializer_list<std::string> refs,
                                  const PC& pc, uint32_t gx, uint32_t spec = 0, uint32_t gz = 1, bool fence = true) {
        uint32_t nx=std::min(gx,c.props.limits.maxComputeWorkGroupCount[0]);
        uint32_t ny=(gx+nx-1)/nx;
        launch(shader, refs, pc, nx, ny, gz, spec, fence);
    }
    void project(const std::string& weight, const std::string& input, const std::string& output, bool fence = true) {
        const auto* tensor = g.find(weight);
        if (!tensor || tensor->nDims != 2) throw std::runtime_error("bad projection: " + weight);
        const uint32_t k = tensor->ne[0], m = tensor->ne[1];
        if (buffers.at(input).size < k*4ull || buffers.at(output).size < m*4ull)
            throw std::runtime_error("projection activation shape mismatch: " + weight);
        const char* shader;
        uint32_t units;
        switch (tensor->type) {
            case GGML_Q5_K: shader = "gemv_q5_k.spv"; units = k/32; break;
            case GGML_Q5_1: shader = "gemv_q5_1.spv"; units = k/32; break;
            case GGML_Q6_K: shader = "gemv_q6_k.spv"; units = k/32; break;
            case GGML_Q8_0: shader = "gemv_q8_0.spv"; units = k/32; break;
            default: throw std::runtime_error("unsupported projection format: " + weight);
        }
        uint32_t tpr = 256;
        while (tpr > 4 && tpr/2 >= units) tpr /= 2;
        if (tensor->type == GGML_Q5_K && c.props.deviceID == 0x1586) {
            if (m == 640 && k == 2560) tpr = 64;
            if (m == 320 && k == 10240) tpr = 128;
        }
        // Measured Q6_K geometry: the 16-element work units of gemv_q6_k are
        // counted as 32-wide above (K=2560 -> 128 lanes), and NAVI31 prefers
        // 64 lanes for the 2560-wide rows (the 248320-row head: 596 vs 714 us).
        if (tensor->type == GGML_Q6_K && k == 2560 && c.props.deviceID == 0x744c) tpr = 64;
        struct { uint32_t m,k; } pc{m,k};
        emit(shader, {weight,input,output}, pc, (m + 256/tpr - 1) / (256/tpr), tpr, 1, fence);
    }
    // Batched projection over T activation rows. Wide outputs use the tiled
    // GEMM (weights read once per 64-token tile); skinny outputs (M < 128)
    // keep the z-batched GEMV, whose rows are cheap to re-read. Strides and
    // offsets (in floats) let a projection read or write a slice of a wider
    // buffer; they are only supported on the GEMM path.
    void projectBatch(const std::string& weight, const std::string& input, const std::string& output,
                      uint32_t T, uint32_t xStride = 0, uint32_t xOff = 0, uint32_t yStride = 0, uint32_t yOff = 0, bool fence = true) {
        const auto* tensor = g.find(weight);
        if (!tensor || tensor->nDims != 2) throw std::runtime_error("bad projection: " + weight);
        const uint32_t k = tensor->ne[0], m = tensor->ne[1];
        if (!xStride) xStride = k;
        if (!yStride) yStride = m;
        if (T < 1 || buffers.at(input).size < (size_t(xOff) + size_t(T-1)*xStride + k)*4ull ||
            buffers.at(output).size < (size_t(yOff) + size_t(T-1)*yStride + m)*4ull)
            throw std::runtime_error("batched projection activation shape mismatch: " + weight);
        // Skinny outputs (M <= 64: HC inject, GDN alpha/beta). The z-batched
        // GEMV re-reads the small weights per token (1.3 ms per dispatch at 512
        // tokens), so wide batches use the split-K kernel below instead.
        // QK_FLASH_SKINNY=splitk enables the split-K kernel (compiled and
        // wired, not yet measured or parity-checked on the GPU).
        static const bool splitK = [] { const char* v = getenv("QK_FLASH_SKINNY"); return v && !strcmp(v,"splitk"); }();
        if (splitK && m <= 64 && T >= 64 && batchCap && buffers.count("$partial")) {
            // Split-K skinny projection: partials per 64-wide K split, folded
            // in order by the reduce kernel (deterministic).
            const char* shader;
            switch (tensor->type) {
                case GGML_Q5_K: shader = "qwen4_gemm_skinny_q5k.spv"; break;
                case GGML_Q6_K: shader = "qwen4_gemm_skinny_q6k.spv"; break;
                case GGML_Q8_0: shader = "qwen4_gemm_skinny_q8_0.spv"; break;
                case GGML_Q5_1: shader = "qwen4_gemm_skinny_q5_1.spv"; break;
                default: throw std::runtime_error("unsupported batched projection format: " + weight);
            }
            if (k % 64 || ((tensor->type == GGML_Q5_K || tensor->type == GGML_Q6_K) && k % 256))
                throw std::runtime_error("batched projection needs K%64 (K%256 for K-quants): " + weight);
            const uint32_t splits = k / 64;
            if (buffers.at("$partial").size < size_t(splits)*T*m*4) throw std::runtime_error("skinny partial buffer too small: " + weight);
            struct { uint32_t m,k,n,xs,xo; } pcs{m,k,T,xStride,xOff};
            launch(shader, {weight,input,"$partial"}, pcs, (T + 63) / 64, splits, 1, 0);
            struct { uint32_t m,n,splits,ys,yo; } pcr{m,T,splits,yStride,yOff};
            emit("qwen4_gemm_reduce.spv", {"$partial",output}, pcr, (m*T + 255) / 256, 0, 1, fence);
            return;
        }
        // Skinny outputs (M < 128) at narrow batches keep the packed GEMV; at
        // T >= 64 the masked scalar GEMM tile (rows >= M are dropped on store)
        // is used instead. QK_FLASH_SKINNY=gemv forces the GEMV for A/B.
        static const bool skinnyGemv = [] { const char* v = getenv("QK_FLASH_SKINNY"); return v && !strcmp(v,"gemv"); }();
        if (m < 128 && (T < 64 || skinnyGemv)) {
            if (xStride != k || xOff || yStride != m || yOff)
                throw std::runtime_error("skinny batched projection needs natural strides: " + weight);
            const char* shader;
            uint32_t units;
            switch (tensor->type) {
                case GGML_Q5_K: shader = "gemv_q5_k.spv"; units = k/32; break;
                case GGML_Q5_1: shader = "gemv_q5_1.spv"; units = k/32; break;
                case GGML_Q6_K: shader = "gemv_q6_k.spv"; units = k/32; break;
                case GGML_Q8_0: shader = "gemv_q8_0.spv"; units = k/32; break;
                default: throw std::runtime_error("unsupported projection format: " + weight);
            }
            // Pack up to 16 output rows per workgroup so each token's activation
            // row is read once per workgroup instead of once per output row
            // (4-row inject: 1 workgroup per token; 48-row alpha/beta: 3).
            uint32_t rowsPerWg = 1;
            while (rowsPerWg < 16 && rowsPerWg < m) rowsPerWg *= 2;
            uint32_t tpr = 256 / rowsPerWg;
            while (tpr > 4 && tpr/2 >= units) tpr /= 2;
            struct { uint32_t m,k; } pc{m,k};
            emit(shader, {weight,input,output}, pc, (m + 256/tpr - 1) / (256/tpr), tpr, T, fence);
            return;
        }
        // QK_FLASH_COOPMAT=1 selects the f16-input cooperative-matrix tier for
        // complete 128-row tiles (reduced precision, measured separately); the
        // scalar F32 GEMM remains the reference path and the fallback.
        const char* coopEnv = getenv("QK_FLASH_COOPMAT");  // read per call so tests can switch tiers in-process
        const bool coopWanted = coopEnv && strcmp(coopEnv,"0");
        const uint32_t mPad = (m + 127) / 128 * 128;
        const bool coop = coopWanted && c.cooperativeMatrix && c.cooperativeMatrixF32Acc && c.subgroupSize == 64 &&
                          (m % 128 == 0 || yStride >= mPad) &&
                          buffers.at(output).size >= (size_t(yOff) + size_t((T + 63) / 64 * 64 - 1)*yStride + mPad)*4ull;
        const char* shader;
        switch (tensor->type) {
            case GGML_Q5_K: shader = coop ? "qwen4_gemm_coop_q5k.spv" : "qwen4_gemm_q5k.spv"; break;
            case GGML_Q6_K: shader = coop ? "qwen4_gemm_coop_q6k.spv" : "qwen4_gemm_q6k.spv"; break;
            case GGML_Q8_0: shader = coop ? "qwen4_gemm_coop_q8_0.spv" : "qwen4_gemm_q8_0.spv"; break;
            case GGML_Q5_1: shader = coop ? "qwen4_gemm_coop_q5_1.spv" : "qwen4_gemm_q5_1.spv"; break;
            default: throw std::runtime_error("unsupported batched projection format: " + weight);
        }
        if (k % 64 || ((tensor->type == GGML_Q5_K || tensor->type == GGML_Q6_K) && k % 256))
            throw std::runtime_error("batched projection needs K%64 (K%256 for K-quants): " + weight);
        struct { uint32_t m,k,n,xs,xo,ys,yo; } pc{m,k,T,xStride,xOff,yStride,yOff};
        launch(shader, {weight,input,output}, pc, (m + 127) / 128, 1, (T + 63) / 64, 0, fence);
    }
    struct HcPC { uint32_t n,h,t,mode; float eps; };
    void tap(const std::string& name, const std::string& source) {
        if (!getenv("QK_LAYER_DUMP")) return;
        const auto target = "$tap." + std::to_string(layer) + "." + name;
        const size_t bytes = rowBytes.count(source) ? rowBytes.at(source) : buffers.at(source).size;
        allocate(target, bytes); taps[std::to_string(layer)+"."+name] = target;
        VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        memory.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(c.cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&memory,0,nullptr,0,nullptr);
        VkBufferCopy copy{0,0,bytes};
        vkCmdCopyBuffer(c.cb,buffers.at(source).buf,buffers.at(target).buf,1,&copy);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    void hcMix(const std::string& kind, const std::string& residual, bool outputHead = false) {
        const auto name = [&](const char* suffix) {
            return outputHead ? std::string("output_hc_")+suffix : w("hc_"+kind+"_"+suffix);
        };
        HcPC pc{n,hc,1,0,eps};
        emit("qwen4_hc.spv", {residual,name("norm.weight"),"$dummy","$dummy","$norm"}, pc, hc);
        tap(kind+".norm", "$norm");
        project(name("down.weight"), "$norm", "$low");
        struct { uint32_t n; float scale; } silu{low,1.0f/hc};
        emit("qwen4_silu.spv", {"$low","$silu"}, silu, (low+255)/256);
        project(name("up.weight"), "$silu", "$gate", outputHead);
        if (!outputHead) {
            project(name("inject.weight"), "$norm", "$inject");
            tap(kind+".inject", "$inject");
        }
        pc.mode = 1;
        emit("qwen4_hc.spv", {"$norm","$dummy","$gate","$dummy","$mixed"}, pc, (n+255)/256);
        tap(kind+".mixed", "$mixed");
    }
    void hcCombine(const std::string& residual, const std::string& out) {
        HcPC pc{n,hc,1,2,eps};
        emit("qwen4_hc.spv", {residual,"$dummy","$inject","$block",out}, pc, (n*hc+255)/256);
    }
    void pleForward() {
        project(w("ple_key.weight"),"$ple_emb","$ple_key",false);
        project(w("ple_value.weight"),"$ple_emb","$ple_value");
        HcPC pc{n,hc,1,0,eps};
        emit("qwen4_hc.spv",{"$ple_key",w("ple_norm_key.weight"),"$dummy","$dummy","$ple_keynorm"},pc,hc,0,1,false);
        emit("qwen4_hc.spv",{"$hidden",w("ple_norm_query.weight"),"$dummy","$dummy","$ple_query"},pc,hc);
        pc.mode = 4;
        emit("qwen4_hc.spv",{"$ple_keynorm","$dummy","$ple_query","$ple_value","$ple_gated"},pc,hc);
        pc.mode = 0;
        emit("qwen4_hc.spv",{"$ple_gated",w("ple_norm_conv.weight"),"$dummy","$dummy","$ple_normalized"},pc,hc);
        struct { uint32_t channels,position; } conv{n*hc,0};
        emit("qwen4_ple_conv.spv",{"$ple_normalized",w("ple_conv1d.weight"),"$ple_history","$ple_gated","$hidden","$position"},conv,(n*hc+255)/256);
        tap("ple.output","$hidden");
    }
    void fullAttention() {
        project(w("attn_q.weight"),"$mixed","$fa_qfull",false);
        project(w("attn_k.weight"),"$mixed","$fa_k",false);
        project(w("attn_v.weight"),"$mixed","$fa_v");
        struct { uint32_t pos,tmax,dh,nrot,hq,hkv; float eps,base; } pc{0,capacity,256,64,24,2,eps,1e7f};
        emit("fa_prep_srv.spv",{"$fa_qfull","$fa_k","$fa_v",w("attn_q_norm.weight"),w("attn_k_norm.weight"),
             "$fa_qhat",state("kcache"),state("vcache"),"$rope","$position"},pc,28);
        emit("fa_attn_srv.spv",{"$fa_qhat",state("kcache"),state("vcache"),"$fa_qfull","$att","$position"},pc,24);
        tap("attn_gated","$att");
        project(w("attn_output.weight"),"$att","$block");
    }
    // ---- batched (prefill) twins of the modules above ----
    void hcMixBatch(const std::string& kind, const std::string& residual, uint32_t T, bool outputHead = false) {
        const auto name = [&](const char* suffix) {
            return outputHead ? std::string("output_hc_")+suffix : w("hc_"+kind+"_"+suffix);
        };
        HcPC pc{n,hc,T,0,eps};
        emit("qwen4_hc.spv", {residual,name("norm.weight"),"$dummy","$dummy","$norm"}, pc, hc, 0, T);
        // The low-rank rows are stored with a 384-float stride so the 320-row
        // down projection fills complete 128-row tiles; silu covers the padding
        // (unused) and the up projection reads only the first 320 columns.
        projectBatch(name("down.weight"), "$norm", "$low", T, 0, 0, lowPad, 0);
        struct { uint32_t n; float scale; } silu{lowPad*T,1.0f/hc};
        emit("qwen4_silu.spv", {"$low","$silu"}, silu, (lowPad*T+255)/256);
        projectBatch(name("up.weight"), "$silu", "$gate", T, lowPad, 0, 0, 0, outputHead);
        if (!outputHead) projectBatch(name("inject.weight"), "$norm", "$inject", T);
        pc.mode = 1;
        emit("qwen4_hc.spv", {"$norm","$dummy","$gate","$dummy","$mixed"}, pc, (n+255)/256, 0, T);
    }
    void hcCombineBatch(const std::string& residual, const std::string& out, uint32_t T) {
        HcPC pc{n,hc,T,2,eps};
        emit("qwen4_hc.spv", {residual,"$dummy","$inject","$block",out}, pc, (n*hc+255)/256, 0, T);
    }
    void pleForwardBatch(uint32_t T, uint32_t base) {
        projectBatch(w("ple_key.weight"),"$ple_emb","$ple_key",T,0,0,0,0,false);
        projectBatch(w("ple_value.weight"),"$ple_emb","$ple_value",T);
        HcPC pc{n,hc,T,0,eps};
        emit("qwen4_hc.spv",{"$ple_key",w("ple_norm_key.weight"),"$dummy","$dummy","$ple_keynorm"},pc,hc,0,T,false);
        emit("qwen4_hc.spv",{"$hidden",w("ple_norm_query.weight"),"$dummy","$dummy","$ple_query"},pc,hc,0,T);
        pc.mode = 4;
        emit("qwen4_hc.spv",{"$ple_keynorm","$dummy","$ple_query","$ple_value","$ple_gated"},pc,hc,0,T);
        pc.mode = 0;
        emit("qwen4_hc.spv",{"$ple_gated",w("ple_norm_conv.weight"),"$dummy","$dummy","$ple_normalized"},pc,hc,0,T);
        struct { uint32_t channels,base,T; } conv{n*hc,base,T};
        emit("qwen4_ple_conv_batch.spv",{"$ple_normalized",w("ple_conv1d.weight"),"$ple_history","$ple_gated","$hidden"},conv,(n*hc+255)/256);
    }
    void fullAttentionBatch(uint32_t T, uint32_t base) {
        projectBatch(w("attn_q.weight"),"$mixed","$fa_qfull",T,0,0,0,0,false);
        projectBatch(w("attn_k.weight"),"$mixed","$fa_k",T,0,0,0,0,false);
        projectBatch(w("attn_v.weight"),"$mixed","$fa_v",T);
        struct { uint32_t tmax,dh,nrot,hq,hkv; float eps,base_; uint32_t base,Tn,qbase; } pc{capacity,256,64,24,2,eps,1e7f,base,T,0};
        emit("fa_prep_batch.spv",{"$fa_qfull","$fa_k","$fa_v",w("attn_q_norm.weight"),w("attn_k_norm.weight"),
             "$fa_qhat",state("kcache"),state("vcache"),"$rope"},pc,28,0,T);
        // Query tiles bound the keys*queries work of one dispatch (QB=16
        // query blocks in fa_attn_batch); every query only reads keys <= its
        // own position, so tiling is exact.
        static const uint64_t attnBudget = [] {
            const char* v = getenv("QK_ATTN_BUDGET"); long x = v ? atol(v) : 2097152;
            return (uint64_t)(x < 4096 ? 4096 : x); }();
        uint32_t qt = (uint32_t)std::max<uint64_t>(1, attnBudget / (uint64_t)(base + T));
        qt = std::min(qt, T); qt = std::max(16u, qt - qt % 16u);
        const char* coopEnv = getenv("QK_FLASH_COOPMAT");
        const bool coopAttn = coopEnv && strcmp(coopEnv,"0") && c.cooperativeMatrix && c.cooperativeMatrixF32Acc && c.subgroupSize == 64;
        for (uint32_t qo = 0; qo < T; qo += qt) {
            uint32_t tile = std::min(qt, T - qo);
            pc.qbase = qo;
            emit(coopAttn ? "fa_attn_batch_coopmat.spv" : "fa_attn_batch.spv",{"$fa_qhat",state("kcache"),state("vcache"),"$fa_qfull","$att"},pc,24,0,(tile+15)/16);
        }
        projectBatch(w("attn_output.weight"),"$att","$block",T);
    }
    void gdnBatch(uint32_t T) {
        projectBatch(w("attn_qkv.weight"),"$mixed","$qkv",T,0,0,0,0,false);
        projectBatch(w("attn_gate.weight"),"$mixed","$z",T,0,0,0,0,false);
        projectBatch(w("ssm_alpha.weight"),"$mixed","$alpha",T,0,0,0,0,false);
        projectBatch(w("ssm_beta.weight"),"$mixed","$beta",T);
        struct { uint32_t heads,T; } params{48,T};
        emit("qwen4_gdn_params.spv",{"$alpha","$beta",w("ssm_dt.bias"),w("ssm_a"),"$gb"},params,(48*T+63)/64);
        // The conv window before this chunk seeds the batched conv, which
        // then rewrites the window for the tokens after the chunk.
        copyWithinGraph(state("convstate"),"$carry",10240*3*4);
        struct { uint32_t channels,ds,qk; float eps; uint32_t Tn; } conv{10240,128,4096,eps,T};
        emit("qwen4_gdn_conv_batch.spv",{"$carry","$qkv",w("ssm_conv1d.weight"),"$conv",state("convstate")},conv,80,0,T);
        struct { uint32_t ds,hk,hv,Tn; float eps; } step{128,16,48,T,eps};
        emit("qwen4_gdn_step_batch.spv",{"$conv","$gb",state("state"),w("ssm_norm.weight"),"$z","$att"},step,48);
        projectBatch(w("ssm_out.weight"),"$att","$block",T);
    }
    void moeBatch(uint32_t T) {
        struct { uint32_t n,ff,experts,used; } moe{n,ff,experts,used};
        const bool downQ51 = g.find(w("ffn_down_exps.weight"))->type==GGML_Q5_1;
        if (T % 32 == 0) emit("moe_logits_gemm.spv",{w("ffn_gate_inp.weight"),"$mixed","$logits"},moe,experts/64,0,T/32);
        else emit("moe_logits.spv",{w("ffn_gate_inp.weight"),"$mixed","$logits"},moe,experts,0,T);
        emit("moe_select_256.spv",{"$logits",w("ffn_gate_inp_shexp.weight"),"$mixed","$sel"},moe,1,0,T);
        struct { uint32_t experts,used,tokens; } pairs{experts,used,T};
        emit("moe_group_pairs.spv",{"$sel","$offsets","$pairs"},pairs,1);
        // QK_MOE_GROUPED=pairs keeps the per-pair reduction kernels for A/B
        // checks; the default tiles each expert's tokens in groups of 16.
        static const bool pairKernels = [] { const char* v = getenv("QK_MOE_GROUPED"); return v && !strcmp(v,"pairs"); }();
        const char* coopEnv = getenv("QK_FLASH_COOPMAT");
        const char* coopMoeEnv = getenv("QK_FLASH_COOPMAT_MOE");  // =0 keeps the scalar expert tiles under the coopmat tier
        const bool coop = coopEnv && strcmp(coopEnv,"0") && !(coopMoeEnv && !strcmp(coopMoeEnv,"0")) &&
                          c.cooperativeMatrix && c.cooperativeMatrixF32Acc && c.subgroupSize == 64;
        if (pairKernels) launch("qwen4_moe_gateup_grouped.spv",{w("ffn_gate_exps.weight"),w("ffn_up_exps.weight"),"$mixed","$offsets","$pairs","$ffh"},moe,ff,experts,1,0);
        else launch(coop?"qwen4_moe_gateup_coop.spv":"qwen4_moe_gateup_tiled.spv",{w("ffn_gate_exps.weight"),w("ffn_up_exps.weight"),"$mixed","$offsets","$pairs","$ffh"},moe,ff/128,experts,1,0);
        projectBatch(w("ffn_gate_shexp.weight"),"$mixed","$sg",T,0,0,0,0,false);
        projectBatch(w("ffn_up_shexp.weight"),"$mixed","$su",T);
        struct { uint32_t n; } sm{ff*T};
        emit("qwen4_silu_mul.spv",{"$sg","$su","$sh"},sm,(ff*T+255)/256);
        projectBatch(w("ffn_down_shexp.weight"),"$sh","$shared_out",T);
        if (pairKernels) launch(downQ51?"qwen4_moe_down_grouped_q51.spv":"qwen4_moe_down_grouped_q8.spv",
               {w("ffn_down_exps.weight"),"$ffh","$sel","$offsets","$pairs","$routed"},moe,n,experts,1,0);
        else launch(coop ? (downQ51?"qwen4_moe_down_coop_q51.spv":"qwen4_moe_down_coop_q8.spv")
                         : (downQ51?"qwen4_moe_down_tiled_q51.spv":"qwen4_moe_down_tiled_q8.spv"),
               {w("ffn_down_exps.weight"),"$ffh","$sel","$offsets","$pairs","$routed"},moe,n/128,experts,1,0);
        struct { uint32_t n,used; } comb{n,used};
        emit("qwen4_moe_combine.spv",{"$routed","$shared_out","$sel","$block"},comb,(n+255)/256,0,T);
    }
    void headBatch(uint32_t T) {
        hcMixBatch("head","$hidden",T,true);
        for (uint32_t g0 = 0; g0 < T; g0 += headTile) {
            const uint32_t rows = std::min(headTile, T-g0);
            projectBatch("output.weight","$mixed","$output_logits",rows,n,g0*n,0,0);
            struct { uint32_t vocab,T,idOff; } am{(uint32_t)logits.size(),rows,g0};
            emit("qwen4_argmax.spv",{"$output_logits","$ids"},am,rows);
        }
    }
    size_t batchBytes(uint32_t rows) const {
        size_t per = size_t(n)*hc*4*6 + size_t(n)*4*3 + (lowPad*2 + hc + 48*2 + 96)*4 + 6144ull*4*2;
        if (lastLayer >= 3) per += (12288ull+512+512+6144)*4;
        if (firstLayer==0 && lastLayer>=1) per += size_t(n)*hc*4*5 + size_t(n)*4*2;
        per += experts*4 + 160 + size_t(used+1)*ff*4;
        per += size_t(used)*n*4 + size_t(ff)*4*3 + size_t(n)*4 + 4 + size_t(used)*4;
        return per*rows + size_t(10240/64)*rows*64*4 + (experts+1)*4 + 10240ull*3*4 + (withHead ? size_t(headTile)*248320*4 : 0);
    }
    void expect(const std::string& name, std::initializer_list<uint64_t> shape, int type = -1) {
        const auto* tensor = g.find(name);
        if (!tensor || !tensor->data || !tensor->nbytes || tensor->nDims != shape.size() ||
            (type >= 0 && tensor->type != uint32_t(type)) ||
            !std::equal(shape.begin(),shape.end(),tensor->ne))
            throw std::runtime_error("unsupported prefix tensor shape/type: " + name);
    }
    void validateLayer() {
        for (const auto& kind : {"attn","ffn"}) {
            const auto prefix = std::string("hc_")+kind;
            expect(w(prefix+"_norm.weight"),{n*hc},GGML_F32);
            expect(w(prefix+"_down.weight"),{n*hc,low});
            expect(w(prefix+"_up.weight"),{low,n*hc});
            expect(w(prefix+"_inject.weight"),{n*hc,hc});
        }
        if (layer % 4 != 3) {
        expect(w("attn_qkv.weight"),{n,10240}); expect(w("attn_gate.weight"),{n,6144});
        expect(w("ssm_alpha.weight"),{n,48}); expect(w("ssm_beta.weight"),{n,48});
        expect(w("ssm_a"),{48},GGML_F32); expect(w("ssm_dt.bias"),{48},GGML_F32);
        expect(w("ssm_conv1d.weight"),{4,10240},GGML_F32);
        expect(w("ssm_norm.weight"),{128},GGML_F32); expect(w("ssm_out.weight"),{6144,n});
        } else {
            expect(w("attn_q.weight"),{n,12288}); expect(w("attn_k.weight"),{n,512}); expect(w("attn_v.weight"),{n,512});
            expect(w("attn_q_norm.weight"),{256},GGML_F32); expect(w("attn_k_norm.weight"),{256},GGML_F32);
            expect(w("attn_output.weight"),{6144,n});
        }
        expect(w("ffn_gate_inp.weight"),{n,experts},GGML_F32);
        expect(w("ffn_gate_inp_shexp.weight"),{n},GGML_F32);
        for (const auto& kind : {"gate","up"}) {
            expect(w(std::string("ffn_")+kind+"_exps.weight"),{n,ff,experts},GGML_Q5_K);
            expect(w(std::string("ffn_")+kind+"_shexp.weight"),{n,ff},GGML_Q5_K);
        }
        expect(w("ffn_down_exps.weight"),{ff,n,experts});
        expect(w("ffn_down_shexp.weight"),{ff,n});
        for (const auto& suffix : {"ffn_down_exps.weight","ffn_down_shexp.weight"}) {
            auto type=g.find(w(suffix))->type;
            if (type!=GGML_Q8_0 && type!=GGML_Q5_1) throw std::runtime_error("unsupported expert down type");
        }
        if (layer == 1) {
            expect(w("ple_key.weight"),{n,n*hc}); expect(w("ple_value.weight"),{n,n});
            for (const auto& name : {"key","query","conv"}) expect(w(std::string("ple_norm_")+name+".weight"),{n*hc},GGML_F32);
            expect(w("ple_conv1d.weight"),{4,n*hc},GGML_F16);
        }
    }
public:
    Qwen4Graph(VkCtx& context, Gguf& model, uint32_t endLayer, uint32_t ctx = 32) : c(context), g(model), lastLayer(endLayer), capacity(ctx) {}
    Qwen4Graph(VkCtx& context, Gguf& model, uint32_t first, uint32_t end, uint32_t ctx, uint64_t budget, bool head)
        : c(context), g(model), firstLayer(first), lastLayer(end-1), capacity(ctx), weightLimit(budget), withHead(head), servingBudget(true) {}
    uint32_t currentPosition() const { return position; }
    uint32_t batchCapacity() const { return batchCap; }
    const std::vector<float>& lastLogits() const { return logits; }
    ~Qwen4Graph() {
        pleStop = true;
        if (pleThread.joinable()) pleThread.join();
        // Return the lookup tables' page cache when the stage closes so the
        // next model load starts with the memory this stage held.
        for (const auto& r : pleKeep) madvise((void*)r.start, r.bytes, MADV_PAGEOUT);
        if (profileQuery) vkDestroyQueryPool(c.dev, profileQuery, nullptr);
        if (pool) vkDestroyDescriptorPool(c.dev, pool, nullptr);
        for (auto& [_, pipe] : pipes) destroyPipe(c, pipe);
        for (auto& [_, buf] : buffers) destroyBuf(c, buf);
        if (mapped) vkUnmapMemory(c.dev, staging.mem);
        destroyBuf(c, staging);
    }
    void open() {
        if (firstLayer>lastLayer || lastLayer>=48 || !capacity || capacity>65536 ||
            (firstLayer==1) || (withHead && lastLayer!=47))
            throw std::runtime_error("unsupported native layer range/context; PLE must stay in the first stage");
        if (g.kvStr("general.architecture", "") != "qwen4exp" ||
            g.kvInt("qwen4exp.block_count",0) != 48 || g.kvInt("qwen4exp.embedding_length",0) != n ||
            g.kvInt("qwen4exp.expert_feed_forward_length",0) != ff ||
            g.kvInt("qwen4exp.hyper_connection.count",0) != hc ||
            g.kvInt("qwen4exp.expert_count",0) != experts ||
            g.kvInt("qwen4exp.expert_used_count",0) != used)
            throw std::runtime_error("prefix harness needs the installed Flash Next shape");
        eps = g.kvFloat("qwen4exp.attention.layer_norm_rms_epsilon",1e-6);
        if (!std::isfinite(eps) || eps <= 0) throw std::runtime_error("invalid RMS epsilon");
        const auto& ratios = g.kvInts("qwen4exp.attention.compress_ratios");
        if (ratios.size() != g.kvInt("qwen4exp.block_count",0) ||
            std::any_of(ratios.begin(),ratios.end(),[](auto r) { return r != 0; }))
            throw std::runtime_error("compressed QSA attention is not supported");
        if (lastLayer >= 3 && (g.kvInt("qwen4exp.rope.dimension_count",0) != 64 ||
            g.kvFloat("qwen4exp.rope.freq_base",0) != 1e7 ||
            g.kvFloat("qwen4exp.attention.scale",0) != 0))
            throw std::runtime_error("unsupported attention/RoPE configuration");
        for (layer=firstLayer; layer<=lastLayer; ++layer) validateLayer();
        if (withHead) {
            expect("output_hc_norm.weight",{n*hc},GGML_F32);
            expect("output_hc_down.weight",{n*hc,low}); expect("output_hc_up.weight",{low,n*hc});
            expect("output.weight",{n,248320},GGML_Q6_K);
            logits.resize(248320);
        }
        size_t totalWeightBytes = 0;
        for (layer=firstLayer; layer<=lastLayer; ++layer) for (const auto& [name,tensor] : g.tensors())
            if (name.compare(0,w("").size(),w("")) == 0) {
                if (tensor.nbytes > weightLimit-totalWeightBytes)
                    throw std::runtime_error("native graph weight budget exceeded");
                totalWeightBytes += tensor.nbytes;
            }
        if (withHead) for (const auto& name : {"output_hc_norm.weight","output_hc_down.weight","output_hc_up.weight","output.weight"}) {
            auto bytes=g.find(name)->nbytes;
            if (bytes>weightLimit-totalWeightBytes) throw std::runtime_error("native head weight budget exceeded");
            totalWeightBytes+=bytes;
        }
        // Fail fast on bad prefetch settings before any weight upload.
        if (const char* v = getenv("QK_PLE_PREFETCH"); v && strcmp(v,"0") && strcmp(v,"1"))
            throw std::runtime_error("QK_PLE_PREFETCH must be 0 or 1");
        if (const char* floorEnv = getenv("QK_PLE_PREFETCH_FLOOR_GIB")) {
            char* end = nullptr; double f = strtod(floorEnv,&end);
            if (end == floorEnv || *end || !std::isfinite(f) || f < 1.0 || f > 1024.0)
                throw std::runtime_error("QK_PLE_PREFETCH_FLOOR_GIB must be a number in 1..1024");
        }
        // Batched prefill rows: QK_FLASH_BATCH (default 512, 0 disables), rounded
        // up to whole 64-token tiles; halved while the device budget is short.
        if (servingBudget) {
            const char* env=getenv("QK_FLASH_BATCH");
            long requested=env ? atol(env) : 512;
            if (!env || requested<0 || requested>512) { if (env) throw std::runtime_error("QK_FLASH_BATCH must be 0..512"); }
            batchCap=(uint32_t)requested;
            if (batchCap) batchCap=(batchCap+63)/64*64;
        } else batchCap=std::min(512u,(capacity+63)/64*64);
        if (servingBudget) {
            if (!c.memoryBudget) throw std::runtime_error("native serving requires Vulkan memory-budget reporting");
            auto gib = [&](const char* env, double def, double lo, double hi) {
                const char* v = getenv(env); if (!v) return def;
                char* end = nullptr; double x = strtod(v,&end);
                if (end == v || *end || !std::isfinite(x) || x < lo || x > hi)
                    throw std::runtime_error(std::string(env) + " must be a number in the accepted range");
                return x;
            };
            // Device-local needs besides weights: staging/scratch/descriptor
            // allowance, KV and recurrent state; batch rows are added below.
            uint64_t fixed=(256ull<<20);
            for (layer=firstLayer; layer<=lastLayer; ++layer)
                fixed+=layer%4==3 ? 2ull*2*capacity*256*4 : (48ull*128*128+10240*3)*4;
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
            props.pNext=&budget; vkGetPhysicalDeviceMemoryProperties2(c.phys,&props);
            auto headroom = [&](uint32_t heap) {
                return budget.heapBudget[heap] > budget.heapUsage[heap] ? budget.heapBudget[heap]-budget.heapUsage[heap] : 0ull; };
            const uint32_t deviceType=findMemType(c.mp,UINT32_MAX,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (deviceType==UINT32_MAX) throw std::runtime_error("no device-local memory type");
            const uint32_t deviceHeap=c.mp.memoryTypes[deviceType].heapIndex;
            // Per-heap margin for the driver's own and the compositor's
            // allocations (QK_HEAP_MARGIN_GIB, default 1); the system reserve is
            // a MemAvailable floor enforced during upload (QK_SYSTEM_RESERVE_GIB,
            // default 12, 0 disables). Neither is a guarantee of driver
            // allocation success; they bound the plan and fail it closed.
            const uint64_t margin=(uint64_t)(gib("QK_HEAP_MARGIN_GIB",1.0,0.0,64.0)*double(1ull<<30));
            systemReserveBytes=(size_t)(gib("QK_SYSTEM_RESERVE_GIB",12.0,0.0,1024.0)*double(1ull<<30));
            uint64_t deviceFree=headroom(deviceHeap); deviceFree=deviceFree>margin ? deviceFree-margin : 0;
            // Host-heap spill (QK_HOST_HEAP=1, default) uses the first
            // host-visible, host-coherent, non-device-local type: measured on
            // this part at 2-7% more GPU read time than device-local, so only
            // the routed expert tensors (10 of 512 read per token) spill, in
            // reverse layer order, and only as much as needed.
            // QK_HOST_HEAP=0|1; unset means on for integrated (UMA) devices
            // only, so discrete cards keep their device-local-only default.
            const bool hostHeapAllowed = [&] {
                const char* v=getenv("QK_HOST_HEAP");
                if (!v) return c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
                if (!strcmp(v,"0")) return false;
                if (!strcmp(v,"1")) return true;
                throw std::runtime_error("QK_HOST_HEAP must be 0 or 1");
            }();
            uint64_t hostFree=0; uint32_t hostHeap=UINT32_MAX;
            for (uint32_t t=0; t<c.mp.memoryTypeCount && hostHeapAllowed; ++t) {
                VkMemoryPropertyFlags f=c.mp.memoryTypes[t].propertyFlags;
                if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) || !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
                    !(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) || c.mp.memoryTypes[t].heapIndex==deviceHeap) continue;
                hostHeapType=(int)t; hostHeap=c.mp.memoryTypes[t].heapIndex; break;
            }
            if (hostHeapType>=0) { hostFree=headroom(hostHeap); hostFree=hostFree>margin ? hostFree-margin : 0; }
            uint64_t deviceWeights=0, hostWeights=0;
            auto plan = [&]() {
                hostHeapTensors.clear(); deviceWeights=totalWeightBytes; hostWeights=0;
                const uint64_t nonWeight=fixed+batchBytes(batchCap);
                const uint64_t deviceForWeights=deviceFree>nonWeight ? deviceFree-nonWeight : 0;
                if (deviceWeights<=deviceForWeights) return true;
                uint64_t needSpill=deviceWeights-deviceForWeights;
                for (int l=(int)lastLayer; l>=(int)firstLayer && needSpill; --l)
                    for (const char* kind : {"ffn_down_exps.weight","ffn_up_exps.weight","ffn_gate_exps.weight"}) {
                        if (!needSpill) break;
                        const std::string name="blk."+std::to_string(l)+"."+kind;
                        const auto* t=g.find(name); if (!t) continue;
                        hostHeapTensors.insert(name); hostWeights+=t->nbytes; deviceWeights-=t->nbytes;
                        needSpill = needSpill>t->nbytes ? needSpill-t->nbytes : 0;
                    }
                return needSpill==0 && hostWeights<=hostFree;
            };
            while (!plan() && batchCap) {
                printf("native placement does not fit with %u batch rows (%.3f GiB); halving\n",batchCap,batchBytes(batchCap)/double(1ull<<30));
                batchCap = batchCap>64 ? batchCap/2/64*64 : 0;
            }
            const uint64_t deviceNeed=deviceWeights+fixed+batchBytes(batchCap);
            const uint64_t total=deviceNeed+hostWeights;
            const uint64_t memTotal=memTotalBytes(), memAvail=memAvailableBytes();
            printf("native placement: device-local %.3f GiB (weights %.3f + state/scratch %.3f + batch %.3f, headroom %.3f), host heap %.3f GiB in %zu expert tensors (headroom %.3f, %s); system MemTotal %.1f GiB, MemAvailable %.1f GiB, reserve %.1f GiB\n",
                   deviceNeed/double(1ull<<30),deviceWeights/double(1ull<<30),fixed/double(1ull<<30),batchBytes(batchCap)/double(1ull<<30),deviceFree/double(1ull<<30),
                   hostWeights/double(1ull<<30),hostHeapTensors.size(),hostFree/double(1ull<<30),
                   hostHeapType>=0 ? memTypeDesc(c,(uint32_t)hostHeapType).c_str() : "spill disabled",
                   memTotal/double(1ull<<30),memAvail/double(1ull<<30),systemReserveBytes/double(1ull<<30));
            if (!plan()) throw std::runtime_error("insufficient device budget: the model does not fit the device-local heap plus the host heap within their budgets");
            // Physical-memory gate, fail closed: /proc/meminfo must be readable,
            // the plan plus the reserve must fit MemTotal, and MemAvailable must
            // already clear the reserve before the first allocation. (Pages
            // retained in the driver's pool are reused by the allocations and
            // are not counted here; the per-upload check catches the rest.)
            if (!memTotal || !memAvail) throw std::runtime_error("cannot read MemTotal/MemAvailable; refusing to plan GPU allocations blind");
            if (total+systemReserveBytes>memTotal)
                throw std::runtime_error("insufficient system memory: planned GPU allocations plus the reserve exceed MemTotal");
            if (memAvail<systemReserveBytes)
                throw std::runtime_error("MemAvailable is already below QK_SYSTEM_RESERVE_GIB; refusing to load");
            // Table warming (QK_PLE_PREFETCH=1) adds the lookup tables to the
            // resident footprint; reject it up front when that cannot fit.
            if (const char* v=getenv("QK_PLE_PREFETCH"); v && !strcmp(v,"1")) {
                uint64_t tables=0;
                for (const auto& [name,tensor] : g.tensors())
                    if (name=="token_embd.weight" || (firstLayer==0 && lastLayer>=1 && name=="per_layer_token_embd.weight")) tables+=tensor.nbytes;
                if (total+tables+systemReserveBytes>memTotal)
                    throw std::runtime_error("QK_PLE_PREFETCH=1 rejected: GPU allocations plus the lookup tables plus the reserve exceed MemTotal; run with QK_PLE_PREFETCH=0");
            }
        }
        printf("prefix weights: %.3f GiB; staging: 16 MiB; PLE table stays mapped\n",totalWeightBytes/double(1ull<<30));
        // Readback goes through this buffer: a host-cached type keeps the CPU
        // copies fast (uncached write-combined memory reads at ~0.3 GB/s).
        staging = createHostCachedBuf(stageBytes);
        VK_CHECK(vkMapMemory(c.dev, staging.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
        for (layer=firstLayer; layer<=lastLayer; ++layer) for (const auto& [name,tensor] : g.tensors()) if (name.compare(0,w("").size(),w("")) == 0) {
            if (!tensor.nbytes) throw std::runtime_error("unknown tensor layout: " + name);
            auto& b = allocate(name,tensor.nbytes); upload(b,tensor.data,tensor.nbytes); releaseUploaded(tensor);
        }
        if (withHead) {
            for (const auto& name : {"output_hc_norm.weight","output_hc_down.weight","output_hc_up.weight","output.weight"}) {
                const auto* tensor=g.find(name); auto& b=allocate(name,tensor->nbytes); upload(b,tensor->data,tensor->nbytes); releaseUploaded(*tensor);
            }
            allocateRows("$output_logits",logits.size()*4,batchCap ? headTile : 1);
        }
        const uint32_t rows = std::max(batchCap,1u);
        for (const auto& name : {"$hidden","$residual","$norm","$gate","$qkv","$conv"}) allocateRows(name,n*hc*4,rows);
        for (const auto& name : {"$mixed","$block","$ffout"}) allocateRows(name,n*4,rows);
        allocateRows("$low",lowPad*4,rows); allocateRows("$silu",lowPad*4,rows); allocateRows("$inject",hc*4,rows); allocate("$dummy",4);
        allocateRows("$alpha",48*4,rows); allocateRows("$beta",48*4,rows); allocateRows("$gb",96*4,rows);
        allocateRows("$z",6144*4,rows); allocateRows("$att",6144*4,rows);
        if (batchCap) {
            allocate("$partial",size_t(10240/64)*rows*64*4);  // split-K partials: 160 splits x rows x 64 outputs
            allocate("$carry",10240*3*4); allocate("$offsets",(experts+1)*4); allocate("$pairs",size_t(rows)*used*4);
            allocateRows("$routed",size_t(used)*n*4,rows); allocateRows("$shared_out",n*4,rows); allocate("$ids",size_t(rows)*4);
            for (const auto& name : {"$sg","$su","$sh"}) allocateRows(name,ff*4,rows);
        }
        for (layer=firstLayer; layer<=lastLayer; ++layer) {
            if (layer % 4 != 3) {
                allocate(state("convstate"),10240*3*4); allocate(state("state"),48*128*128*4);
            } else {
                allocate(state("kcache"),2ull*capacity*256*4); allocate(state("vcache"),2ull*capacity*256*4);
            }
        }
        if (lastLayer >= 3) {
            allocateRows("$fa_qfull",12288*4,rows); allocateRows("$fa_k",512*4,rows); allocateRows("$fa_v",512*4,rows); allocateRows("$fa_qhat",6144*4,rows);
            auto& rope = allocate("$rope",capacity*64*4);
            std::vector<float> values(capacity*64);
            for (uint32_t pos=0; pos<capacity; ++pos) for (uint32_t j=0; j<32; ++j) {
                double theta = pos*std::pow(1e7,-double(2*j)/64);
                values[2*(pos*32+j)] = std::cos(theta); values[2*(pos*32+j)+1] = std::sin(theta);
            }
            upload(rope,values.data(),values.size()*4);
        }
        if (firstLayer==0 && lastLayer>=1) {
            const auto* table = g.find("per_layer_token_embd.weight");
            if (!table) throw std::runtime_error("missing PLE table");
            ple = std::make_unique<Qwen4PleLookup>(*table,Qwen4PleConfig::read(g));
            if (ple->config().width * ple->config().offsets.size() != n ||
                ple->config().ngram != 3 || g.kvInt("qwen4exp.ple.conv_kernel",0) != 4)
                throw std::runtime_error("unsupported PLE dimensions");
            for (const auto& name : {"$ple_key","$ple_keynorm","$ple_query","$ple_gated","$ple_normalized"}) allocateRows(name,n*hc*4,rows);
            allocateRows("$ple_emb",n*4,rows); allocateRows("$ple_value",n*4,rows); allocate("$ple_history",9*n*hc*4);
        }
        // Page-cache plan: the uploaded weights are never read from the host
        // again, so their file pages are reclaimed (MADV_PAGEOUT) to make room
        // for the tables the CPU reads per token: the 35.763 GiB PLE table and
        // the token embedding, which are then read ahead and touched so every
        // per-token row lookup is a memory read instead of an NVMe page fault.
        // The service memory limit must leave room for those tables' page cache.
        // QK_PLE_PREFETCH_FLOOR_GIB (default 16) stops the readahead while
        // MemAvailable is below that floor: the GPU driver allocates system
        // memory for the Halo stage and fails outright (not by reclaiming
        // cache) when the node runs out, which took down the display session
        // once on 2026-09-11 during concurrent GPU testing.
        // Opt-in only (QK_PLE_PREFETCH=1): the plan above already rejected it
        // when the tables cannot fit beside the GPU allocations.
        if (const char* v = getenv("QK_PLE_PREFETCH"); v && !strcmp(v,"1")) {
            std::vector<PageRange> release, keep;
            const size_t page = (size_t)sysconf(_SC_PAGESIZE);
            auto range = [&](const GgufTensor& t) {
                uintptr_t start = (uintptr_t)t.data & ~(uintptr_t)(page-1);
                return PageRange{start, ((uintptr_t)t.data + t.nbytes + page-1 & ~(uintptr_t)(page-1)) - start};
            };
            for (const auto& [name,tensor] : g.tensors()) {
                if (!tensor.data || !tensor.nbytes) continue;
                if (name == "token_embd.weight" || (ple && name == "per_layer_token_embd.weight")) keep.push_back(range(tensor));
                else if (buffers.count(name)) release.push_back(range(tensor));
            }
            pleKeep = keep;
            // Floor in GiB: default 16, accepted range 1..1024; anything else
            // (empty, non-numeric, out of range) is an error, not a silent 0.
            double floorGib = 16.0;
            if (const char* floorEnv = getenv("QK_PLE_PREFETCH_FLOOR_GIB")) {
                char* end = nullptr; floorGib = strtod(floorEnv,&end);
                if (end == floorEnv || *end || !std::isfinite(floorGib) || floorGib < 1.0 || floorGib > 1024.0)
                    throw std::runtime_error("QK_PLE_PREFETCH_FLOOR_GIB must be a number in 1..1024");
            }
            const size_t floorBytes = (size_t)(floorGib * double(1ull<<30));
            pleThread = std::thread([this,release,keep,page,floorBytes] {
                const auto began = std::chrono::steady_clock::now();
                size_t released = 0, kept = 0;
                bool floored = false;
                for (const auto& r : release) { if (pleStop) return; madvise((void*)r.start, r.bytes, MADV_PAGEOUT); released += r.bytes; }
                // The kernel clamps each WILLNEED call to the device readahead
                // window, so issue small chunks, then touch every page; check
                // the free-memory floor every 64 MiB.
                // `kept` counts only bytes actually touched (one page at a
                // time), so the resident figure is exact for the pages this
                // thread referenced; the floor is checked every 64 MiB of both
                // the readahead and the touch passes. Readahead alone does not
                // count as resident.
                const size_t chunk = 512u<<10, check = 64u<<20;
                volatile uint8_t sink = 0;
                for (const auto& r : keep) {
                    for (size_t off = 0; off < r.bytes && !pleStop && !floored; off += chunk) {
                        if (off % check == 0 && memAvailableBytes() < floorBytes) { floored = true; break; }
                        madvise((void*)(r.start+off), std::min(chunk,r.bytes-off), MADV_WILLNEED);
                    }
                    for (size_t off = 0; off < r.bytes && !pleStop && !floored; off += page) {
                        if (off % check == 0 && memAvailableBytes() < floorBytes) { floored = true; break; }
                        sink += *(const volatile uint8_t*)(r.start+off);
                        kept += std::min(page, r.bytes-off);
                    }
                }
                if (!pleStop) fprintf(stderr,"[flash] page cache: released %.3f GiB of uploaded weights; %.3f GiB of %.3f GiB of lookup tables touched after %.1f s%s\n",
                    released/double(1ull<<30),kept/double(1ull<<30),
                    std::accumulate(keep.begin(),keep.end(),size_t(0),[](size_t a,const PageRange& r) { return a+r.bytes; })/double(1ull<<30),
                    std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count(),
                    floored ? " (stopped at the MemAvailable floor)" : "");
            });
        }
        allocateRows("$logits",experts*4,rows); allocateRows("$sel",160,rows); allocateRows("$ffh",(used+1)*ff*4,rows);
        allocate("$position",4);
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384};
        VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        info.maxSets = 4096; info.poolSizeCount = 1; info.pPoolSizes = &size;
        VK_CHECK(vkCreateDescriptorPool(c.dev, &info, nullptr, &pool));
        if (const char* profile = getenv("QK_FLASH_PROFILE"); profile && strcmp(profile,"0") && c.hasTimestamps) {
            VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            q.queryType = VK_QUERY_TYPE_TIMESTAMP; q.queryCount = profileEntries;
            VK_CHECK(vkCreateQueryPool(c.dev, &q, nullptr, &profileQuery));
        }
    }
    // QK_FLASH_TIMING=1: host-side phase timing of the serial forward.
    struct Timing { double prep = 0, ple = 0, gpu = 0, post = 0; uint32_t tokens = 0; } timing;
    static double nowMs() { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    std::vector<float> forward(uint32_t token, bool reset = true, const float* residualInput = nullptr) {
        static const bool timed = [] { const char* v = getenv("QK_FLASH_TIMING"); return v && strcmp(v,"0"); }();
        double t0 = timed ? nowMs() : 0;
        if (reset) { position = 0; tokenHistory.clear(); }
        if (position >= capacity) throw std::runtime_error("native graph context capacity exceeded");
        const auto* embedding = g.find("token_embd.weight");
        if (!embedding || embedding->type != GGML_Q5_K || embedding->ne[0] != n || token >= embedding->ne[1])
            throw std::runtime_error("unsupported embedding or token");
        std::vector<float> hidden(n*hc);
        if (firstLayer==0) {
        if (residualInput) throw std::runtime_error("first stage requires token input");
        dequant_row_q5_K((const block_q5_K*)(embedding->data + token*ggmlRowBytes(embedding->type,n)),hidden.data(),n);
        for (uint32_t s = 1; s < hc; ++s) memcpy(hidden.data()+s*n,hidden.data(),n*4);
        } else {
            if (!residualInput) throw std::runtime_error("later stage requires all HC streams");
            std::copy(residualInput,residualInput+n*hc,hidden.begin());
            for (float value:hidden) if (!std::isfinite(value)) throw std::runtime_error("nonfinite stage input");
        }
        // One submission per token: pack the small host inputs into a region
        // disjoint from readback, then record their copies with the graph.
        // Weight loading retains its bounded synchronous staging path.
        const size_t inputOffset = 2 << 20;
        const size_t positionOffset = inputOffset + hidden.size()*4;
        const size_t pleOffset = positionOffset + 4;
        memcpy((uint8_t*)mapped+inputOffset,hidden.data(),hidden.size()*4);
        memcpy((uint8_t*)mapped+positionOffset,&position,4);
        double t1 = timed ? nowMs() : 0;
        if (ple) {
            std::vector<float> row(n);
            ple->gather(ple->config().rows(token,tokenHistory),row.data());
            memcpy((uint8_t*)mapped+pleOffset,row.data(),row.size()*4);
        }
        double t2 = timed ? nowMs() : 0;
        const char* replayEnv=getenv("QK_FLASH_REPLAY");
        const bool replayEnabled=(!replayEnv || strcmp(replayEnv,"0")) && !getenv("QK_LAYER_DUMP");
        const bool reuse=replayReady && replayEnabled && !reset;
        VkMemoryBarrier read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        if (!reuse) {
        VK_CHECK(vkResetDescriptorPool(c.dev,pool,0));
        begin();
        VkBufferCopy hiddenCopy{inputOffset,0,hidden.size()*4};
        vkCmdCopyBuffer(c.cb,staging.buf,buffers.at("$hidden").buf,1,&hiddenCopy);
        VkBufferCopy positionCopy{positionOffset,0,4};
        vkCmdCopyBuffer(c.cb,staging.buf,buffers.at("$position").buf,1,&positionCopy);
        if (ple) {
            VkBufferCopy pleCopy{pleOffset,0,n*4};
            vkCmdCopyBuffer(c.cb,staging.buf,buffers.at("$ple_emb").buf,1,&pleCopy);
        }
        if (reset) {
            // Attention reads only [0,position], and fa_prep overwrites the
            // current row before it is read. Reset its logical length instead
            // of clearing GiBs of unreachable KV. Recurrent state must clear.
            for (layer=firstLayer; layer<=lastLayer; ++layer) for (const auto& kind : {"convstate","state"})
                if (buffers.count(state(kind))) vkCmdFillBuffer(c.cb,buffers.at(state(kind)).buf,0,VK_WHOLE_SIZE,0);
            if (ple) vkCmdFillBuffer(c.cb,buffers.at("$ple_history").buf,0,VK_WHOLE_SIZE,0);
        }
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
        for (layer=firstLayer; layer<=lastLayer; ++layer) {
        if (layer == 1) pleForward();
        hcMix("attn","$hidden");
        if (layer % 4 == 3) fullAttention();
        else {
        project(w("attn_qkv.weight"),"$mixed","$qkv",false);
        project(w("attn_gate.weight"),"$mixed","$z",false);
        project(w("ssm_alpha.weight"),"$mixed","$alpha",false);
        project(w("ssm_beta.weight"),"$mixed","$beta");
        tap("qkv", "$qkv"); tap("z", "$z"); tap("alpha", "$alpha"); tap("beta", "$beta");
        struct { uint32_t heads,T; } params{48,1};
        emit("qwen4_gdn_params.spv",{"$alpha","$beta",w("ssm_dt.bias"),w("ssm_a"),"$gb"},params,1,0,1,false);
        struct { uint32_t channels,ds,qk; float eps; } conv{10240,128,4096,eps};
        emit("dn_convn.spv",{state("convstate"),"$qkv",w("ssm_conv1d.weight"),"$conv"},conv,80,1);
        tap("conv", "$conv");
        struct { uint32_t ds,hk,hv,kdiv; float eps; } step{128,16,48,0,eps};
        // QK_GDN_STEP=v1 keeps the 128-thread dn_step_gate kernel for A/B.
        static const bool stepV1 = [] { const char* v = getenv("QK_GDN_STEP"); return v && !strcmp(v,"v1"); }();
        if (stepV1) emit("dn_step_gate.spv",{"$conv","$gb",state("state"),w("ssm_norm.weight"),"$z","$att"},step,48,1);
        else emit("qwen4_gdn_step.spv",{"$conv","$gb",state("state"),w("ssm_norm.weight"),"$z","$att"},step,48);
        tap("final_output", "$att");
        project(w("ssm_out.weight"),"$att","$block");
        tap("linear_attn_out", "$block");
        }
        hcCombine("$hidden","$residual");
        tap("attn.combine", "$residual");
        hcMix("ffn","$residual");
        struct { uint32_t n,ff,experts,used; } moe{n,ff,experts,used};
        const bool halo = c.props.deviceID == 0x1586;
        const bool downQ51 = g.find(w("ffn_down_exps.weight"))->type==GGML_Q5_1;
        const bool sharedQ51 = g.find(w("ffn_down_shexp.weight"))->type==GGML_Q5_1;
        emit("moe_logits.spv",{w("ffn_gate_inp.weight"),"$mixed","$logits"},moe,experts);
        emit("moe_select_256.spv",{"$logits",w("ffn_gate_inp_shexp.weight"),"$mixed","$sel"},moe,1);
        emit("moe_gateup_q5k.spv",{w("ffn_gate_exps.weight"),w("ffn_up_exps.weight"),"$mixed","$sel","$ffh"},moe,used*ff,halo&&!downQ51?128:64,1,false);
        emit("moe_shared_q5k.spv",{w("ffn_gate_shexp.weight"),w("ffn_up_shexp.weight"),"$mixed","$ffh"},moe,ff,halo&&!downQ51?128:64);
        emit(downQ51?"moe_down_q5_1.spv":"moe_down_q8_routed.spv",{w("ffn_down_exps.weight"),"$ffh","$sel","$block"},moe,n,halo?128:256);
        emit(sharedQ51?"moe_down_shared_q5_1.spv":"moe_down_q8.spv",{w("ffn_down_shexp.weight"),"$ffh","$sel","$block"},moe,n,sharedQ51?64:0);
        tap("ffn_out", "$block");
        hcCombine("$residual","$hidden");
        }
        if (withHead) {
            hcMix("head","$hidden",true);
            project("output.weight","$mixed","$output_logits");
        }
        read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(c.cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&read,0,nullptr,0,nullptr);
        VkBufferCopy copy{0,0,hidden.size()*4};
        vkCmdCopyBuffer(c.cb,buffers.at("$hidden").buf,staging.buf,1,&copy);
        if (withHead) {
            VkBufferCopy headCopy{0,hidden.size()*4,logits.size()*4};
            vkCmdCopyBuffer(c.cb,buffers.at("$output_logits").buf,staging.buf,1,&headCopy);
        }
        read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(c.cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&read,0,nullptr,0,nullptr);
        submit();
        } else submit(false);
        double t3 = timed ? nowMs() : 0;
        replayReady=replayEnabled && !reset;
        if (profileQuery && !reuse) printProfile("serial forward, 1 token");
        memcpy(hidden.data(),mapped,hidden.size()*4);
        if (withHead) {
            memcpy(logits.data(),(uint8_t*)mapped+hidden.size()*4,logits.size()*4);
            for (float value:logits) if (!std::isfinite(value)) throw std::runtime_error("nonfinite native logits");
        }
        if (const char* prefix = getenv("QK_LAYER_DUMP")) for (const auto& [name,ref] : taps) {
            const auto& buffer = buffers.at(ref);
            if (buffer.size > stageBytes) throw std::runtime_error("debug tap too large");
            begin();
            read.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(c.cb,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&read,0,nullptr,0,nullptr);
            VkBufferCopy debugCopy{0,0,buffer.size};
            vkCmdCopyBuffer(c.cb,buffer.buf,staging.buf,1,&debugCopy);
            read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(c.cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&read,0,nullptr,0,nullptr);
            submit();
            const auto path = std::string(prefix)+".t"+std::to_string(position)+"."+name;
            FILE* file = fopen(path.c_str(),"wb");
            if (!file) throw std::runtime_error("cannot write debug tap");
            bool ok = fwrite(mapped,1,buffer.size,file) == buffer.size;
            if (fclose(file) != 0 || !ok) throw std::runtime_error("debug tap write failed");
        }
        ++position; tokenHistory.push_back(token);
        if (tokenHistory.size() > 2) tokenHistory.erase(tokenHistory.begin());
        if (timed) {
            double t4 = nowMs();
            timing.prep += t1-t0; timing.ple += t2-t1; timing.gpu += t3-t2; timing.post += t4-t3; ++timing.tokens;
            if (timing.tokens % 16 == 0) {
                fprintf(stderr,"[flash timing] stage %u:%u avg over %u tokens: prep %.2f ms, ple %.2f ms, submit+wait %.2f ms, post %.2f ms\n",
                        firstLayer,lastLayer+1,timing.tokens,timing.prep/timing.tokens,timing.ple/timing.tokens,timing.gpu/timing.tokens,timing.post/timing.tokens);
                timing = Timing{};
            }
        }
        return hidden;
    }
    // Batched prefill of T consecutive positions. The first stage takes token
    // ids, later stages T*10240 residual rows. Non-head stages write T hidden
    // rows to hiddenOut; the head stage writes the greedy id after each
    // position to idsOut and keeps the final position's full logits. State
    // (conv window, GDN state, KV rows, PLE history) continues exactly as the
    // serial path would have left it, so decode can follow without a reset.
    void forwardBatch(const uint32_t* tokens, const float* residualIn, uint32_t T, bool reset,
                      float* hiddenOut, uint32_t* idsOut) {
        if (!batchCap || T < 1 || T > batchCap) throw std::runtime_error("native batch size unsupported");
        if (withHead ? !idsOut : !hiddenOut) throw std::runtime_error("native batch output missing");
        if (reset) { position = 0; tokenHistory.clear(); }
        if (position + T > capacity) throw std::runtime_error("native graph context capacity exceeded");
        replayReady = false;
        const auto* embedding = g.find("token_embd.weight");
        if (!embedding || embedding->type != GGML_Q5_K || embedding->ne[0] != n)
            throw std::runtime_error("unsupported embedding");
        std::vector<float> hidden(size_t(T)*n*hc);
        if (firstLayer==0) {
            if (residualIn || !tokens) throw std::runtime_error("first stage requires token input");
            for (uint32_t t = 0; t < T; ++t) {
                if (tokens[t] >= embedding->ne[1]) throw std::runtime_error("unsupported token");
                float* row = hidden.data()+size_t(t)*n*hc;
                dequant_row_q5_K((const block_q5_K*)(embedding->data + size_t(tokens[t])*ggmlRowBytes(embedding->type,n)),row,n);
                for (uint32_t s = 1; s < hc; ++s) memcpy(row+s*n,row,n*4);
            }
        } else {
            if (!residualIn) throw std::runtime_error("later stage requires all HC streams");
            std::copy(residualIn,residualIn+hidden.size(),hidden.begin());
            for (float value:hidden) if (!std::isfinite(value)) throw std::runtime_error("nonfinite stage input");
        }
        auto history = tokenHistory;
        std::vector<float> pleRows;
        if (ple) pleRows.resize(size_t(T)*n);
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t token = tokens ? tokens[t] : 0;
            if (ple) ple->gather(ple->config().rows(token,history),pleRows.data()+size_t(t)*n);
            history.push_back(token);
            if (history.size() > 2) history.erase(history.begin());
        }
        upload(buffers.at("$hidden"),hidden.data(),hidden.size()*4);
        if (ple) upload(buffers.at("$ple_emb"),pleRows.data(),pleRows.size()*4);
        VK_CHECK(vkResetDescriptorPool(c.dev,pool,0));
        begin();
        if (reset) {
            for (layer=firstLayer; layer<=lastLayer; ++layer) for (const auto& kind : {"convstate","state"})
                if (buffers.count(state(kind))) vkCmdFillBuffer(c.cb,buffers.at(state(kind)).buf,0,VK_WHOLE_SIZE,0);
            if (ple) vkCmdFillBuffer(c.cb,buffers.at("$ple_history").buf,0,VK_WHOLE_SIZE,0);
        }
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
        const uint32_t base = position;
        // Flush every few layers so no single submission approaches the
        // kernel driver's ring timeout; order and visibility are unchanged.
        static const uint32_t flushEvery = [] {
            const char* v = getenv("QK_SUBMIT_LAYERS"); long x = v ? atol(v) : 8; return (uint32_t)(x < 1 ? 1 : x); }();
        uint32_t sinceFlush = 0;
        for (layer=firstLayer; layer<=lastLayer; ++layer) {
            if (layer == 1) pleForwardBatch(T,base);
            hcMixBatch("attn","$hidden",T);
            if (layer % 4 == 3) fullAttentionBatch(T,base); else gdnBatch(T);
            hcCombineBatch("$hidden","$residual",T);
            hcMixBatch("ffn","$residual",T);
            moeBatch(T);
            hcCombineBatch("$residual","$hidden",T);
            if (++sinceFlush == flushEvery && layer < lastLayer) {
                submit(); begin(); barrier(); sinceFlush = 0;
            }
        }
        if (withHead) headBatch(T);
        submit();
        if (withHead) {
            download(buffers.at("$ids"),0,idsOut,size_t(T)*4);
            const uint32_t last = (T-1) % headTile;
            download(buffers.at("$output_logits"),size_t(last)*logits.size()*4,logits.data(),logits.size()*4);
            for (float value:logits) if (!std::isfinite(value)) throw std::runtime_error("nonfinite native logits");
        } else download(buffers.at("$hidden"),0,hiddenOut,hidden.size()*4);
        position += T; tokenHistory = history;
        if (profileQuery) { char what[64]; snprintf(what,sizeof what,"batched forward, %u tokens",T); printProfile(what); }
    }
};

// Serial-versus-batched consistency on the bounded prefix graph: the same
// token sequence through the serial forward, one whole batch, mixed chunk
// sizes and a batch followed by serial continuation. No external reference.
static bool caseQwen4Batch(VkCtx& c, const char* path, uint32_t token, uint32_t steps, uint32_t lastLayer) {
    if (!steps || steps > 512 || token > UINT32_MAX-steps || lastLayer>3) return false;
    Gguf model;
    if (!model.open(path)) return false;
    try {
        Qwen4Graph test(c,model,lastLayer,std::max(32u,steps)); test.open();
        std::vector<uint32_t> tokens(steps);
        for (uint32_t i = 0; i < steps; ++i) tokens[i] = token+i;
        std::vector<float> serial;
        for (uint32_t step = 0; step < steps; ++step) {
            auto frame = test.forward(tokens[step],step==0);
            serial.insert(serial.end(),frame.begin(),frame.end());
        }
        // Exact tier (F32 GEMMs): every frame within 1e-5 of serial. The
        // coopmat tier rounds inputs to f16, so near-tie expert routing can
        // flip on a few tokens; it is judged on the median frame and the
        // number of flipped frames, and labeled as reduced precision.
        const char* coopEnv = getenv("QK_FLASH_COOPMAT");
        const bool reducedTier = coopEnv && strcmp(coopEnv,"0");
        auto compare = [&](const char* label, const std::vector<float>& actual) {
            std::vector<double> frames(steps);
            double maxAbs = 0; bool exact = true;
            for (uint32_t step = 0; step < steps; ++step) {
                double err = 0, ref = 0;
                for (size_t i = size_t(step)*10240; i < size_t(step+1)*10240; ++i) {
                    if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite batched output");
                    const double delta = double(actual[i])-serial[i];
                    err += delta*delta; ref += double(serial[i])*serial[i];
                    maxAbs = std::max(maxAbs,std::fabs(delta));
                    exact = exact && actual[i]==serial[i];
                }
                frames[step] = std::sqrt(err/std::max(ref,1e-20));
            }
            std::vector<double> sorted(frames); std::sort(sorted.begin(),sorted.end());
            const double worst = sorted.back(), median = sorted[steps/2];
            const uint32_t flipped = (uint32_t)std::count_if(frames.begin(),frames.end(),[](double f) { return f > 1e-2; });
            bool ok = reducedTier ? (median < 5e-3 && flipped <= std::max(1u,steps/50)) : worst < 1e-5;
            printf("batch check %-16s frames=%u worst_frame_relative_rms=%.3g median=%.3g frames>1e-2=%u max_abs=%.3g%s%s -> %s\n",
                   label,steps,worst,median,flipped,maxAbs,exact?" (bit-exact)":"",reducedTier?" [coopmat f16 tier]":"",ok?"PASS":"FAIL");
            return ok;
        };
        std::vector<float> rows(serial.size());
        auto elapsedBatch = std::chrono::steady_clock::now();
        test.forwardBatch(tokens.data(),nullptr,steps,true,rows.data(),nullptr);
        double batchSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-elapsedBatch).count();
        bool ok = compare("whole",rows);
        std::fill(rows.begin(),rows.end(),0.f);
        uint32_t done = 0;
        for (uint32_t chunk : {5u,1u,7u,3u}) {
            uint32_t n = std::min(chunk,steps-done);
            if (!n) break;
            if (n == 1) { auto frame = test.forward(tokens[done],done==0); std::copy(frame.begin(),frame.end(),rows.begin()+size_t(done)*10240); }
            else test.forwardBatch(tokens.data()+done,nullptr,n,done==0,rows.data()+size_t(done)*10240,nullptr);
            done += n;
        }
        while (done < steps) { test.forwardBatch(tokens.data()+done,nullptr,steps-done,false,rows.data()+size_t(done)*10240,nullptr); done = steps; }
        ok &= compare("mixed 5+1+7+3",rows);
        std::fill(rows.begin(),rows.end(),0.f);
        uint32_t half = steps/2;
        if (half) test.forwardBatch(tokens.data(),nullptr,half,true,rows.data(),nullptr);
        for (uint32_t step = half; step < steps; ++step) {
            auto frame = test.forward(tokens[step],step==0);
            std::copy(frame.begin(),frame.end(),rows.begin()+size_t(step)*10240);
        }
        ok &= compare("batch then serial",rows);
        auto elapsedSerial = std::chrono::steady_clock::now();
        for (uint32_t step = 0; step < steps; ++step) test.forward(tokens[step],step==0);
        double serialSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-elapsedSerial).count();
        // Warm timing: pipelines were compiled lazily during the first batch.
        auto elapsedWarm = std::chrono::steady_clock::now();
        test.forwardBatch(tokens.data(),nullptr,steps,true,rows.data(),nullptr);
        double warmSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-elapsedWarm).count();
        printf("prefix timing: serial %.3f s (%.1f tok/s), first batch %.3f s, warm batch %.3f s (%.1f tok/s) for %u tokens through layers 0:%u\n",
               serialSeconds,steps/serialSeconds,batchSeconds,warmSeconds,steps/warmSeconds,steps,lastLayer+1);
        return ok;
    } catch (const std::exception& error) { fprintf(stderr,"%s\n",error.what()); return false; }
}

static bool caseQwen4Prefix(VkCtx& c, const char* path, uint32_t token, const char* reference, uint32_t steps, uint32_t lastLayer = 0) {
    if (!steps || steps > 32 || token > UINT32_MAX-steps || lastLayer>3) return false;
    Gguf model;
    if (!model.open(path)) return false;
    try {
        Qwen4Graph test(c,model,lastLayer); test.open();
        std::vector<float> actual;
        for (uint32_t step = 0; step < steps; ++step) {
            auto frame = test.forward(token+step,step==0);
            actual.insert(actual.end(),frame.begin(),frame.end());
        }
        auto resetFrame = test.forward(token,true);
        if (!std::equal(resetFrame.begin(),resetFrame.end(),actual.begin())) {
            fprintf(stderr,"prefix reset did not reproduce its first frame\n"); return false;
        }
        if (!reference) {
            double sum = 0;
            for (float x : actual) { if (!std::isfinite(x)) return false; sum += x; }
            printf("native prefix: %zu finite values; reset exact; checksum=%.9g (not a parity test)\n",actual.size(),sum);
            return true;
        }
        FILE* file = fopen(reference,"rb");
        if (!file) { fprintf(stderr,"cannot open layer reference: %s\n",reference); return false; }
        std::vector<float> expected(actual.size());
        bool sizeOk = fread(expected.data(),4,expected.size(),file) == expected.size() && fgetc(file) == EOF;
        fclose(file); if (!sizeOk) return false;
        double error = 0, energy = 0, maxError = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) return false;
            double delta = double(actual[i])-expected[i];
            error += delta*delta; energy += double(expected[i])*expected[i];
            maxError = std::max(maxError,std::fabs(delta));
        }
        double relative = std::sqrt(error/std::max(energy,1e-20));
        // Generate the reference with Vulkan MMVQ/F16 disabled: otherwise its
        // quantized matrix inputs introduce a second, different approximation.
        bool ok = relative < 1e-5;
        double worstFrame = 0;
        for (uint32_t step=0; step<steps; ++step) {
            double err = 0, ref = 0;
            for (size_t i=step*10240; i<(step+1)*10240; ++i) {
                const double delta = double(actual[i])-expected[i];
                err += delta*delta; ref += double(expected[i])*expected[i];
            }
            worstFrame = std::max(worstFrame,std::sqrt(err/std::max(ref,1e-20)));
        }
        ok &= worstFrame < 1e-5;
        printf("native layers 0:%u (%u steps) vs F32 reference: relative_rms=%.6g max_abs=%.6g -> %s\n",
               lastLayer+1,steps,relative,maxError,ok?"PASS":"FAIL");
        printf("worst frame relative_rms=%.6g; reset exact\n",worstFrame);
        return ok;
    } catch (const std::exception& error) { fprintf(stderr,"%s\n",error.what()); return false; }
}

// Synthetic check of the batched GEMM tiers: random Q8_0 weights [M][K] and
// activations [N][K] against a double-precision reference, scalar and coopmat.
static bool caseQwen4Gemm(VkCtx& c, uint32_t M, uint32_t K, uint32_t N) {
    if (!M || !K || !N || K % 64 || M % 128 || N > 4096) { fprintf(stderr,"qwen4-gemm needs M%%128==0, K%%64==0\n"); return false; }
    std::mt19937 rng(7);
    std::vector<block_q8_0> blocks((size_t)M*K/32);
    for (auto& b : blocks) { b.d = qk_f32_to_f16(0.001f + 0.003f*(rng()&0xffff)/65536.f); for (auto& q : b.qs) q = (int8_t)(rng()%255-127); }
    std::vector<float> x((size_t)N*K), dq(K);
    for (auto& v : x) v = ((rng()&0xffff)/65536.f - 0.5f)*4.f;
    std::vector<double> ref((size_t)N*M);
    for (uint32_t r = 0; r < M; ++r) {
        dequant_row_q8_0(&blocks[(size_t)r*K/32], dq.data(), K);
        for (uint32_t t = 0; t < N; ++t) { double acc = 0; for (uint32_t k = 0; k < K; ++k) acc += double(dq[k])*x[(size_t)t*K+k]; ref[(size_t)t*M+r] = acc; }
    }
    auto host = [&](size_t bytes) { return createBuf(c, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false); };
    Buf bw = host(blocks.size()*sizeof(block_q8_0)), bx = host(x.size()*4), by = host((size_t)N*M*4);
    auto fill = [&](Buf& b, const void* src, size_t bytes) { void* p; VK_CHECK(vkMapMemory(c.dev,b.mem,0,VK_WHOLE_SIZE,0,&p)); memcpy(p,src,bytes); vkUnmapMemory(c.dev,b.mem); };
    fill(bw, blocks.data(), blocks.size()*sizeof(block_q8_0)); fill(bx, x.data(), x.size()*4);
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16};
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.maxSets = 4; info.poolSizeCount = 1; info.pPoolSizes = &size;
    VkDescriptorPool pool; VK_CHECK(vkCreateDescriptorPool(c.dev,&info,nullptr,&pool));
    bool ok = true;
    for (const char* shader : {"qwen4_gemm_q8_0.spv","qwen4_gemm_coop_q8_0.spv"}) {
        std::vector<float> zero((size_t)N*M, 0.f); fill(by, zero.data(), zero.size()*4);
        Pipe pipe = makePipe(c, shader, 3, 28, 0);
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &pipe.dsl;
        VkDescriptorSet set; VK_CHECK(vkAllocateDescriptorSets(c.dev,&alloc,&set));
        Buf* bufs[3] = {&bw,&bx,&by};
        VkDescriptorBufferInfo bi[3]; VkWriteDescriptorSet wr[3];
        for (uint32_t i = 0; i < 3; ++i) { bi[i] = {bufs[i]->buf,0,bufs[i]->size}; wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstSet = set; wr[i].dstBinding = i; wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &bi[i]; }
        vkUpdateDescriptorSets(c.dev,3,wr,0,nullptr);
        VK_CHECK(vkResetCommandBuffer(c.cb,0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; VK_CHECK(vkBeginCommandBuffer(c.cb,&begin));
        struct { uint32_t m,k,n,xs,xo,ys,yo; } pc{M,K,N,K,0,M,0};
        vkCmdBindPipeline(c.cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe.p);
        vkCmdBindDescriptorSets(c.cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe.pl,0,1,&set,0,nullptr);
        vkCmdPushConstants(c.cb,pipe.pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof pc,&pc);
        vkCmdDispatch(c.cb,M/128,1,(N+63)/64);
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue,1,&si,VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(c.queue));
        float* out; VK_CHECK(vkMapMemory(c.dev,by.mem,0,VK_WHOLE_SIZE,0,(void**)&out));
        double err = 0, energy = 0, worst = 0; size_t bad = 0; uint32_t firstBadRow = 0, firstBadTok = 0;
        for (uint32_t t = 0; t < N; ++t) for (uint32_t r = 0; r < M; ++r) {
            double v = out[(size_t)t*M+r], e = ref[(size_t)t*M+r], d = v-e;
            err += d*d; energy += e*e;
            double rel = std::fabs(d)/std::max(std::fabs(e),1e-3);
            if (rel > worst) worst = rel;
            if (rel > 0.05) { if (!bad) { firstBadRow = r; firstBadTok = t; } ++bad; }
        }
        vkUnmapMemory(c.dev,by.mem);
        printf("%-28s M=%u K=%u N=%u relative_rms=%.3g worst_rel=%.3g bad(>5%%)=%zu/%zu first_bad=(row %u, token %u)\n",
               shader,M,K,N,std::sqrt(err/std::max(energy,1e-20)),worst,bad,(size_t)N*M,firstBadRow,firstBadTok);
        ok &= bad == 0;
        destroyPipe(c,pipe);
    }
    vkDestroyDescriptorPool(c.dev,pool,nullptr);
    destroyBuf(c,bw); destroyBuf(c,bx); destroyBuf(c,by);
    return ok;
}
