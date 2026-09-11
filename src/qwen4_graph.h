// Single-sequence native qwen4exp graph. Prefix correctness tests use the same
// kernels as the experimental split-stage adapter, with a small weight budget.
#include "qwen4_ple.h"
#include <memory>
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
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Buf staging;
    void* mapped = nullptr;
    static constexpr size_t stageBytes = 16 << 20;
    const uint32_t n = 2560, hc = 4, low = 320, ff = 640, experts = 512, used = 10;
    float eps = 1e-6f;
    std::string w(const std::string& suffix) const { return "blk." + std::to_string(layer) + "." + suffix; }
    std::string state(const std::string& kind) const { return "$" + kind + "." + std::to_string(layer); }
    void begin() {
        VK_CHECK(vkResetCommandBuffer(c.cb, 0));
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(c.cb, &info));
    }
    void submit(bool endRecording = true) {
        if (endRecording) VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        info.commandBufferCount = 1; info.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &info, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    }
    void barrier(VkPipelineStageFlags source = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VkAccessFlags access = VK_ACCESS_SHADER_WRITE_BIT) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = access; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(c.cb, source, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }
    Buf& allocate(const std::string& name, size_t bytes) {
        auto [it, inserted] = buffers.emplace(name, Buf{});
        if (inserted) it->second = createBuf(c, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        if (!it->second.deviceLocal) throw std::runtime_error("native graph refuses host-memory spill: " + name);
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
    template<class PC> void emit(const char* shader, std::initializer_list<std::string> refs,
                                  const PC& pc, uint32_t gx, uint32_t spec = 0) {
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
        uint32_t nx=std::min(gx,c.props.limits.maxComputeWorkGroupCount[0]);
        uint32_t ny=(gx+nx-1)/nx;
        if (ny>c.props.limits.maxComputeWorkGroupCount[1]) throw std::runtime_error("dispatch exceeds device limits");
        vkCmdDispatch(c.cb, nx, ny, 1); barrier();
    }
    void project(const std::string& weight, const std::string& input, const std::string& output) {
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
            case GGML_Q6_K: shader = "gemv_q6_k.spv"; units = k/16; break;
            case GGML_Q8_0: shader = "gemv_q8_0.spv"; units = k/32; break;
            default: throw std::runtime_error("unsupported projection format: " + weight);
        }
        uint32_t tpr = 256;
        while (tpr > 4 && tpr/2 >= units) tpr /= 2;
        if (tensor->type == GGML_Q5_K && c.props.deviceID == 0x1586) {
            if (m == 640 && k == 2560) tpr = 64;
            if (m == 320 && k == 10240) tpr = 128;
        }
        struct { uint32_t m,k; } pc{m,k};
        emit(shader, {weight,input,output}, pc, (m + 256/tpr - 1) / (256/tpr), tpr);
    }
    struct HcPC { uint32_t n,h,t,mode; float eps; };
    void tap(const std::string& name, const std::string& source) {
        if (!getenv("QK_LAYER_DUMP")) return;
        const auto target = "$tap." + std::to_string(layer) + "." + name;
        const size_t bytes = buffers.at(source).size;
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
        project(name("up.weight"), "$silu", "$gate");
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
        project(w("ple_key.weight"),"$ple_emb","$ple_key");
        project(w("ple_value.weight"),"$ple_emb","$ple_value");
        HcPC pc{n,hc,1,0,eps};
        emit("qwen4_hc.spv",{"$ple_key",w("ple_norm_key.weight"),"$dummy","$dummy","$ple_keynorm"},pc,hc);
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
        project(w("attn_q.weight"),"$mixed","$fa_qfull");
        project(w("attn_k.weight"),"$mixed","$fa_k");
        project(w("attn_v.weight"),"$mixed","$fa_v");
        struct { uint32_t pos,tmax,dh,nrot,hq,hkv; float eps,base; } pc{0,capacity,256,64,24,2,eps,1e7f};
        emit("fa_prep_srv.spv",{"$fa_qfull","$fa_k","$fa_v",w("attn_q_norm.weight"),w("attn_k_norm.weight"),
             "$fa_qhat",state("kcache"),state("vcache"),"$rope","$position"},pc,28);
        emit("fa_attn_srv.spv",{"$fa_qhat",state("kcache"),state("vcache"),"$fa_qfull","$att","$position"},pc,24);
        tap("attn_gated","$att");
        project(w("attn_output.weight"),"$att","$block");
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
    Qwen4Graph(VkCtx& context, Gguf& model, uint32_t endLayer) : c(context), g(model), lastLayer(endLayer) {}
    Qwen4Graph(VkCtx& context, Gguf& model, uint32_t first, uint32_t end, uint32_t ctx, uint64_t budget, bool head)
        : c(context), g(model), firstLayer(first), lastLayer(end-1), capacity(ctx), weightLimit(budget), withHead(head), servingBudget(true) {}
    uint32_t currentPosition() const { return position; }
    const std::vector<float>& lastLogits() const { return logits; }
    ~Qwen4Graph() {
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
        if (servingBudget) {
            if (!c.memoryBudget) throw std::runtime_error("native serving requires Vulkan memory-budget reporting");
            uint64_t need=totalWeightBytes+(256ull<<20); // staging, scratch, descriptors and allocation padding
            for (layer=firstLayer; layer<=lastLayer; ++layer)
                need+=layer%4==3 ? 2ull*2*capacity*256*4 : (48ull*128*128+10240*3)*4;
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
            props.pNext=&budget; vkGetPhysicalDeviceMemoryProperties2(c.phys,&props);
            uint32_t type=findMemType(c.mp,UINT32_MAX,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (type==UINT32_MAX) throw std::runtime_error("no device-local memory type");
            uint32_t heap=c.mp.memoryTypes[type].heapIndex;
            uint64_t free=budget.heapBudget[heap]>budget.heapUsage[heap] ? budget.heapBudget[heap]-budget.heapUsage[heap] : 0;
            printf("native memory estimate %.3f GiB; reported heap headroom %.3f GiB\n",need/double(1ull<<30),free/double(1ull<<30));
            if (need>free) throw std::runtime_error("insufficient device budget: unload the other model or reduce the native stage/context");
        }
        printf("prefix weights: %.3f GiB; staging: 16 MiB; PLE table stays mapped\n",totalWeightBytes/double(1ull<<30));
        staging = createBuf(c, stageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        VK_CHECK(vkMapMemory(c.dev, staging.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
        for (layer=firstLayer; layer<=lastLayer; ++layer) for (const auto& [name,tensor] : g.tensors()) if (name.compare(0,w("").size(),w("")) == 0) {
            if (!tensor.nbytes) throw std::runtime_error("unknown tensor layout: " + name);
            auto& b = allocate(name,tensor.nbytes); upload(b,tensor.data,tensor.nbytes);
        }
        if (withHead) {
            for (const auto& name : {"output_hc_norm.weight","output_hc_down.weight","output_hc_up.weight","output.weight"}) {
                const auto* tensor=g.find(name); auto& b=allocate(name,tensor->nbytes); upload(b,tensor->data,tensor->nbytes);
            }
            allocate("$output_logits",logits.size()*4);
        }
        for (const auto& name : {"$hidden","$residual","$norm","$gate","$qkv","$conv"}) allocate(name,n*hc*4);
        for (const auto& name : {"$mixed","$block","$ffout"}) allocate(name,n*4);
        allocate("$low",low*4); allocate("$silu",low*4); allocate("$inject",hc*4); allocate("$dummy",4);
        allocate("$alpha",48*4); allocate("$beta",48*4); allocate("$gb",96*4);
        allocate("$z",6144*4); allocate("$att",6144*4);
        for (layer=firstLayer; layer<=lastLayer; ++layer) {
            if (layer % 4 != 3) {
                allocate(state("convstate"),10240*3*4); allocate(state("state"),48*128*128*4);
            } else {
                allocate(state("kcache"),2ull*capacity*256*4); allocate(state("vcache"),2ull*capacity*256*4);
            }
        }
        if (lastLayer >= 3) {
            allocate("$fa_qfull",12288*4); allocate("$fa_k",512*4); allocate("$fa_v",512*4); allocate("$fa_qhat",6144*4);
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
            for (const auto& name : {"$ple_key","$ple_keynorm","$ple_query","$ple_gated","$ple_normalized"}) allocate(name,n*hc*4);
            allocate("$ple_emb",n*4); allocate("$ple_value",n*4); allocate("$ple_history",9*n*hc*4);
        }
        allocate("$logits",experts*4); allocate("$sel",160); allocate("$ffh",(used+1)*ff*4);
        allocate("$position",4);
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192};
        VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        info.maxSets = 2048; info.poolSizeCount = 1; info.pPoolSizes = &size;
        VK_CHECK(vkCreateDescriptorPool(c.dev, &info, nullptr, &pool));
    }
    std::vector<float> forward(uint32_t token, bool reset = true, const float* residualInput = nullptr) {
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
        if (ple) {
            std::vector<float> row(n);
            ple->gather(ple->config().rows(token,tokenHistory),row.data());
            memcpy((uint8_t*)mapped+pleOffset,row.data(),row.size()*4);
        }
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
        project(w("attn_qkv.weight"),"$mixed","$qkv");
        project(w("attn_gate.weight"),"$mixed","$z");
        project(w("ssm_alpha.weight"),"$mixed","$alpha");
        project(w("ssm_beta.weight"),"$mixed","$beta");
        tap("qkv", "$qkv"); tap("z", "$z"); tap("alpha", "$alpha"); tap("beta", "$beta");
        struct { uint32_t heads; } params{48};
        emit("qwen4_gdn_params.spv",{"$alpha","$beta",w("ssm_dt.bias"),w("ssm_a"),"$gb"},params,1);
        struct { uint32_t channels,ds,qk; float eps; } conv{10240,128,4096,eps};
        emit("dn_convn.spv",{state("convstate"),"$qkv",w("ssm_conv1d.weight"),"$conv"},conv,80,1);
        tap("conv", "$conv");
        struct { uint32_t ds,hk,hv,kdiv; float eps; } step{128,16,48,0,eps};
        emit("dn_step_gate.spv",{"$conv","$gb",state("state"),w("ssm_norm.weight"),"$z","$att"},step,48,1);
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
        emit("moe_gateup_q5k.spv",{w("ffn_gate_exps.weight"),w("ffn_up_exps.weight"),"$mixed","$sel","$ffh"},moe,used*ff,halo&&!downQ51?128:64);
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
        replayReady=replayEnabled && !reset;
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
        return hidden;
    }
};

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
