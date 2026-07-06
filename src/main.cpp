// qk — Qwen kernel harness.
// M1: fp16 GEMV. M2: quantized GEMV (Q8_0, Q6_K) on raw ggml blocks,
// validated against CPU dequant reference and real tensors from the GGUF.
//
// Usage:
//   qk                        synthetic suite: f16, q8_0, q6_k
//   qk f16|q8_0|q6_k [M] [K] [iters]
//   qk gguf <tensor> [iters]  real weights (QK_GGUF overrides model path)
//   qk list [filter]          list tensors in the GGUF
//
// Env: QK_DEVICE=<n> device index; QK_SHADER_DIR; QK_GGUF=<path>.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <string>
#include <vector>

#include "gguf.h"
#include "quants.h"
#include "../include/qk.h"

static const char* kDefaultGguf =
    "/home/ryan/intellij/ggerganov/llama.cpp/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf";

#define VK_CHECK(call)                                                        \
    do {                                                                      \
        VkResult r_ = (call);                                                 \
        if (r_ != VK_SUCCESS) {                                               \
            fprintf(stderr, "%s:%d: %s -> VkResult %d\n", __FILE__, __LINE__, \
                    #call, (int)r_);                                          \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ---------- Vulkan context ----------

struct Buf {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
    bool           deviceLocal = false;
};

struct VkCtx {
    VkInstance            inst = VK_NULL_HANDLE;
    VkPhysicalDevice      phys = VK_NULL_HANDLE;
    VkDevice              dev = VK_NULL_HANDLE;
    VkQueue               queue = VK_NULL_HANDLE;
    uint32_t              qfi = 0;
    bool                  hasTimestamps = false;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mp{};
    VkCommandPool         pool = VK_NULL_HANDLE;
    VkCommandBuffer       cb = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout      pl = VK_NULL_HANDLE;
    const char*           argv0 = nullptr;
};

static uint32_t findMemType(const VkPhysicalDeviceMemoryProperties& mp,
                            uint32_t typeBits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

static Buf createBuf(VkCtx& c, VkDeviceSize size, VkBufferUsageFlags usage,
                     bool preferDevice) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c.dev, &bci, nullptr, &b.buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c.dev, b.buf, &req);

    const VkMemoryPropertyFlags hostFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemType(
        c.mp, req.memoryTypeBits,
        preferDevice ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : hostFlags);

    VkResult r = mai.memoryTypeIndex == UINT32_MAX
                     ? VK_ERROR_OUT_OF_DEVICE_MEMORY
                     : vkAllocateMemory(c.dev, &mai, nullptr, &b.mem);
    b.deviceLocal = preferDevice;
    if (r != VK_SUCCESS && preferDevice) {
        fprintf(stderr, "note: device-local alloc failed (%zu MiB), using host-visible\n",
                (size_t)(size >> 20));
        mai.memoryTypeIndex = findMemType(c.mp, req.memoryTypeBits, hostFlags);
        b.deviceLocal = false;
        r = vkAllocateMemory(c.dev, &mai, nullptr, &b.mem);
    }
    VK_CHECK(r);
    VK_CHECK(vkBindBufferMemory(c.dev, b.buf, b.mem, 0));
    return b;
}

static void destroyBuf(VkCtx& c, Buf& b) {
    if (b.buf) vkDestroyBuffer(c.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(c.dev, b.mem, nullptr);
    b = Buf{};
}

static void initVk(VkCtx& c, const char* argv0) {
    c.argv0 = argv0;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "qwen-kernel";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &c.inst));

    uint32_t ndev = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(c.inst, &ndev, nullptr));
    if (!ndev) {
        fprintf(stderr, "no Vulkan devices\n");
        exit(1);
    }
    std::vector<VkPhysicalDevice> devs(ndev);
    VK_CHECK(vkEnumeratePhysicalDevices(c.inst, &ndev, devs.data()));
    int pick = -1;
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (pick < 0 && p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            pick = (int)i;
    }
    if (const char* e = getenv("QK_DEVICE")) pick = atoi(e);
    if (pick < 0) pick = 0;
    c.phys = devs[pick];
    vkGetPhysicalDeviceProperties(c.phys, &c.props);
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mp);
    printf("device: %s\n", c.props.deviceName);

    // 8/16-bit storage + int8/fp16 arithmetic for raw ggml block access
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features have11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    have11.pNext = &have12;
    VkPhysicalDeviceFeatures2 have2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    have2.pNext = &have11;
    vkGetPhysicalDeviceFeatures2(c.phys, &have2);
    if (!have11.storageBuffer16BitAccess || !have12.storageBuffer8BitAccess ||
        !have12.shaderInt8 || !have12.shaderFloat16) {
        fprintf(stderr, "device lacks 8/16-bit storage or int8/fp16 features\n");
        exit(1);
    }

    VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    en12.storageBuffer8BitAccess = VK_TRUE;
    en12.uniformAndStorageBuffer8BitAccess = have12.uniformAndStorageBuffer8BitAccess;
    en12.shaderInt8 = VK_TRUE;
    en12.shaderFloat16 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features en11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    en11.storageBuffer16BitAccess = VK_TRUE;
    en11.uniformAndStorageBuffer16BitAccess = have11.uniformAndStorageBuffer16BitAccess;
    en11.pNext = &en12;

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nqf, qf.data());
    c.qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nqf; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            c.qfi = i;
            break;
        }
    if (c.qfi == UINT32_MAX) {
        fprintf(stderr, "no compute queue\n");
        exit(1);
    }
    c.hasTimestamps = qf[c.qfi].timestampValidBits > 0;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c.qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &en11;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(c.phys, &dci, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.qfi, 0, &c.queue);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = c.qfi;
    VK_CHECK(vkCreateCommandPool(c.dev, &cpci, nullptr, &c.pool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = c.pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(c.dev, &cbai, &c.cb));

    VkDescriptorSetLayoutBinding binds[3]{};
    for (uint32_t i = 0; i < 3; i++)
        binds[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 3;
    dsli.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(c.dev, &dsli, nullptr, &c.dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &c.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(c.dev, &plci, nullptr, &c.pl));
}

static std::vector<uint32_t> loadSpv(const char* argv0, const char* name) {
    std::string path = std::string("shaders/") + name;
    if (const char* d = getenv("QK_SHADER_DIR")) {
        path = std::string(d) + "/" + name;
    } else {
        std::string exe(argv0);
        auto slash = exe.rfind('/');
        if (slash != std::string::npos) {
            std::string cand = exe.substr(0, slash) + "/shaders/" + name;
            if (FILE* f = fopen(cand.c_str(), "rb")) {
                fclose(f);
                path = cand;
            }
        }
    }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (set QK_SHADER_DIR)\n", path.c_str());
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code(((size_t)n + 3) / 4);
    if (fread(code.data(), 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "short read on %s\n", path.c_str());
        exit(1);
    }
    fclose(f);
    code.resize((size_t)n / 4);
    return code;
}

// ---------- generic GEMV run: upload, verify, benchmark ----------

static bool runGemv(VkCtx& c, const char* spvName, const void* wBytes,
                    size_t wSize, const std::vector<float>& x, uint32_t M,
                    uint32_t K, const std::vector<float>& yref, uint32_t iters,
                    uint32_t unitsPerRow, double tol = 1e-2) {
    size_t sizeX = (size_t)K * 4, sizeY = (size_t)M * 4;

    // threads-per-row spec constant: shrink for skinny rows so a workgroup
    // covers 256/TPR rows and stays fully occupied
    uint32_t tpr = 256;
    while (tpr > 4 && tpr / 2 >= unitsPerRow) tpr /= 2;
    uint32_t rowsPerWg = 256 / tpr;

    Buf bW = createBuf(c, wSize,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    Buf bX = createBuf(c, sizeX,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    Buf bY = createBuf(c, sizeY,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    Buf stage = createBuf(c, std::max(wSize, std::max(sizeX, sizeY)),
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          false);

    auto begin = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    };
    auto submitWait = [&]() {
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    };

    void* mapped;
    VK_CHECK(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    auto upload = [&](Buf& dst, const void* src, size_t n) {
        memcpy(mapped, src, n);
        begin();
        VkBufferCopy cp{0, 0, n};
        vkCmdCopyBuffer(c.cb, stage.buf, dst.buf, 1, &cp);
        submitWait();
    };
    upload(bW, wBytes, wSize);
    upload(bX, x.data(), sizeX);

    std::vector<uint32_t> code = loadSpv(c.argv0, spvName);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size() * 4;
    smci.pCode = code.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(c.dev, &smci, nullptr, &sm));

    VkSpecializationMapEntry sme{0, 0, 4};
    VkSpecializationInfo spec{1, &sme, 4, &tpr};
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = &spec;
    cpi.layout = c.pl;
    VkPipeline pipe;
    VK_CHECK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &c.dsl;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(c.dev, &dsai, &ds));

    VkDescriptorBufferInfo dbi[3] = {{bW.buf, 0, VK_WHOLE_SIZE},
                                     {bX.buf, 0, VK_WHOLE_SIZE},
                                     {bY.buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[3]{};
    for (uint32_t i = 0; i < 3; i++) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = ds;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(c.dev, 3, wr, 0, nullptr);

    // 2D grid so the workgroup count can exceed maxComputeWorkGroupCount[0]
    uint32_t wgs = (M + rowsPerWg - 1) / rowsPerWg;
    uint32_t gx = std::min(wgs, c.props.limits.maxComputeWorkGroupCount[0]);
    uint32_t gy = (wgs + gx - 1) / gx;

    struct { uint32_t M, K; } pc{M, K};
    auto bindAll = [&]() {
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(c.cb, c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &pc);
    };

    // correctness pass
    begin();
    bindAll();
    vkCmdDispatch(c.cb, gx, gy, 1);
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = bY.buf;
    bmb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bmb, 0, nullptr);
    VkBufferCopy cp{0, 0, sizeY};
    vkCmdCopyBuffer(c.cb, bY.buf, stage.buf, 1, &cp);
    submitWait();

    std::vector<float> ygpu(M);
    memcpy(ygpu.data(), mapped, sizeY);

    // scale-aware: near-zero outputs (catastrophic cancellation) are judged
    // against the output RMS, not their own tiny magnitude
    double rms = 0;
    for (uint32_t m = 0; m < M; m++) rms += (double)yref[m] * yref[m];
    rms = std::sqrt(rms / M);
    double denomFloor = std::max(1e-3, 1e-3 * rms);

    double maxRel = 0;
    uint32_t bad = 0, badShown = 0;
    for (uint32_t m = 0; m < M; m++) {
        double denom = std::max(denomFloor, (double)std::fabs(yref[m]));
        double rel = std::fabs((double)ygpu[m] - yref[m]) / denom;
        maxRel = std::max(maxRel, rel);
        if (rel > tol && bad++ < 5 && badShown++ < 5)
            printf("  y[%u]: gpu=%g ref=%g\n", m, ygpu[m], yref[m]);
    }
    bool pass = bad == 0;
    printf("correctness: max_rel_err = %.3g  ->  %s\n", maxRel, pass ? "PASS" : "FAIL");

    if (pass && c.hasTimestamps && iters > 0) {
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VkQueryPool qp;
        VK_CHECK(vkCreateQueryPool(c.dev, &qpci, nullptr, &qp));

        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        auto runBench = [&]() {
            begin();
            vkCmdResetQueryPool(c.cb, qp, 0, 2);
            bindAll();
            vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            for (uint32_t i = 0; i < iters; i++) {
                vkCmdDispatch(c.cb, gx, gy, 1);
                vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                     &mb, 0, nullptr, 0, nullptr);
            }
            vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            submitWait();
        };
        runBench();  // warm-up
        runBench();

        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(c.dev, qp, 0, 2, sizeof(ts), ts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double ns = (double)(ts[1] - ts[0]) * c.props.limits.timestampPeriod / iters;
        double bytes = (double)wSize + sizeX + sizeY;
        double flops = 2.0 * M * K;
        printf("gpu: %8.1f µs/iter | %7.1f GB/s | %8.1f GFLOP/s | %.1f MiB/iter (%s, tpr %u)\n",
               ns / 1e3, bytes / ns, flops / ns, bytes / (1 << 20),
               bW.deviceLocal ? "VRAM" : "GTT", tpr);
        vkDestroyQueryPool(c.dev, qp, nullptr);
    }

    vkUnmapMemory(c.dev, stage.mem);
    vkDestroyDescriptorPool(c.dev, dpool, nullptr);
    vkDestroyPipeline(c.dev, pipe, nullptr);
    vkDestroyShaderModule(c.dev, sm, nullptr);
    destroyBuf(c, bW);
    destroyBuf(c, bX);
    destroyBuf(c, bY);
    destroyBuf(c, stage);
    return pass;
}

// ---------- test-case builders ----------

static std::vector<float> randomX(uint32_t K, uint32_t seed = 43) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(K);
    for (auto& v : x) v = nd(rng);
    return x;
}

static void dotRefRows(const float* Wf, const std::vector<float>& x,
                       std::vector<float>& yref, uint32_t M, uint32_t K) {
    for (uint32_t m = 0; m < M; m++) {
        const float* row = Wf + (size_t)m * K;
        double acc = 0;
        for (uint32_t k = 0; k < K; k++) acc += (double)row[k] * x[k];
        yref[m] = (float)acc;
    }
}

static bool caseF16(VkCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
    printf("\n== f16 GEMV  M=%u K=%u (W %.1f MiB) ==\n", M, K, (double)M * K * 2 / (1 << 20));
    std::vector<uint16_t> Wh((size_t)M * K);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& w : Wh) w = qk_f32_to_f16(nd(rng) * 0.05f);
    auto x = randomX(K);

    std::vector<float> lut(65536);
    for (uint32_t i = 0; i < 65536; i++) lut[i] = qk_f16_to_f32((uint16_t)i);
    std::vector<float> yref(M);
    for (uint32_t m = 0; m < M; m++) {
        const uint16_t* row = &Wh[(size_t)m * K];
        double acc = 0;
        for (uint32_t k = 0; k < K; k++) acc += (double)lut[row[k]] * x[k];
        yref[m] = (float)acc;
    }
    return runGemv(c, "gemv_f16.spv", Wh.data(), Wh.size() * 2, x, M, K, yref, iters, K / 2);
}

static bool caseQ80(VkCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
    printf("\n== q8_0 GEMV  M=%u K=%u (W %.1f MiB) ==\n", M, K,
           (double)M * K / 32 * 34 / (1 << 20));
    if (K % 32) {
        fprintf(stderr, "K must be a multiple of 32\n");
        return false;
    }
    size_t nb = (size_t)M * K / 32;
    std::vector<block_q8_0> blocks(nb);
    std::mt19937 rng(42);
    for (auto& b : blocks) {
        b.d = qk_f32_to_f16(0.005f + 0.02f * (rng() & 0xFFFF) / 65536.0f);
        for (auto& q : b.qs) q = (int8_t)((int)(rng() % 255) - 127);
    }
    auto x = randomX(K);

    std::vector<float> yref(M), tmp(K);
    for (uint32_t m = 0; m < M; m++) {
        dequant_row_q8_0(&blocks[(size_t)m * (K / 32)], tmp.data(), K);
        double acc = 0;
        for (uint32_t k = 0; k < K; k++) acc += (double)tmp[k] * x[k];
        yref[m] = (float)acc;
    }
    return runGemv(c, "gemv_q8_0.spv", blocks.data(), nb * sizeof(block_q8_0),
                   x, M, K, yref, iters, K / 32);
}

static bool caseQ6K(VkCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
    printf("\n== q6_k GEMV  M=%u K=%u (W %.1f MiB) ==\n", M, K,
           (double)M * K / 256 * 210 / (1 << 20));
    if (K % 256) {
        fprintf(stderr, "K must be a multiple of 256\n");
        return false;
    }
    size_t nb = (size_t)M * K / 256;
    std::vector<block_q6_K> blocks(nb);
    std::mt19937 rng(42);
    for (auto& b : blocks) {
        for (auto& v : b.ql) v = (uint8_t)rng();
        for (auto& v : b.qh) v = (uint8_t)rng();
        for (auto& v : b.scales) v = (int8_t)((int)(rng() % 255) - 127);
        b.d = qk_f32_to_f16(0.0002f + 0.0008f * (rng() & 0xFFFF) / 65536.0f);
    }
    auto x = randomX(K);

    std::vector<float> yref(M), tmp(K);
    for (uint32_t m = 0; m < M; m++) {
        dequant_row_q6_K(&blocks[(size_t)m * (K / 256)], tmp.data(), K);
        double acc = 0;
        for (uint32_t k = 0; k < K; k++) acc += (double)tmp[k] * x[k];
        yref[m] = (float)acc;
    }
    return runGemv(c, "gemv_q6_k.spv", blocks.data(), nb * sizeof(block_q6_K),
                   x, M, K, yref, iters, K / 16);
}

static bool caseIQ4XS(VkCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
    printf("\n== iq4_xs GEMV  M=%u K=%u (W %.1f MiB) ==\n", M, K,
           (double)M * K / 256 * 136 / (1 << 20));
    if (K % 256) {
        fprintf(stderr, "K must be a multiple of 256\n");
        return false;
    }
    size_t nb = (size_t)M * K / 256;
    std::vector<block_iq4_xs> blocks(nb);
    std::mt19937 rng(42);
    for (auto& b : blocks) {
        b.d = qk_f32_to_f16(0.002f + 0.004f * (rng() & 0xFFFF) / 65536.0f);
        b.scales_h = (uint16_t)rng();
        for (auto& v : b.scales_l) v = (uint8_t)rng();
        for (auto& v : b.qs) v = (uint8_t)rng();
    }
    auto x = randomX(K);

    std::vector<float> yref(M), tmp(K);
    for (uint32_t m = 0; m < M; m++) {
        dequant_row_iq4_xs(&blocks[(size_t)m * (K / 256)], tmp.data(), K);
        double acc = 0;
        for (uint32_t k = 0; k < K; k++) acc += (double)tmp[k] * x[k];
        yref[m] = (float)acc;
    }
    return runGemv(c, "gemv_iq4_xs.spv", blocks.data(), nb * sizeof(block_iq4_xs),
                   x, M, K, yref, iters, K / 32);
}

static bool caseIQ3XXS(VkCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
    printf("\n== iq3_xxs GEMV  M=%u K=%u (W %.1f MiB) ==\n", M, K,
           (double)M * K / 256 * 98 / (1 << 20));
    if (K % 256) {
        fprintf(stderr, "K must be a multiple of 256\n");
        return false;
    }
    size_t nb = (size_t)M * K / 256;
    std::vector<block_iq3_xxs> blocks(nb);
    std::mt19937 rng(42);
    for (auto& b : blocks) {
        b.d = qk_f32_to_f16(0.002f + 0.004f * (rng() & 0xFFFF) / 65536.0f);
        for (auto& v : b.qs) v = (uint8_t)rng();
    }
    auto x = randomX(K);

    std::vector<float> yref(M), tmp(K);
    for (uint32_t m = 0; m < M; m++) {
        dequant_row_iq3_xxs(&blocks[(size_t)m * (K / 256)], tmp.data(), K);
        double acc = 0;
        for (uint32_t k = 0; k < K; k++) acc += (double)tmp[k] * x[k];
        yref[m] = (float)acc;
    }
    return runGemv(c, "gemv_iq3_xxs.spv", blocks.data(), nb * sizeof(block_iq3_xxs),
                   x, M, K, yref, iters, K / 8);
}

// ---------- multi-pipeline helpers (MoE chain) ----------

struct Pipe {
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout      pl = VK_NULL_HANDLE;
    VkShaderModule        sm = VK_NULL_HANDLE;
    VkPipeline            p = VK_NULL_HANDLE;
    uint32_t              nBind = 0;
};

static Pipe makePipe(VkCtx& c, const char* spvName, uint32_t nBind, uint32_t pcSize,
                     uint32_t tpr = 0) {
    Pipe pp;
    pp.nBind = nBind;
    std::vector<VkDescriptorSetLayoutBinding> binds(nBind);
    for (uint32_t i = 0; i < nBind; i++)
        binds[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = nBind;
    dsli.pBindings = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(c.dev, &dsli, nullptr, &pp.dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &pp.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(c.dev, &plci, nullptr, &pp.pl));

    std::vector<uint32_t> code = loadSpv(c.argv0, spvName);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size() * 4;
    smci.pCode = code.data();
    VK_CHECK(vkCreateShaderModule(c.dev, &smci, nullptr, &pp.sm));

    VkSpecializationMapEntry sme{0, 0, 4};
    VkSpecializationInfo spec{1, &sme, 4, &tpr};
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = pp.sm;
    cpi.stage.pName = "main";
    if (tpr) cpi.stage.pSpecializationInfo = &spec;
    cpi.layout = pp.pl;
    VK_CHECK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pp.p));
    return pp;
}

static void destroyPipe(VkCtx& c, Pipe& pp) {
    vkDestroyPipeline(c.dev, pp.p, nullptr);
    vkDestroyShaderModule(c.dev, pp.sm, nullptr);
    vkDestroyPipelineLayout(c.dev, pp.pl, nullptr);
    vkDestroyDescriptorSetLayout(c.dev, pp.dsl, nullptr);
}

static const char* ggufPath() {
    const char* p = getenv("QK_GGUF");
    return p ? p : kDefaultGguf;
}

static bool caseGguf(VkCtx& c, const std::string& tensorName, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    const GgufTensor* t = g.find(tensorName);
    if (!t) {
        fprintf(stderr, "tensor '%s' not found (try: qk list)\n", tensorName.c_str());
        return false;
    }
    uint32_t K = (uint32_t)t->ne[0];
    uint32_t M = (uint32_t)t->ne[1];
    size_t rowBytes = ggmlRowBytes(t->type, K);
    printf("\n== gguf %s  %s  ne=[%llu,%llu,%llu] ==\n", tensorName.c_str(),
           ggmlTypeName(t->type), (unsigned long long)t->ne[0],
           (unsigned long long)t->ne[1], (unsigned long long)t->ne[2]);
    if (!rowBytes) {
        fprintf(stderr, "type %s not supported yet (IQ kernels are a later milestone)\n",
                ggmlTypeName(t->type));
        return false;
    }
    if (t->ne[2] > 1) printf("3D tensor: using slice [:,:,0] (expert 0)\n");
    printf("GEMV M=%u K=%u, W %.1f MiB\n", M, K, (double)M * rowBytes / (1 << 20));

    auto x = randomX(K);
    std::vector<float> yref(M), tmp(K);
    printf("cpu reference...\n");
    for (uint32_t m = 0; m < M; m++) {
        const uint8_t* row = t->data + (size_t)m * rowBytes;
        switch (t->type) {
            case GGML_Q8_0: dequant_row_q8_0((const block_q8_0*)row, tmp.data(), K); break;
            case GGML_Q6_K: dequant_row_q6_K((const block_q6_K*)row, tmp.data(), K); break;
            case GGML_IQ4_XS: dequant_row_iq4_xs((const block_iq4_xs*)row, tmp.data(), K); break;
            case GGML_IQ3_XXS: dequant_row_iq3_xxs((const block_iq3_xxs*)row, tmp.data(), K); break;
            case GGML_F16:
                for (uint32_t k = 0; k < K; k++)
                    tmp[k] = qk_f16_to_f32(((const uint16_t*)row)[k]);
                break;
            case GGML_F32: memcpy(tmp.data(), row, K * 4); break;
            default: return false;
        }
        double acc = 0;
        for (uint32_t k = 0; k < K; k++) acc += (double)tmp[k] * x[k];
        yref[m] = (float)acc;
    }

    const char* spv = nullptr;
    uint32_t units = K;
    switch (t->type) {
        case GGML_Q8_0:    spv = "gemv_q8_0.spv";    units = K / 32; break;
        case GGML_Q6_K:    spv = "gemv_q6_k.spv";    units = K / 16; break;
        case GGML_IQ4_XS:  spv = "gemv_iq4_xs.spv";  units = K / 32; break;
        case GGML_IQ3_XXS: spv = "gemv_iq3_xxs.spv"; units = K / 8;  break;
        case GGML_F16:     spv = "gemv_f16.spv";     units = K / 2;  break;
        default:
            fprintf(stderr, "no kernel for %s\n", ggmlTypeName(t->type));
            return false;
    }
    return runGemv(c, spv, t->data, (size_t)M * rowBytes, x, M, K, yref, iters, units);
}

// Fused MoE decode step for one layer: logits -> select -> gate/up (routed
// IQ3_XXS + shared Q8_0) -> weighted down (IQ4_XS) + shared down (Q8_0).
// Six dispatches, four barriers, ONE queue submission per iteration.
static bool caseMoe(VkCtx& c, uint32_t layer, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    char nb[128];
    auto T = [&](const char* suffix) -> const GgufTensor* {
        snprintf(nb, sizeof nb, "blk.%u.%s", layer, suffix);
        const GgufTensor* t = g.find(nb);
        if (!t) fprintf(stderr, "missing tensor %s\n", nb);
        return t;
    };
    const GgufTensor* tGI  = T("ffn_gate_inp.weight");
    const GgufTensor* tGIS = T("ffn_gate_inp_shexp.weight");
    const GgufTensor* tGE  = T("ffn_gate_exps.weight");
    const GgufTensor* tUE  = T("ffn_up_exps.weight");
    const GgufTensor* tDE  = T("ffn_down_exps.weight");
    const GgufTensor* tGS  = T("ffn_gate_shexp.weight");
    const GgufTensor* tUS  = T("ffn_up_shexp.weight");
    const GgufTensor* tDS  = T("ffn_down_shexp.weight");
    if (!tGI || !tGIS || !tGE || !tUE || !tDE || !tGS || !tUS || !tDS) return false;
    if (tGI->type != GGML_F32 || tGIS->type != GGML_F32 ||
        tGE->type != GGML_IQ3_XXS || tUE->type != GGML_IQ3_XXS ||
        (tDE->type != GGML_IQ4_XS && tDE->type != GGML_Q6_K) ||
        tGS->type != GGML_Q8_0 || tUS->type != GGML_Q8_0 || tDS->type != GGML_Q8_0) {
        fprintf(stderr, "layer %u tensor types don't match the compiled kernels\n", layer);
        return false;
    }
    const bool downQ6 = tDE->type == GGML_Q6_K;  // layers 34/38/39 in this GGUF

    const uint32_t nEmbd = (uint32_t)tGE->ne[0];
    const uint32_t nFf   = (uint32_t)tGE->ne[1];
    const uint32_t nExp  = (uint32_t)tGE->ne[2];
    const uint32_t nUsed = 8;
    if (nExp > 256) {
        fprintf(stderr, "n_expert %u > 256 not supported by moe_select\n", nExp);
        return false;
    }
    printf("\n== moe blk.%u  n_embd=%u n_ff=%u experts=%u top-%u + shared ==\n",
           layer, nEmbd, nFf, nExp, nUsed);

    auto x = randomX(nEmbd);
    const size_t rbGE = ggmlRowBytes(GGML_IQ3_XXS, nEmbd);
    const size_t rbDE = ggmlRowBytes(tDE->type, nFf);
    const size_t rbGS = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbDS = ggmlRowBytes(GGML_Q8_0, nFf);

    // ---- CPU reference (mirrors llama.cpp build_moe_ffn semantics) ----
    printf("cpu reference...\n");
    const float* gi  = (const float*)tGI->data;
    const float* gis = (const float*)tGIS->data;
    std::vector<double> logit(nExp);
    for (uint32_t e = 0; e < nExp; e++) {
        double a = 0;
        for (uint32_t k = 0; k < nEmbd; k++) a += (double)gi[(size_t)e * nEmbd + k] * x[k];
        logit[e] = a;
    }
    uint32_t ids[8];
    double wsel[8];
    {
        std::vector<uint32_t> order(nExp);
        for (uint32_t e = 0; e < nExp; e++) order[e] = e;
        std::partial_sort(order.begin(), order.begin() + nUsed, order.end(),
                          [&](uint32_t a, uint32_t b) { return logit[a] > logit[b]; });
        double m = logit[order[0]], sum = 0;
        for (uint32_t i = 0; i < nUsed; i++) {
            ids[i] = order[i];
            wsel[i] = std::exp(logit[order[i]] - m);
            sum += wsel[i];
        }
        for (uint32_t i = 0; i < nUsed; i++) wsel[i] /= sum;
    }
    double sgDot = 0;
    for (uint32_t k = 0; k < nEmbd; k++) sgDot += (double)gis[k] * x[k];
    const double wShared = 1.0 / (1.0 + std::exp(-sgDot));

    auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
    std::vector<float> yref(nEmbd, 0.f), tmpE(nEmbd), tmpF(nFf), hrow(nFf);
    for (uint32_t s = 0; s < nUsed; s++) {
        uint32_t e = ids[s];
        for (uint32_t r = 0; r < nFf; r++) {
            dequant_row_iq3_xxs((const block_iq3_xxs*)(tGE->data + ((size_t)e * nFf + r) * rbGE),
                                tmpE.data(), nEmbd);
            double ga = 0;
            for (uint32_t k = 0; k < nEmbd; k++) ga += (double)tmpE[k] * x[k];
            dequant_row_iq3_xxs((const block_iq3_xxs*)(tUE->data + ((size_t)e * nFf + r) * rbGE),
                                tmpE.data(), nEmbd);
            double ua = 0;
            for (uint32_t k = 0; k < nEmbd; k++) ua += (double)tmpE[k] * x[k];
            hrow[r] = (float)(silu(ga) * ua);
        }
        for (uint32_t o = 0; o < nEmbd; o++) {
            const uint8_t* drow = tDE->data + ((size_t)e * nEmbd + o) * rbDE;
            if (downQ6) dequant_row_q6_K((const block_q6_K*)drow, tmpF.data(), nFf);
            else        dequant_row_iq4_xs((const block_iq4_xs*)drow, tmpF.data(), nFf);
            double a = 0;
            for (uint32_t k = 0; k < nFf; k++) a += (double)tmpF[k] * hrow[k];
            yref[o] += (float)(wsel[s] * a);
        }
    }
    for (uint32_t r = 0; r < nFf; r++) {
        dequant_row_q8_0((const block_q8_0*)(tGS->data + (size_t)r * rbGS), tmpE.data(), nEmbd);
        double ga = 0;
        for (uint32_t k = 0; k < nEmbd; k++) ga += (double)tmpE[k] * x[k];
        dequant_row_q8_0((const block_q8_0*)(tUS->data + (size_t)r * rbGS), tmpE.data(), nEmbd);
        double ua = 0;
        for (uint32_t k = 0; k < nEmbd; k++) ua += (double)tmpE[k] * x[k];
        hrow[r] = (float)(silu(ga) * ua);
    }
    for (uint32_t o = 0; o < nEmbd; o++) {
        dequant_row_q8_0((const block_q8_0*)(tDS->data + (size_t)o * rbDS), tmpF.data(), nFf);
        double a = 0;
        for (uint32_t k = 0; k < nFf; k++) a += (double)tmpF[k] * hrow[k];
        yref[o] += (float)(wShared * a);
    }

    // ---- GPU setup ----
    Pipe pLogits = makePipe(c, "moe_logits.spv", 3, 16);
    Pipe pSelect = makePipe(c, "moe_select.spv", 4, 16);
    Pipe pGuIq3  = makePipe(c, "moe_gateup_iq3.spv", 5, 16);
    Pipe pGuQ8   = makePipe(c, "moe_gateup_q8.spv", 4, 16);
    Pipe pDnIq4  = makePipe(c, downQ6 ? "moe_down_q6k.spv" : "moe_down_iq4.spv", 4, 16);
    Pipe pDnQ8   = makePipe(c, "moe_down_q8.spv", 4, 16);

    const size_t szGI = (size_t)nExp * nEmbd * 4, szGIS = (size_t)nEmbd * 4;
    const size_t szGE = (size_t)nExp * nFf * rbGE, szDE = (size_t)nExp * nEmbd * rbDE;
    const size_t szGS = (size_t)nFf * rbGS, szDS = (size_t)nEmbd * rbDS;
    const size_t szX = (size_t)nEmbd * 4, szY = (size_t)nEmbd * 4;
    const size_t szH = (size_t)(nUsed + 1) * nFf * 4, szSel = 128, szL = (size_t)nExp * 4;

    const VkBufferUsageFlags stor = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    Buf bGI = createBuf(c, szGI, stor, true), bGIS = createBuf(c, szGIS, stor, true);
    Buf bGE = createBuf(c, szGE, stor, true), bUE = createBuf(c, szGE, stor, true);
    Buf bDE = createBuf(c, szDE, stor, true);
    Buf bGS = createBuf(c, szGS, stor, true), bUS = createBuf(c, szGS, stor, true);
    Buf bDS = createBuf(c, szDS, stor, true);
    Buf bX = createBuf(c, szX, stor, true);
    Buf bL = createBuf(c, szL, stor, true), bH = createBuf(c, szH, stor, true);
    Buf bSel = createBuf(c, szSel, stor | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    Buf bY = createBuf(c, szY, stor | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    Buf stage = createBuf(c, std::max(szGE, szDE),
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);

    auto begin = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    };
    auto submitWait = [&]() {
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    };
    void* mapped;
    VK_CHECK(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    auto upload = [&](Buf& dst, const void* src, size_t n) {
        memcpy(mapped, src, n);
        begin();
        VkBufferCopy cp{0, 0, n};
        vkCmdCopyBuffer(c.cb, stage.buf, dst.buf, 1, &cp);
        submitWait();
    };
    upload(bGI, tGI->data, szGI);
    upload(bGIS, tGIS->data, szGIS);
    upload(bGE, tGE->data, szGE);
    upload(bUE, tUE->data, szGE);
    upload(bDE, tDE->data, szDE);
    upload(bGS, tGS->data, szGS);
    upload(bUS, tUS->data, szGS);
    upload(bDS, tDS->data, szDS);
    upload(bX, x.data(), szX);

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 6;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &dpool));

    auto mkSet = [&](Pipe& pp, std::vector<VkBuffer> bufs) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &pp.dsl;
        VkDescriptorSet ds;
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &ai, &ds));
        std::vector<VkDescriptorBufferInfo> dbi(bufs.size());
        std::vector<VkWriteDescriptorSet> wr(bufs.size());
        for (size_t i = 0; i < bufs.size(); i++) {
            dbi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[i].dstSet = ds;
            wr[i].dstBinding = (uint32_t)i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(c.dev, (uint32_t)wr.size(), wr.data(), 0, nullptr);
        return ds;
    };
    VkDescriptorSet sLogits = mkSet(pLogits, {bGI.buf, bX.buf, bL.buf});
    VkDescriptorSet sSelect = mkSet(pSelect, {bL.buf, bGIS.buf, bX.buf, bSel.buf});
    VkDescriptorSet sGuIq3  = mkSet(pGuIq3, {bGE.buf, bUE.buf, bX.buf, bSel.buf, bH.buf});
    VkDescriptorSet sGuQ8   = mkSet(pGuQ8, {bGS.buf, bUS.buf, bX.buf, bH.buf});
    VkDescriptorSet sDnIq4  = mkSet(pDnIq4, {bDE.buf, bH.buf, bSel.buf, bY.buf});
    VkDescriptorSet sDnQ8   = mkSet(pDnQ8, {bDS.buf, bH.buf, bSel.buf, bY.buf});

    struct { uint32_t nEmbd, nFf, nExp, nUsed; } pcv{nEmbd, nFf, nExp, nUsed};
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    auto barrier = [&]() {
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    };
    auto dispatchP = [&](Pipe& pp, VkDescriptorSet ds, uint32_t wgs) {
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(c.cb, pp.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pcv);
        uint32_t gx = std::min(wgs, c.props.limits.maxComputeWorkGroupCount[0]);
        uint32_t gy = (wgs + gx - 1) / gx;
        vkCmdDispatch(c.cb, gx, gy, 1);
    };
    auto sequence = [&]() {
        dispatchP(pLogits, sLogits, nExp);   // independent of selection:
        dispatchP(pGuQ8, sGuQ8, nFf);        // shared expert overlaps router
        barrier();
        dispatchP(pSelect, sSelect, 1);
        barrier();
        dispatchP(pGuIq3, sGuIq3, nUsed * nFf);
        barrier();
        dispatchP(pDnIq4, sDnIq4, nEmbd);
        barrier();
        dispatchP(pDnQ8, sDnQ8, nEmbd);
    };

    // ---- correctness ----
    begin();
    sequence();
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = bY.buf;
    bmb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bmb, 0, nullptr);
    VkBufferCopy cpY{0, 0, szY}, cpS{0, szY, szSel};
    vkCmdCopyBuffer(c.cb, bY.buf, stage.buf, 1, &cpY);
    bmb.buffer = bSel.buf;
    vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bmb, 0, nullptr);
    vkCmdCopyBuffer(c.cb, bSel.buf, stage.buf, 1, &cpS);
    submitWait();

    std::vector<float> ygpu(nEmbd);
    memcpy(ygpu.data(), mapped, szY);
    struct SelOut { uint32_t ids[8]; float w[8]; float wShared; } selGpu;
    memcpy(&selGpu, (const uint8_t*)mapped + szY, sizeof(selGpu));

    bool selOk = true;
    for (uint32_t i = 0; i < nUsed; i++)
        if (selGpu.ids[i] != ids[i] || std::fabs(selGpu.w[i] - wsel[i]) > 1e-4) selOk = false;
    if (std::fabs(selGpu.wShared - wShared) > 1e-4) selOk = false;
    printf("router: experts [");
    for (uint32_t i = 0; i < nUsed; i++) printf("%u%s", selGpu.ids[i], i + 1 < nUsed ? " " : "");
    printf("] shared_gate=%.4f  ->  %s\n", selGpu.wShared, selOk ? "MATCH" : "MISMATCH");

    double rms = 0;
    for (uint32_t o = 0; o < nEmbd; o++) rms += (double)yref[o] * yref[o];
    rms = std::sqrt(rms / nEmbd);
    double denomFloor = std::max(1e-3, 1e-3 * rms);
    double maxRel = 0;
    uint32_t bad = 0;
    for (uint32_t o = 0; o < nEmbd; o++) {
        double rel = std::fabs((double)ygpu[o] - yref[o]) /
                     std::max(denomFloor, (double)std::fabs(yref[o]));
        maxRel = std::max(maxRel, rel);
        if (rel > 1e-2 && bad++ < 5) printf("  y[%u]: gpu=%g ref=%g\n", o, ygpu[o], yref[o]);
    }
    bool pass = selOk && bad == 0;
    printf("correctness: max_rel_err = %.3g  ->  %s\n", maxRel, pass ? "PASS" : "FAIL");

    // ---- benchmark ----
    if (pass && c.hasTimestamps && iters > 0) {
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VkQueryPool qp;
        VK_CHECK(vkCreateQueryPool(c.dev, &qpci, nullptr, &qp));
        auto runBench = [&]() {
            begin();
            vkCmdResetQueryPool(c.cb, qp, 0, 2);
            vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            for (uint32_t i = 0; i < iters; i++) {
                sequence();
                barrier();
            }
            vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            submitWait();
        };
        runBench();
        runBench();
        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(c.dev, qp, 0, 2, sizeof(ts), ts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double ns = (double)(ts[1] - ts[0]) * c.props.limits.timestampPeriod / iters;
        // weights actually touched per token (selected experts only)
        double bytes = (double)szGI + szGIS +
                       (double)nUsed * (2.0 * nFf * rbGE + (double)nEmbd * rbDE) +
                       2.0 * (double)nFf * rbGS + (double)nEmbd * rbDS;
        printf("gpu: %8.1f µs/layer-moe | %6.1f GB/s (active weights %.1f MiB) | 6 dispatches, 1 submit\n",
               ns / 1e3, bytes / ns, bytes / (1 << 20));
        printf("     40 layers -> %.2f ms/token MoE-FFN share\n", ns * 40 / 1e6);
        vkDestroyQueryPool(c.dev, qp, nullptr);
    }

    vkUnmapMemory(c.dev, stage.mem);
    vkDestroyDescriptorPool(c.dev, dpool, nullptr);
    for (Pipe* pp : {&pLogits, &pSelect, &pGuIq3, &pGuQ8, &pDnIq4, &pDnQ8}) destroyPipe(c, *pp);
    for (Buf* b : {&bGI, &bGIS, &bGE, &bUE, &bDE, &bGS, &bUS, &bDS, &bX, &bL, &bH, &bSel, &bY, &stage})
        destroyBuf(c, *b);
    return pass;
}

// ---------- MoE tensor set + CPU reference (shared by moe/block modes) ----------

struct MoeT {
    const GgufTensor *gi, *gis, *ge, *ue, *de, *gs, *us, *ds;
    uint32_t nEmbd, nFf, nExp, nUsed;
    bool downQ6;
    size_t rbGE, rbDE, rbGS, rbDS;
};

static bool loadMoeT(Gguf& g, uint32_t layer, MoeT& m) {
    char nb[128];
    auto T = [&](const char* suffix) -> const GgufTensor* {
        snprintf(nb, sizeof nb, "blk.%u.%s", layer, suffix);
        return g.find(nb);
    };
    m.gi = T("ffn_gate_inp.weight");
    m.gis = T("ffn_gate_inp_shexp.weight");
    m.ge = T("ffn_gate_exps.weight");
    m.ue = T("ffn_up_exps.weight");
    m.de = T("ffn_down_exps.weight");
    m.gs = T("ffn_gate_shexp.weight");
    m.us = T("ffn_up_shexp.weight");
    m.ds = T("ffn_down_shexp.weight");
    if (!m.gi || !m.gis || !m.ge || !m.ue || !m.de || !m.gs || !m.us || !m.ds) {
        fprintf(stderr, "layer %u: missing MoE tensors\n", layer);
        return false;
    }
    if (m.gi->type != GGML_F32 || m.gis->type != GGML_F32 ||
        m.ge->type != GGML_IQ3_XXS || m.ue->type != GGML_IQ3_XXS ||
        (m.de->type != GGML_IQ4_XS && m.de->type != GGML_Q6_K) ||
        m.gs->type != GGML_Q8_0 || m.us->type != GGML_Q8_0 || m.ds->type != GGML_Q8_0) {
        fprintf(stderr, "layer %u: unexpected MoE tensor types\n", layer);
        return false;
    }
    m.nEmbd = (uint32_t)m.ge->ne[0];
    m.nFf = (uint32_t)m.ge->ne[1];
    m.nExp = (uint32_t)m.ge->ne[2];
    m.nUsed = 8;
    m.downQ6 = m.de->type == GGML_Q6_K;
    m.rbGE = ggmlRowBytes(GGML_IQ3_XXS, m.nEmbd);
    m.rbDE = ggmlRowBytes(m.de->type, m.nFf);
    m.rbGS = ggmlRowBytes(GGML_Q8_0, m.nEmbd);
    m.rbDS = ggmlRowBytes(GGML_Q8_0, m.nFf);
    return true;
}

struct MoeRefSel {
    uint32_t ids[8];
    double w[8];
    double wShared;
};

static void moeCpuRef(const MoeT& m, const float* x, float* yout, MoeRefSel* selOut) {
    std::vector<double> logit(m.nExp);
    const float* gi = (const float*)m.gi->data;
    const float* gis = (const float*)m.gis->data;
    for (uint32_t e = 0; e < m.nExp; e++) {
        double a = 0;
        for (uint32_t k = 0; k < m.nEmbd; k++) a += (double)gi[(size_t)e * m.nEmbd + k] * x[k];
        logit[e] = a;
    }
    MoeRefSel sel;
    {
        std::vector<uint32_t> order(m.nExp);
        for (uint32_t e = 0; e < m.nExp; e++) order[e] = e;
        std::partial_sort(order.begin(), order.begin() + m.nUsed, order.end(),
                          [&](uint32_t a, uint32_t b) { return logit[a] > logit[b]; });
        double mx = logit[order[0]], sum = 0;
        for (uint32_t i = 0; i < m.nUsed; i++) {
            sel.ids[i] = order[i];
            sel.w[i] = std::exp(logit[order[i]] - mx);
            sum += sel.w[i];
        }
        for (uint32_t i = 0; i < m.nUsed; i++) sel.w[i] /= sum;
    }
    double sg = 0;
    for (uint32_t k = 0; k < m.nEmbd; k++) sg += (double)gis[k] * x[k];
    sel.wShared = 1.0 / (1.0 + std::exp(-sg));

    auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
    std::vector<float> tmpE(m.nEmbd), tmpF(m.nFf), hrow(m.nFf);
    for (uint32_t o = 0; o < m.nEmbd; o++) yout[o] = 0.f;
    for (uint32_t s = 0; s < m.nUsed; s++) {
        uint32_t e = sel.ids[s];
        for (uint32_t r = 0; r < m.nFf; r++) {
            dequant_row_iq3_xxs((const block_iq3_xxs*)(m.ge->data + ((size_t)e * m.nFf + r) * m.rbGE),
                                tmpE.data(), m.nEmbd);
            double ga = 0;
            for (uint32_t k = 0; k < m.nEmbd; k++) ga += (double)tmpE[k] * x[k];
            dequant_row_iq3_xxs((const block_iq3_xxs*)(m.ue->data + ((size_t)e * m.nFf + r) * m.rbGE),
                                tmpE.data(), m.nEmbd);
            double ua = 0;
            for (uint32_t k = 0; k < m.nEmbd; k++) ua += (double)tmpE[k] * x[k];
            hrow[r] = (float)(silu(ga) * ua);
        }
        for (uint32_t o = 0; o < m.nEmbd; o++) {
            const uint8_t* drow = m.de->data + ((size_t)e * m.nEmbd + o) * m.rbDE;
            if (m.downQ6) dequant_row_q6_K((const block_q6_K*)drow, tmpF.data(), m.nFf);
            else          dequant_row_iq4_xs((const block_iq4_xs*)drow, tmpF.data(), m.nFf);
            double a = 0;
            for (uint32_t k = 0; k < m.nFf; k++) a += (double)tmpF[k] * hrow[k];
            yout[o] += (float)(sel.w[s] * a);
        }
    }
    for (uint32_t r = 0; r < m.nFf; r++) {
        dequant_row_q8_0((const block_q8_0*)(m.gs->data + (size_t)r * m.rbGS), tmpE.data(), m.nEmbd);
        double ga = 0;
        for (uint32_t k = 0; k < m.nEmbd; k++) ga += (double)tmpE[k] * x[k];
        dequant_row_q8_0((const block_q8_0*)(m.us->data + (size_t)r * m.rbGS), tmpE.data(), m.nEmbd);
        double ua = 0;
        for (uint32_t k = 0; k < m.nEmbd; k++) ua += (double)tmpE[k] * x[k];
        hrow[r] = (float)(silu(ga) * ua);
    }
    for (uint32_t o = 0; o < m.nEmbd; o++) {
        dequant_row_q8_0((const block_q8_0*)(m.ds->data + (size_t)o * m.rbDS), tmpF.data(), m.nFf);
        double a = 0;
        for (uint32_t k = 0; k < m.nFf; k++) a += (double)tmpF[k] * hrow[k];
        yout[o] += (float)(sel.wShared * a);
    }
    if (selOut) *selOut = sel;
}

// M5: one full deltanet decode block (attn_norm -> qkv/z/alpha/beta -> conv+silu
// -> l2norm -> delta rule -> gated norm -> ssm_out -> residual/post-norm ->
// MoE-FFN -> residual) as ONE queue submission of 17 dispatches. Conv and
// delta states persist on-GPU across tokens, mirrored on CPU for validation.
static bool caseBlock(VkCtx& c, uint32_t layer, uint32_t nTok, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    char nb[128];
    auto T = [&](const char* suffix) -> const GgufTensor* {
        snprintf(nb, sizeof nb, "blk.%u.%s", layer, suffix);
        return g.find(nb);
    };
    const GgufTensor* tAv = T("ssm_a");
    if (!tAv) {
        fprintf(stderr, "blk.%u is a full-attention layer (every 4th); pick a deltanet layer\n", layer);
        return false;
    }
    const GgufTensor* tANorm = T("attn_norm.weight");
    const GgufTensor* tQkvW  = T("attn_qkv.weight");
    const GgufTensor* tZW    = T("attn_gate.weight");
    const GgufTensor* tAl    = T("ssm_alpha.weight");
    const GgufTensor* tBe    = T("ssm_beta.weight");
    const GgufTensor* tDt    = T("ssm_dt.bias");
    const GgufTensor* tKer   = T("ssm_conv1d.weight");
    if (!tKer) tKer = T("ssm_conv1d");
    const GgufTensor* tSN    = T("ssm_norm.weight");
    const GgufTensor* tOutW  = T("ssm_out.weight");
    const GgufTensor* tPN    = T("post_attention_norm.weight");
    if (!tANorm || !tQkvW || !tZW || !tAl || !tBe || !tDt || !tKer || !tSN || !tOutW || !tPN) {
        fprintf(stderr, "blk.%u: missing deltanet tensors\n", layer);
        return false;
    }
    if (tQkvW->type != GGML_Q8_0 || tZW->type != GGML_Q8_0 || tOutW->type != GGML_Q8_0) {
        fprintf(stderr, "blk.%u: unexpected projection types\n", layer);
        return false;
    }
    MoeT moe;
    if (!loadMoeT(g, layer, moe)) return false;

    const uint32_t nEmbd = (uint32_t)tANorm->ne[0];
    const uint32_t chQkv = (uint32_t)tQkvW->ne[1];
    const uint32_t dIn   = (uint32_t)tZW->ne[1];
    const uint32_t hV    = (uint32_t)tAv->ne[0];
    const uint32_t dS    = (uint32_t)tSN->ne[0];
    const uint32_t hK    = (chQkv - dIn) / dS / 2;
    const float eps = 1e-6f;
    printf("\n== block blk.%u (deltanet)  n_embd=%u qkv=%u d_inner=%u v-heads=%u k-heads=%u d_state=%u ==\n",
           layer, nEmbd, chQkv, dIn, hV, hK, dS);
    if (moe.nEmbd != nEmbd || dIn != hV * dS) {
        fprintf(stderr, "dimension mismatch\n");
        return false;
    }

    const size_t rbQ8e = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbQ8i = ggmlRowBytes(GGML_Q8_0, dIn);

    // ---- pipes ----
    Pipe pRms   = makePipe(c, "rmsnorm.spv", 3, 8);
    Pipe pGemvA = makePipe(c, "gemv_q8_0.spv", 3, 8, 64);    // K = n_embd
    Pipe pAb    = makePipe(c, "dn_ab.spv", 6, 8);
    Pipe pConv  = makePipe(c, "dn_conv.spv", 4, 4);
    Pipe pQk    = makePipe(c, "dn_qknorm.spv", 1, 8);
    Pipe pStep  = makePipe(c, "dn_step.spv", 4, 12);
    Pipe pGate  = makePipe(c, "dn_gate.spv", 4, 12);
    Pipe pGemvO = makePipe(c, "gemv_q8_0.spv", 3, 8, 128);   // K = d_inner
    Pipe pAddN  = makePipe(c, "add_rmsnorm.spv", 5, 8);
    Pipe pAdd   = makePipe(c, "vec_add.spv", 3, 4);
    Pipe pMoeL  = makePipe(c, "moe_logits.spv", 3, 16);
    Pipe pMoeS  = makePipe(c, "moe_select.spv", 4, 16);
    Pipe pMoeGU = makePipe(c, "moe_gateup_iq3.spv", 5, 16);
    Pipe pMoeGUs = makePipe(c, "moe_gateup_q8.spv", 4, 16);
    Pipe pMoeDn = makePipe(c, moe.downQ6 ? "moe_down_q6k.spv" : "moe_down_iq4.spv", 4, 16);
    Pipe pMoeDns = makePipe(c, "moe_down_q8.spv", 4, 16);

    // ---- buffers ----
    const VkBufferUsageFlags stor = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags storSrc = stor | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const size_t szMGE = (size_t)moe.nExp * moe.nFf * moe.rbGE;
    const size_t szMDE = (size_t)moe.nExp * moe.nEmbd * moe.rbDE;
    Buf bANorm = createBuf(c, nEmbd * 4, stor, true);
    Buf bQkvW = createBuf(c, (size_t)chQkv * rbQ8e, stor, true);
    Buf bZW = createBuf(c, (size_t)dIn * rbQ8e, stor, true);
    Buf bAlW = createBuf(c, (size_t)hV * nEmbd * 4, stor, true);
    Buf bBeW = createBuf(c, (size_t)hV * nEmbd * 4, stor, true);
    Buf bDt = createBuf(c, hV * 4, stor, true);
    Buf bAv = createBuf(c, hV * 4, stor, true);
    Buf bKer = createBuf(c, (size_t)chQkv * 4 * 4, stor, true);
    Buf bSN = createBuf(c, dS * 4, stor, true);
    Buf bOutW = createBuf(c, (size_t)nEmbd * rbQ8i, stor, true);
    Buf bPN = createBuf(c, nEmbd * 4, stor, true);
    Buf bMGI = createBuf(c, (size_t)moe.nExp * nEmbd * 4, stor, true);
    Buf bMGIS = createBuf(c, nEmbd * 4, stor, true);
    Buf bMGE = createBuf(c, szMGE, stor, true);
    Buf bMUE = createBuf(c, szMGE, stor, true);
    Buf bMDE = createBuf(c, szMDE, stor, true);
    Buf bMGS = createBuf(c, (size_t)moe.nFf * moe.rbGS, stor, true);
    Buf bMUS = createBuf(c, (size_t)moe.nFf * moe.rbGS, stor, true);
    Buf bMDS = createBuf(c, (size_t)moe.nEmbd * moe.rbDS, stor, true);

    Buf bXin = createBuf(c, nEmbd * 4, stor, true);
    Buf bXn = createBuf(c, nEmbd * 4, stor, true);
    Buf bQkv = createBuf(c, chQkv * 4, stor, true);
    Buf bZ = createBuf(c, dIn * 4, stor, true);
    Buf bGb = createBuf(c, 2 * hV * 4, stor, true);
    Buf bConvSt = createBuf(c, (size_t)chQkv * 3 * 4, storSrc, true);
    Buf bConvOut = createBuf(c, chQkv * 4, stor, true);
    Buf bS = createBuf(c, (size_t)hV * dS * dS * 4, storSrc, true);
    Buf bO = createBuf(c, dIn * 4, stor, true);
    Buf bAtt = createBuf(c, dIn * 4, stor, true);
    Buf bAttnOut = createBuf(c, nEmbd * 4, stor, true);
    Buf bY = createBuf(c, nEmbd * 4, stor, true);
    Buf bXn2 = createBuf(c, nEmbd * 4, stor, true);
    Buf bML = createBuf(c, moe.nExp * 4, stor, true);
    Buf bMH = createBuf(c, (size_t)(moe.nUsed + 1) * moe.nFf * 4, stor, true);
    Buf bMSel = createBuf(c, 128, stor, true);
    Buf bMY = createBuf(c, nEmbd * 4, stor, true);
    Buf bOut = createBuf(c, nEmbd * 4, storSrc, true);
    Buf stage = createBuf(c, std::max(szMDE, szMGE),
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);

    auto begin = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    };
    auto submitWait = [&]() {
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    };
    void* mapped;
    VK_CHECK(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    auto upload = [&](Buf& dst, const void* src, size_t n) {
        if (src) memcpy(mapped, src, n);
        else memset(mapped, 0, n);
        begin();
        VkBufferCopy cp{0, 0, n};
        vkCmdCopyBuffer(c.cb, stage.buf, dst.buf, 1, &cp);
        submitWait();
    };
    upload(bANorm, tANorm->data, nEmbd * 4);
    upload(bQkvW, tQkvW->data, (size_t)chQkv * rbQ8e);
    upload(bZW, tZW->data, (size_t)dIn * rbQ8e);
    upload(bAlW, tAl->data, (size_t)hV * nEmbd * 4);
    upload(bBeW, tBe->data, (size_t)hV * nEmbd * 4);
    upload(bDt, tDt->data, hV * 4);
    upload(bAv, tAv->data, hV * 4);
    upload(bKer, tKer->data, (size_t)chQkv * 4 * 4);
    upload(bSN, tSN->data, dS * 4);
    upload(bOutW, tOutW->data, (size_t)nEmbd * rbQ8i);
    upload(bPN, tPN->data, nEmbd * 4);
    upload(bMGI, moe.gi->data, (size_t)moe.nExp * nEmbd * 4);
    upload(bMGIS, moe.gis->data, nEmbd * 4);
    upload(bMGE, moe.ge->data, szMGE);
    upload(bMUE, moe.ue->data, szMGE);
    upload(bMDE, moe.de->data, szMDE);
    upload(bMGS, moe.gs->data, (size_t)moe.nFf * moe.rbGS);
    upload(bMUS, moe.us->data, (size_t)moe.nFf * moe.rbGS);
    upload(bMDS, moe.ds->data, (size_t)moe.nEmbd * moe.rbDS);
    upload(bConvSt, nullptr, (size_t)chQkv * 3 * 4);   // zero state
    upload(bS, nullptr, (size_t)hV * dS * dS * 4);

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 20;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &dpool));
    auto mkSet = [&](Pipe& pp, std::vector<VkBuffer> bufs) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &pp.dsl;
        VkDescriptorSet ds;
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &ai, &ds));
        std::vector<VkDescriptorBufferInfo> dbi(bufs.size());
        std::vector<VkWriteDescriptorSet> wr(bufs.size());
        for (size_t i = 0; i < bufs.size(); i++) {
            dbi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[i].dstSet = ds;
            wr[i].dstBinding = (uint32_t)i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(c.dev, (uint32_t)wr.size(), wr.data(), 0, nullptr);
        return ds;
    };
    VkDescriptorSet sRms = mkSet(pRms, {bXin.buf, bANorm.buf, bXn.buf});
    VkDescriptorSet sQkv = mkSet(pGemvA, {bQkvW.buf, bXn.buf, bQkv.buf});
    VkDescriptorSet sZ = mkSet(pGemvA, {bZW.buf, bXn.buf, bZ.buf});
    VkDescriptorSet sAb = mkSet(pAb, {bXn.buf, bAlW.buf, bBeW.buf, bDt.buf, bAv.buf, bGb.buf});
    VkDescriptorSet sConv = mkSet(pConv, {bConvSt.buf, bQkv.buf, bKer.buf, bConvOut.buf});
    VkDescriptorSet sQk = mkSet(pQk, {bConvOut.buf});
    VkDescriptorSet sStep = mkSet(pStep, {bConvOut.buf, bGb.buf, bS.buf, bO.buf});
    VkDescriptorSet sGate = mkSet(pGate, {bO.buf, bSN.buf, bZ.buf, bAtt.buf});
    VkDescriptorSet sOut = mkSet(pGemvO, {bOutW.buf, bAtt.buf, bAttnOut.buf});
    VkDescriptorSet sAddN = mkSet(pAddN, {bXin.buf, bAttnOut.buf, bPN.buf, bY.buf, bXn2.buf});
    VkDescriptorSet sMoeL = mkSet(pMoeL, {bMGI.buf, bXn2.buf, bML.buf});
    VkDescriptorSet sMoeS = mkSet(pMoeS, {bML.buf, bMGIS.buf, bXn2.buf, bMSel.buf});
    VkDescriptorSet sMoeGU = mkSet(pMoeGU, {bMGE.buf, bMUE.buf, bXn2.buf, bMSel.buf, bMH.buf});
    VkDescriptorSet sMoeGUs = mkSet(pMoeGUs, {bMGS.buf, bMUS.buf, bXn2.buf, bMH.buf});
    VkDescriptorSet sMoeDn = mkSet(pMoeDn, {bMDE.buf, bMH.buf, bMSel.buf, bMY.buf});
    VkDescriptorSet sMoeDns = mkSet(pMoeDns, {bMDS.buf, bMH.buf, bMSel.buf, bMY.buf});
    VkDescriptorSet sAdd = mkSet(pAdd, {bY.buf, bMY.buf, bOut.buf});

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    auto barrier = [&]() {
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    auto dispatchB = [&](Pipe& pp, VkDescriptorSet ds, uint32_t wgs, const void* pc, uint32_t pcSize) {
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(c.cb, pp.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
        uint32_t gx = std::min(wgs, c.props.limits.maxComputeWorkGroupCount[0]);
        uint32_t gy = (wgs + gx - 1) / gx;
        vkCmdDispatch(c.cb, gx, gy, 1);
    };

    struct { uint32_t n; float e; } pcRms{nEmbd, eps}, pcAddN{nEmbd, eps};
    struct { uint32_t m, k; } pcQkv{chQkv, nEmbd}, pcZ{dIn, nEmbd}, pcOut{nEmbd, dIn};
    struct { uint32_t n, h; } pcAb{nEmbd, hV};
    struct { uint32_t ch; } pcConv{chQkv};
    struct { uint32_t d; float e; } pcQk{dS, eps};
    struct { uint32_t d, hk, hv; } pcStep{dS, hK, hV};
    struct { uint32_t d, hv; float e; } pcGate{dS, hV, eps};
    struct { uint32_t n; } pcAdd{nEmbd};
    struct { uint32_t a, b, cc, d; } pcv{moe.nEmbd, moe.nFf, moe.nExp, moe.nUsed};

    auto sequence = [&]() {
        dispatchB(pRms, sRms, 1, &pcRms, 8);
        barrier();
        dispatchB(pGemvA, sQkv, (chQkv + 3) / 4, &pcQkv, 8);
        dispatchB(pGemvA, sZ, (dIn + 3) / 4, &pcZ, 8);
        dispatchB(pAb, sAb, 2 * hV, &pcAb, 8);
        barrier();
        dispatchB(pConv, sConv, (chQkv + 255) / 256, &pcConv, 4);
        barrier();
        dispatchB(pQk, sQk, 2 * hK, &pcQk, 8);
        barrier();
        dispatchB(pStep, sStep, hV, &pcStep, 12);
        barrier();
        dispatchB(pGate, sGate, hV, &pcGate, 12);
        barrier();
        dispatchB(pGemvO, sOut, (nEmbd + 1) / 2, &pcOut, 8);
        barrier();
        dispatchB(pAddN, sAddN, 1, &pcAddN, 8);
        barrier();
        dispatchB(pMoeL, sMoeL, moe.nExp, &pcv, 16);
        dispatchB(pMoeGUs, sMoeGUs, moe.nFf, &pcv, 16);
        barrier();
        dispatchB(pMoeS, sMoeS, 1, &pcv, 16);
        barrier();
        dispatchB(pMoeGU, sMoeGU, moe.nUsed * moe.nFf, &pcv, 16);
        barrier();
        dispatchB(pMoeDn, sMoeDn, moe.nEmbd, &pcv, 16);
        barrier();
        dispatchB(pMoeDns, sMoeDns, moe.nEmbd, &pcv, 16);
        barrier();
        dispatchB(pAdd, sAdd, 1, &pcAdd, 4);
    };

    // ---- CPU mirror state ----
    std::vector<float> convSt((size_t)chQkv * 3, 0.f);
    std::vector<float> S((size_t)hV * dS * dS, 0.f);
    const float* anW = (const float*)tANorm->data;
    const float* pnW = (const float*)tPN->data;
    const float* alW = (const float*)tAl->data;
    const float* beW = (const float*)tBe->data;
    const float* dtB = (const float*)tDt->data;
    const float* aV = (const float*)tAv->data;
    const float* ker = (const float*)tKer->data;
    const float* snW = (const float*)tSN->data;

    std::vector<float> x(nEmbd), xn(nEmbd), qkv(chQkv), z(dIn), gvec(hV), bvec(hV),
        convOut(chQkv), o(dIn), att(dIn), attnOut(nEmbd), y(nEmbd), xn2(nEmbd),
        moeOut(nEmbd), refOut(nEmbd), gpuOut(nEmbd), tmpK(std::max(nEmbd, dIn));
    auto q8Gemv = [&](const GgufTensor* t, const std::vector<float>& xin,
                      std::vector<float>& out, uint32_t M, uint32_t K) {
        size_t rb = ggmlRowBytes(GGML_Q8_0, K);
        for (uint32_t r = 0; r < M; r++) {
            dequant_row_q8_0((const block_q8_0*)(t->data + (size_t)r * rb), tmpK.data(), K);
            double a = 0;
            for (uint32_t k = 0; k < K; k++) a += (double)tmpK[k] * xin[k];
            out[r] = (float)a;
        }
    };
    auto rmsRef = [&](const std::vector<float>& in, const float* w, std::vector<float>& out) {
        double ss = 0;
        for (uint32_t i = 0; i < nEmbd; i++) ss += (double)in[i] * in[i];
        double sc = 1.0 / std::sqrt(ss / nEmbd + eps);
        for (uint32_t i = 0; i < nEmbd; i++) out[i] = (float)(in[i] * sc * w[i]);
    };

    std::mt19937 xr(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    bool pass = true;

    printf("cpu reference + gpu, %u tokens...\n", nTok);
    for (uint32_t t = 0; t < nTok; t++) {
        for (auto& v : x) v = nd(xr);

        // --- CPU reference ---
        rmsRef(x, anW, xn);
        q8Gemv(tQkvW, xn, qkv, chQkv, nEmbd);
        q8Gemv(tZW, xn, z, dIn, nEmbd);
        for (uint32_t h = 0; h < hV; h++) {
            double da = 0, db = 0;
            for (uint32_t k = 0; k < nEmbd; k++) {
                da += (double)alW[(size_t)h * nEmbd + k] * xn[k];
                db += (double)beW[(size_t)h * nEmbd + k] * xn[k];
            }
            double v = da + dtB[h];
            double sp = v > 20.0 ? v : std::log1p(std::exp(v));
            gvec[h] = (float)(aV[h] * sp);
            bvec[h] = (float)(1.0 / (1.0 + std::exp(-db)));
        }
        for (uint32_t ch = 0; ch < chQkv; ch++) {
            float* st = &convSt[(size_t)ch * 3];
            double v = (double)st[0] * ker[ch * 4] + (double)st[1] * ker[ch * 4 + 1] +
                       (double)st[2] * ker[ch * 4 + 2] + (double)qkv[ch] * ker[ch * 4 + 3];
            convOut[ch] = (float)(v / (1.0 + std::exp(-v)));
            st[0] = st[1];
            st[1] = st[2];
            st[2] = qkv[ch];
        }
        for (uint32_t w = 0; w < 2 * hK; w++) {
            double ss = 0;
            for (uint32_t i = 0; i < dS; i++) {
                double v = convOut[(size_t)w * dS + i];
                ss += v * v;
            }
            double sc = 1.0 / std::max(std::sqrt(ss), (double)eps);
            for (uint32_t i = 0; i < dS; i++) convOut[(size_t)w * dS + i] *= (float)sc;
        }
        for (uint32_t h = 0; h < hV; h++) {
            uint32_t kh = h % hK;
            const float* qh = &convOut[(size_t)kh * dS];
            const float* khp = &convOut[(size_t)(hK + kh) * dS];
            const float* vh = &convOut[(size_t)(2 * hK) * dS + (size_t)h * dS];
            float decay = std::exp(gvec[h]);
            float beta = bvec[h];
            float qsc = 1.0f / std::sqrt((float)dS);
            for (uint32_t j = 0; j < dS; j++) {
                float* row = &S[((size_t)h * dS + j) * dS];
                double sk = 0;
                for (uint32_t i = 0; i < dS; i++) sk += (double)row[i] * khp[i];
                sk *= decay;
                float dj = beta * (vh[j] - (float)sk);
                double oj = 0;
                for (uint32_t i = 0; i < dS; i++) {
                    float sn = row[i] * decay + khp[i] * dj;
                    row[i] = sn;
                    oj += (double)sn * (qh[i] * qsc);
                }
                o[(size_t)h * dS + j] = (float)oj;
            }
        }
        for (uint32_t h = 0; h < hV; h++) {
            double ss = 0;
            for (uint32_t j = 0; j < dS; j++) {
                double v = o[(size_t)h * dS + j];
                ss += v * v;
            }
            double sc = 1.0 / std::sqrt(ss / dS + eps);
            for (uint32_t j = 0; j < dS; j++) {
                double zv = z[(size_t)h * dS + j];
                att[(size_t)h * dS + j] =
                    (float)(o[(size_t)h * dS + j] * sc * snW[j] * (zv / (1.0 + std::exp(-zv))));
            }
        }
        q8Gemv(tOutW, att, attnOut, nEmbd, dIn);
        for (uint32_t i = 0; i < nEmbd; i++) y[i] = x[i] + attnOut[i];
        rmsRef(y, pnW, xn2);
        moeCpuRef(moe, xn2.data(), moeOut.data(), nullptr);
        for (uint32_t i = 0; i < nEmbd; i++) refOut[i] = y[i] + moeOut[i];

        // --- GPU ---
        upload(bXin, x.data(), nEmbd * 4);
        begin();
        sequence();
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = bOut.buf;
        bmb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bmb, 0, nullptr);
        VkBufferCopy cp{0, 0, nEmbd * 4};
        vkCmdCopyBuffer(c.cb, bOut.buf, stage.buf, 1, &cp);
        submitWait();
        memcpy(gpuOut.data(), mapped, nEmbd * 4);

        double rms = 0;
        for (uint32_t i = 0; i < nEmbd; i++) rms += (double)refOut[i] * refOut[i];
        rms = std::sqrt(rms / nEmbd);
        double floorD = std::max(1e-3, 1e-3 * rms);
        double maxRel = 0;
        uint32_t bad = 0;
        for (uint32_t i = 0; i < nEmbd; i++) {
            double rel = std::fabs((double)gpuOut[i] - refOut[i]) /
                         std::max(floorD, (double)std::fabs(refOut[i]));
            maxRel = std::max(maxRel, rel);
            if (rel > 1e-2 && bad++ < 3)
                printf("  tok%u y[%u]: gpu=%g ref=%g\n", t, i, gpuOut[i], refOut[i]);
        }
        pass &= bad == 0;
        printf("token %u: max_rel_err = %.3g  ->  %s\n", t, maxRel, bad == 0 ? "PASS" : "FAIL");
    }

    // delta state drift check after nTok tokens
    {
        begin();
        VkBufferCopy cp{0, 0, (size_t)hV * dS * dS * 4};
        vkCmdCopyBuffer(c.cb, bS.buf, stage.buf, 1, &cp);
        submitWait();
        const float* sg = (const float*)mapped;
        double maxAbs = 0, rmsS = 0;
        for (size_t i = 0; i < S.size(); i++) {
            maxAbs = std::max(maxAbs, (double)std::fabs(sg[i] - S[i]));
            rmsS += (double)S[i] * S[i];
        }
        rmsS = std::sqrt(rmsS / S.size());
        printf("delta state after %u tokens: max_abs_diff = %.3g (state rms %.3g)\n", nTok, maxAbs, rmsS);
    }

    if (pass && c.hasTimestamps && iters > 0) {
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VkQueryPool qp;
        VK_CHECK(vkCreateQueryPool(c.dev, &qpci, nullptr, &qp));
        auto runBench = [&]() {
            begin();
            vkCmdResetQueryPool(c.cb, qp, 0, 2);
            vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            for (uint32_t i = 0; i < iters; i++) {
                sequence();
                barrier();
            }
            vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            submitWait();
        };
        runBench();
        runBench();
        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(c.dev, qp, 0, 2, sizeof(ts), ts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double ns = (double)(ts[1] - ts[0]) * c.props.limits.timestampPeriod / iters;
        printf("gpu: %8.1f µs/block | 17 dispatches, 1 submit\n", ns / 1e3);
        printf("     30 deltanet blocks -> %.2f ms/token (+10 full-attn blocks, LM head 0.58 ms)\n",
               ns * 30 / 1e6);
        vkDestroyQueryPool(c.dev, qp, nullptr);
    }

    vkUnmapMemory(c.dev, stage.mem);
    vkDestroyDescriptorPool(c.dev, dpool, nullptr);
    for (Pipe* pp : {&pRms, &pGemvA, &pAb, &pConv, &pQk, &pStep, &pGate, &pGemvO, &pAddN, &pAdd,
                     &pMoeL, &pMoeS, &pMoeGU, &pMoeGUs, &pMoeDn, &pMoeDns})
        destroyPipe(c, *pp);
    for (Buf* b : {&bANorm, &bQkvW, &bZW, &bAlW, &bBeW, &bDt, &bAv, &bKer, &bSN, &bOutW, &bPN,
                   &bMGI, &bMGIS, &bMGE, &bMUE, &bMDE, &bMGS, &bMUS, &bMDS, &bXin, &bXn, &bQkv,
                   &bZ, &bGb, &bConvSt, &bConvOut, &bS, &bO, &bAtt, &bAttnOut, &bY, &bXn2, &bML,
                   &bMH, &bMSel, &bMY, &bOut, &stage})
        destroyBuf(c, *b);
    return pass;
}

// M6a: one full-attention decode block (layers 3,7,...,39) as one submission:
// attn_norm -> q(+gate)/k/v projections -> per-head RMS + partial NeoX rope
// (IMROPE degenerates to this for text) -> KV-cache attention + sigmoid gate
// -> wo -> residual/post-norm -> MoE -> residual. KV cache persists on GPU.
static const float kFreqBase = 1e7f;  // qwen35moe.rope.freq_base

static bool caseABlock(VkCtx& c, uint32_t layer, uint32_t nTok, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    char nb[128];
    auto T = [&](const char* suffix) -> const GgufTensor* {
        snprintf(nb, sizeof nb, "blk.%u.%s", layer, suffix);
        return g.find(nb);
    };
    if (T("ssm_a")) {
        fprintf(stderr, "blk.%u is a deltanet layer; use `qk block`\n", layer);
        return false;
    }
    const GgufTensor* tANorm = T("attn_norm.weight");
    const GgufTensor* tWq = T("attn_q.weight");
    const GgufTensor* tWk = T("attn_k.weight");
    const GgufTensor* tWv = T("attn_v.weight");
    const GgufTensor* tQN = T("attn_q_norm.weight");
    const GgufTensor* tKN = T("attn_k_norm.weight");
    const GgufTensor* tWo = T("attn_output.weight");
    const GgufTensor* tPN = T("post_attention_norm.weight");
    if (!tANorm || !tWq || !tWk || !tWv || !tQN || !tKN || !tWo || !tPN) {
        fprintf(stderr, "blk.%u: missing attention tensors\n", layer);
        return false;
    }
    MoeT moe;
    if (!loadMoeT(g, layer, moe)) return false;

    const uint32_t nEmbd = (uint32_t)tANorm->ne[0];
    const uint32_t dh    = (uint32_t)tQN->ne[0];            // 256
    const uint32_t hQ    = (uint32_t)tWq->ne[1] / (2 * dh); // 16 (q+gate interleaved)
    const uint32_t hKV   = (uint32_t)tWk->ne[1] / dh;       // 2
    const uint32_t nRot  = 64;
    const uint32_t qfN   = hQ * 2 * dh, kvN = hKV * dh, atN = hQ * dh;
    const uint32_t tmax  = 128;
    const float eps = 1e-6f;
    printf("\n== ablock blk.%u (full attn)  n_embd=%u heads=%ux%u kv=%u rot=%u ==\n",
           layer, nEmbd, hQ, dh, hKV, nRot);

    const size_t rbQ8e = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbQ8a = ggmlRowBytes(GGML_Q8_0, atN);

    Pipe pRms   = makePipe(c, "rmsnorm.spv", 3, 8);
    Pipe pGemvA = makePipe(c, "gemv_q8_0.spv", 3, 8, 64);
    Pipe pPrep  = makePipe(c, "fa_prep.spv", 9, 32);
    Pipe pAttn  = makePipe(c, "fa_attn.spv", 5, 32);
    Pipe pGemvO = makePipe(c, "gemv_q8_0.spv", 3, 8, 128);
    Pipe pAddN  = makePipe(c, "add_rmsnorm.spv", 5, 8);
    Pipe pAdd   = makePipe(c, "vec_add.spv", 3, 4);
    Pipe pMoeL  = makePipe(c, "moe_logits.spv", 3, 16);
    Pipe pMoeS  = makePipe(c, "moe_select.spv", 4, 16);
    Pipe pMoeGU = makePipe(c, "moe_gateup_iq3.spv", 5, 16);
    Pipe pMoeGUs = makePipe(c, "moe_gateup_q8.spv", 4, 16);
    Pipe pMoeDn = makePipe(c, moe.downQ6 ? "moe_down_q6k.spv" : "moe_down_iq4.spv", 4, 16);
    Pipe pMoeDns = makePipe(c, "moe_down_q8.spv", 4, 16);

    const VkBufferUsageFlags stor = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const size_t szMGE = (size_t)moe.nExp * moe.nFf * moe.rbGE;
    const size_t szMDE = (size_t)moe.nExp * moe.nEmbd * moe.rbDE;
    Buf bANorm = createBuf(c, nEmbd * 4, stor, true);
    Buf bWq = createBuf(c, (size_t)qfN * rbQ8e, stor, true);
    Buf bWk = createBuf(c, (size_t)kvN * rbQ8e, stor, true);
    Buf bWv = createBuf(c, (size_t)kvN * rbQ8e, stor, true);
    Buf bQN = createBuf(c, dh * 4, stor, true);
    Buf bKN = createBuf(c, dh * 4, stor, true);
    Buf bWo = createBuf(c, (size_t)nEmbd * rbQ8a, stor, true);
    Buf bPN = createBuf(c, nEmbd * 4, stor, true);
    Buf bMGI = createBuf(c, (size_t)moe.nExp * nEmbd * 4, stor, true);
    Buf bMGIS = createBuf(c, nEmbd * 4, stor, true);
    Buf bMGE = createBuf(c, szMGE, stor, true);
    Buf bMUE = createBuf(c, szMGE, stor, true);
    Buf bMDE = createBuf(c, szMDE, stor, true);
    Buf bMGS = createBuf(c, (size_t)moe.nFf * moe.rbGS, stor, true);
    Buf bMUS = createBuf(c, (size_t)moe.nFf * moe.rbGS, stor, true);
    Buf bMDS = createBuf(c, (size_t)moe.nEmbd * moe.rbDS, stor, true);

    Buf bXin = createBuf(c, nEmbd * 4, stor, true);
    Buf bXn = createBuf(c, nEmbd * 4, stor, true);
    Buf bQfull = createBuf(c, qfN * 4, stor, true);
    Buf bKin = createBuf(c, kvN * 4, stor, true);
    Buf bVin = createBuf(c, kvN * 4, stor, true);
    Buf bQhat = createBuf(c, atN * 4, stor, true);
    Buf bKC = createBuf(c, (size_t)hKV * tmax * dh * 4, stor, true);
    Buf bVC = createBuf(c, (size_t)hKV * tmax * dh * 4, stor, true);
    Buf bRope = createBuf(c, (size_t)tmax * (nRot / 2) * 2 * 4, stor, true);
    Buf bAtt = createBuf(c, atN * 4, stor, true);
    Buf bAttnOut = createBuf(c, nEmbd * 4, stor, true);
    Buf bY = createBuf(c, nEmbd * 4, stor, true);
    Buf bXn2 = createBuf(c, nEmbd * 4, stor, true);
    Buf bML = createBuf(c, moe.nExp * 4, stor, true);
    Buf bMH = createBuf(c, (size_t)(moe.nUsed + 1) * moe.nFf * 4, stor, true);
    Buf bMSel = createBuf(c, 128, stor, true);
    Buf bMY = createBuf(c, nEmbd * 4, stor, true);
    Buf bOut = createBuf(c, nEmbd * 4, stor | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    Buf stage = createBuf(c, std::max(szMDE, szMGE),
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);

    auto begin = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    };
    auto submitWait = [&]() {
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    };
    void* mapped;
    VK_CHECK(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    auto upload = [&](Buf& dst, const void* src, size_t n) {
        if (src) memcpy(mapped, src, n);
        else memset(mapped, 0, n);
        begin();
        VkBufferCopy cp{0, 0, n};
        vkCmdCopyBuffer(c.cb, stage.buf, dst.buf, 1, &cp);
        submitWait();
    };
    upload(bANorm, tANorm->data, nEmbd * 4);
    upload(bWq, tWq->data, (size_t)qfN * rbQ8e);
    upload(bWk, tWk->data, (size_t)kvN * rbQ8e);
    upload(bWv, tWv->data, (size_t)kvN * rbQ8e);
    upload(bQN, tQN->data, dh * 4);
    upload(bKN, tKN->data, dh * 4);
    upload(bWo, tWo->data, (size_t)nEmbd * rbQ8a);
    upload(bPN, tPN->data, nEmbd * 4);
    upload(bMGI, moe.gi->data, (size_t)moe.nExp * nEmbd * 4);
    upload(bMGIS, moe.gis->data, nEmbd * 4);
    upload(bMGE, moe.ge->data, szMGE);
    upload(bMUE, moe.ue->data, szMGE);
    upload(bMDE, moe.de->data, szMDE);
    upload(bMGS, moe.gs->data, (size_t)moe.nFf * moe.rbGS);
    upload(bMUS, moe.us->data, (size_t)moe.nFf * moe.rbGS);
    upload(bMDS, moe.ds->data, (size_t)moe.nEmbd * moe.rbDS);
    {   // precomputed RoPE cos/sin table (see fa_prep.comp binding 8)
        const uint32_t half = nRot / 2;
        std::vector<float> rope((size_t)tmax * half * 2);
        for (uint32_t p = 0; p < tmax; p++)
            for (uint32_t j = 0; j < half; j++) {
                float th = (float)p * std::pow(kFreqBase, -2.f * (float)j / (float)nRot);
                rope[2 * ((size_t)p * half + j)]     = std::cos(th);
                rope[2 * ((size_t)p * half + j) + 1] = std::sin(th);
            }
        upload(bRope, rope.data(), rope.size() * 4);
    }

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 20;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &dpool));
    auto mkSet = [&](Pipe& pp, std::vector<VkBuffer> bufs) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &pp.dsl;
        VkDescriptorSet ds;
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &ai, &ds));
        std::vector<VkDescriptorBufferInfo> dbi(bufs.size());
        std::vector<VkWriteDescriptorSet> wr(bufs.size());
        for (size_t i = 0; i < bufs.size(); i++) {
            dbi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[i].dstSet = ds;
            wr[i].dstBinding = (uint32_t)i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(c.dev, (uint32_t)wr.size(), wr.data(), 0, nullptr);
        return ds;
    };
    VkDescriptorSet sRms = mkSet(pRms, {bXin.buf, bANorm.buf, bXn.buf});
    VkDescriptorSet sQf = mkSet(pGemvA, {bWq.buf, bXn.buf, bQfull.buf});
    VkDescriptorSet sK = mkSet(pGemvA, {bWk.buf, bXn.buf, bKin.buf});
    VkDescriptorSet sV = mkSet(pGemvA, {bWv.buf, bXn.buf, bVin.buf});
    VkDescriptorSet sPrep = mkSet(pPrep, {bQfull.buf, bKin.buf, bVin.buf, bQN.buf, bKN.buf,
                                          bQhat.buf, bKC.buf, bVC.buf, bRope.buf});
    VkDescriptorSet sAttn = mkSet(pAttn, {bQhat.buf, bKC.buf, bVC.buf, bQfull.buf, bAtt.buf});
    VkDescriptorSet sWo = mkSet(pGemvO, {bWo.buf, bAtt.buf, bAttnOut.buf});
    VkDescriptorSet sAddN = mkSet(pAddN, {bXin.buf, bAttnOut.buf, bPN.buf, bY.buf, bXn2.buf});
    VkDescriptorSet sMoeL = mkSet(pMoeL, {bMGI.buf, bXn2.buf, bML.buf});
    VkDescriptorSet sMoeS = mkSet(pMoeS, {bML.buf, bMGIS.buf, bXn2.buf, bMSel.buf});
    VkDescriptorSet sMoeGU = mkSet(pMoeGU, {bMGE.buf, bMUE.buf, bXn2.buf, bMSel.buf, bMH.buf});
    VkDescriptorSet sMoeGUs = mkSet(pMoeGUs, {bMGS.buf, bMUS.buf, bXn2.buf, bMH.buf});
    VkDescriptorSet sMoeDn = mkSet(pMoeDn, {bMDE.buf, bMH.buf, bMSel.buf, bMY.buf});
    VkDescriptorSet sMoeDns = mkSet(pMoeDns, {bMDS.buf, bMH.buf, bMSel.buf, bMY.buf});
    VkDescriptorSet sAdd = mkSet(pAdd, {bY.buf, bMY.buf, bOut.buf});

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    auto barrier = [&]() {
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    auto dispatchB = [&](Pipe& pp, VkDescriptorSet ds, uint32_t wgs, const void* pc, uint32_t pcSize) {
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(c.cb, pp.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
        uint32_t gx = std::min(wgs, c.props.limits.maxComputeWorkGroupCount[0]);
        uint32_t gy = (wgs + gx - 1) / gx;
        vkCmdDispatch(c.cb, gx, gy, 1);
    };

    struct { uint32_t n; float e; } pcRms{nEmbd, eps}, pcAddN{nEmbd, eps};
    struct { uint32_t m, k; } pcQf{qfN, nEmbd}, pcK{kvN, nEmbd}, pcWo{nEmbd, atN};
    struct FaPc { uint32_t pos, tmax, dh, nRot, hQ, hKV; float eps, fb; }
        pcFa{0, tmax, dh, nRot, hQ, hKV, eps, kFreqBase};
    struct { uint32_t n; } pcAdd{nEmbd};
    struct { uint32_t a, b, cc, d; } pcv{moe.nEmbd, moe.nFf, moe.nExp, moe.nUsed};

    auto sequence = [&](uint32_t pos) {
        pcFa.pos = pos;
        dispatchB(pRms, sRms, 1, &pcRms, 8);
        barrier();
        dispatchB(pGemvA, sQf, (qfN + 3) / 4, &pcQf, 8);
        dispatchB(pGemvA, sK, (kvN + 3) / 4, &pcK, 8);
        dispatchB(pGemvA, sV, (kvN + 3) / 4, &pcK, 8);
        barrier();
        dispatchB(pPrep, sPrep, hQ + 2 * hKV, &pcFa, 32);
        barrier();
        dispatchB(pAttn, sAttn, hQ, &pcFa, 32);
        barrier();
        dispatchB(pGemvO, sWo, (nEmbd + 1) / 2, &pcWo, 8);
        barrier();
        dispatchB(pAddN, sAddN, 1, &pcAddN, 8);
        barrier();
        dispatchB(pMoeL, sMoeL, moe.nExp, &pcv, 16);
        dispatchB(pMoeGUs, sMoeGUs, moe.nFf, &pcv, 16);
        barrier();
        dispatchB(pMoeS, sMoeS, 1, &pcv, 16);
        barrier();
        dispatchB(pMoeGU, sMoeGU, moe.nUsed * moe.nFf, &pcv, 16);
        barrier();
        dispatchB(pMoeDn, sMoeDn, moe.nEmbd, &pcv, 16);
        barrier();
        dispatchB(pMoeDns, sMoeDns, moe.nEmbd, &pcv, 16);
        barrier();
        dispatchB(pAdd, sAdd, 1, &pcAdd, 4);
    };

    // ---- CPU mirror ----
    const float* anW = (const float*)tANorm->data;
    const float* pnW = (const float*)tPN->data;
    const float* qnW = (const float*)tQN->data;
    const float* knW = (const float*)tKN->data;
    std::vector<float> kcCpu((size_t)hKV * tmax * dh, 0.f), vcCpu((size_t)hKV * tmax * dh, 0.f);
    std::vector<float> x(nEmbd), xn(nEmbd), qfull(qfN), kin(kvN), vin(kvN), qhat(atN),
        att(atN), attnOut(nEmbd), y(nEmbd), xn2(nEmbd), moeOut(nEmbd), refOut(nEmbd),
        gpuOut(nEmbd), tmpK(std::max(nEmbd, atN));
    auto q8Gemv = [&](const GgufTensor* t, const std::vector<float>& xin,
                      std::vector<float>& out, uint32_t M, uint32_t K) {
        size_t rb = ggmlRowBytes(GGML_Q8_0, K);
        for (uint32_t r = 0; r < M; r++) {
            dequant_row_q8_0((const block_q8_0*)(t->data + (size_t)r * rb), tmpK.data(), K);
            double a = 0;
            for (uint32_t k = 0; k < K; k++) a += (double)tmpK[k] * xin[k];
            out[r] = (float)a;
        }
    };
    auto rmsRef = [&](const float* in, const float* w, float* out, uint32_t n) {
        double ss = 0;
        for (uint32_t i = 0; i < n; i++) ss += (double)in[i] * in[i];
        double sc = 1.0 / std::sqrt(ss / n + eps);
        for (uint32_t i = 0; i < n; i++) out[i] = (float)(in[i] * sc * w[i]);
    };
    auto ropeRef = [&](float* v, uint32_t pos) {
        for (uint32_t j = 0; j < nRot / 2; j++) {
            float th = pos * std::pow(kFreqBase, -2.f * j / nRot);
            float cs = std::cos(th), sn = std::sin(th);
            float x0 = v[j], x1 = v[j + nRot / 2];
            v[j] = x0 * cs - x1 * sn;
            v[j + nRot / 2] = x0 * sn + x1 * cs;
        }
    };

    std::mt19937 xr(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    bool pass = true;
    printf("cpu reference + gpu, %u tokens...\n", nTok);
    for (uint32_t t = 0; t < nTok; t++) {
        for (auto& v : x) v = nd(xr);
        // CPU
        rmsRef(x.data(), anW, xn.data(), nEmbd);
        q8Gemv(tWq, xn, qfull, qfN, nEmbd);
        q8Gemv(tWk, xn, kin, kvN, nEmbd);
        q8Gemv(tWv, xn, vin, kvN, nEmbd);
        for (uint32_t h = 0; h < hQ; h++) {
            rmsRef(&qfull[(size_t)h * 2 * dh], qnW, &qhat[(size_t)h * dh], dh);
            ropeRef(&qhat[(size_t)h * dh], t);
        }
        for (uint32_t h = 0; h < hKV; h++) {
            float* kd = &kcCpu[((size_t)h * tmax + t) * dh];
            rmsRef(&kin[(size_t)h * dh], knW, kd, dh);
            ropeRef(kd, t);
            memcpy(&vcCpu[((size_t)h * tmax + t) * dh], &vin[(size_t)h * dh], dh * 4);
        }
        for (uint32_t h = 0; h < hQ; h++) {
            uint32_t kv = h / (hQ / hKV);
            std::vector<double> sc(t + 1);
            double mx = -1e300;
            for (uint32_t p = 0; p <= t; p++) {
                double s = 0;
                for (uint32_t j = 0; j < dh; j++)
                    s += (double)qhat[(size_t)h * dh + j] * kcCpu[((size_t)kv * tmax + p) * dh + j];
                sc[p] = s / std::sqrt((double)dh);
                mx = std::max(mx, sc[p]);
            }
            double sum = 0;
            for (uint32_t p = 0; p <= t; p++) {
                sc[p] = std::exp(sc[p] - mx);
                sum += sc[p];
            }
            for (uint32_t j = 0; j < dh; j++) {
                double o = 0;
                for (uint32_t p = 0; p <= t; p++)
                    o += sc[p] * vcCpu[((size_t)kv * tmax + p) * dh + j];
                o /= sum;
                double gate = 1.0 / (1.0 + std::exp(-(double)qfull[(size_t)h * 2 * dh + dh + j]));
                att[(size_t)h * dh + j] = (float)(o * gate);
            }
        }
        q8Gemv(tWo, att, attnOut, nEmbd, atN);
        for (uint32_t i = 0; i < nEmbd; i++) y[i] = x[i] + attnOut[i];
        rmsRef(y.data(), pnW, xn2.data(), nEmbd);
        moeCpuRef(moe, xn2.data(), moeOut.data(), nullptr);
        for (uint32_t i = 0; i < nEmbd; i++) refOut[i] = y[i] + moeOut[i];

        // GPU
        upload(bXin, x.data(), nEmbd * 4);
        begin();
        sequence(t);
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer = bOut.buf;
        bmb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bmb, 0, nullptr);
        VkBufferCopy cp{0, 0, nEmbd * 4};
        vkCmdCopyBuffer(c.cb, bOut.buf, stage.buf, 1, &cp);
        submitWait();
        memcpy(gpuOut.data(), mapped, nEmbd * 4);

        double rms = 0;
        for (uint32_t i = 0; i < nEmbd; i++) rms += (double)refOut[i] * refOut[i];
        rms = std::sqrt(rms / nEmbd);
        double floorD = std::max(1e-3, 1e-3 * rms);
        double maxRel = 0;
        uint32_t bad = 0;
        for (uint32_t i = 0; i < nEmbd; i++) {
            double rel = std::fabs((double)gpuOut[i] - refOut[i]) /
                         std::max(floorD, (double)std::fabs(refOut[i]));
            maxRel = std::max(maxRel, rel);
            if (rel > 1e-2 && bad++ < 3)
                printf("  tok%u y[%u]: gpu=%g ref=%g\n", t, i, gpuOut[i], refOut[i]);
        }
        pass &= bad == 0;
        printf("token %u: max_rel_err = %.3g  ->  %s\n", t, maxRel, bad == 0 ? "PASS" : "FAIL");
    }

    if (pass && c.hasTimestamps && iters > 0) {
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VkQueryPool qp;
        VK_CHECK(vkCreateQueryPool(c.dev, &qpci, nullptr, &qp));
        auto runBench = [&]() {
            begin();
            vkCmdResetQueryPool(c.cb, qp, 0, 2);
            vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            for (uint32_t i = 0; i < iters; i++) {
                sequence(nTok - 1);
                barrier();
            }
            vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            submitWait();
        };
        runBench();
        runBench();
        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(c.dev, qp, 0, 2, sizeof(ts), ts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double ns = (double)(ts[1] - ts[0]) * c.props.limits.timestampPeriod / iters;
        printf("gpu: %8.1f µs/block (pos=%u) | 15 dispatches, 1 submit\n", ns / 1e3, nTok - 1);
        vkDestroyQueryPool(c.dev, qp, nullptr);
    }

    vkUnmapMemory(c.dev, stage.mem);
    vkDestroyDescriptorPool(c.dev, dpool, nullptr);
    for (Pipe* pp : {&pRms, &pGemvA, &pPrep, &pAttn, &pGemvO, &pAddN, &pAdd, &pMoeL, &pMoeS,
                     &pMoeGU, &pMoeGUs, &pMoeDn, &pMoeDns})
        destroyPipe(c, *pp);
    for (Buf* b : {&bANorm, &bWq, &bWk, &bWv, &bQN, &bKN, &bWo, &bPN, &bMGI, &bMGIS, &bMGE,
                   &bMUE, &bMDE, &bMGS, &bMUS, &bMDS, &bXin, &bXn, &bQfull, &bKin, &bVin,
                   &bQhat, &bKC, &bVC, &bAtt, &bAttnOut, &bY, &bXn2, &bML, &bMH, &bMSel,
                   &bMY, &bOut, &stage})
        destroyBuf(c, *b);
    return pass;
}

// M6b: end-to-end greedy decode over all 40 layers + embeddings + LM head.
// Requires the whole model in VRAM (~16.5 GB) — quiesce llama-server first.
// usage: qk token <ids-file> <nGen> [tmax]
static bool caseToken(VkCtx& c, const char* idsFile, uint32_t nGen, uint32_t tmax, uint32_t nB,
                      bool warmDemo = false) {
    std::vector<int> promptIds;
    {
        FILE* f = fopen(idsFile, "r");
        if (!f) { perror(idsFile); return false; }
        int v;
        while (fscanf(f, "%d%*[, \n]", &v) == 1) promptIds.push_back(v);
        fclose(f);
    }
    if (promptIds.empty() || promptIds.size() + nGen > tmax) {
        fprintf(stderr, "bad prompt (%zu ids, tmax %u)\n", promptIds.size(), tmax);
        return false;
    }
    Gguf g;
    if (!g.open(ggufPath())) return false;
    const GgufTensor* tEmbd = g.find("token_embd.weight");
    const GgufTensor* tONorm = g.find("output_norm.weight");
    const GgufTensor* tHead = g.find("output.weight");
    if (!tEmbd || !tONorm || !tHead || tEmbd->type != GGML_Q6_K || tHead->type != GGML_Q6_K) {
        fprintf(stderr, "missing/unexpected embd/head tensors\n");
        return false;
    }
    const uint32_t nEmbd = 2048, chQkv = 8192, dIn = 4096, hV = 32, dS = 128, hK = 16;
    const uint32_t dh = 256, hQ = 16, hKV = 2, nRot = 64, nLayer = 40;
    const uint32_t vocab = (uint32_t)tHead->ne[1];
    const float eps = 1e-6f;
    const size_t rbQ8e = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbQ8i = ggmlRowBytes(GGML_Q8_0, dIn);
    const size_t rbE = ggmlRowBytes(GGML_Q6_K, nEmbd);
    if (nB < 1 || nB > 64) { fprintf(stderr, "batch must be 1..64\n"); return false; }
    printf("token mode: %zu prompt ids, gen %u, vocab %u, tmax %u, batch %u\n",
           promptIds.size(), nGen, vocab, tmax, nB);

    // ---- pipes (shared across layers) ----
    Pipe pRms = makePipe(c, "rmsnorm.spv", 3, 8);
    Pipe pGemvA = makePipe(c, "gemv_q8_0.spv", 3, 8, 64);
    Pipe pAb = makePipe(c, "dn_ab.spv", 6, 8);
    Pipe pConvN = makePipe(c, "dn_convn.spv", 4, 16);
    Pipe pStep = makePipe(c, "dn_step.spv", 4, 12);
    Pipe pGate = makePipe(c, "dn_gate.spv", 4, 12);
    Pipe pGemvO = makePipe(c, "gemv_q8_0.spv", 3, 8, 128);
    Pipe pAddN = makePipe(c, "add_rmsnorm.spv", 5, 8);
    Pipe pAdd = makePipe(c, "vec_add.spv", 3, 4);
    Pipe pPrep = makePipe(c, "fa_prep.spv", 9, 32);
    Pipe pAttn = makePipe(c, "fa_attn.spv", 5, 32);
    Pipe pMoeL = makePipe(c, "moe_logits.spv", 3, 16);
    Pipe pMoeS = makePipe(c, "moe_select.spv", 4, 16);
    Pipe pMoeGU = makePipe(c, "moe_gateup_iq3.spv", 5, 16);
    Pipe pMoeGUs = makePipe(c, "moe_gateup_q8.spv", 4, 16);
    Pipe pMoeDn4 = makePipe(c, "moe_down_iq4.spv", 4, 16);
    Pipe pMoeDn6 = makePipe(c, "moe_down_q6k.spv", 4, 16);
    Pipe pMoeDnsB = makePipe(c, "moe_down_q8b.spv", 4, 16);
    Pipe pAdd3 = makePipe(c, "add_rms3.spv", 6, 8);
    Pipe pHead = makePipe(c, "gemv_q6_k.spv", 3, 8, 128);
    Pipe pAm1 = makePipe(c, "argmax1.spv", 3, 8);
    Pipe pAm2 = makePipe(c, "argmax2.spv", 3, 4);
    Pipe pEmb = makePipe(c, "embed_q6k.spv", 3, 12);

    // ---- shared activation buffers ----
    const VkBufferUsageFlags stor = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags storSrc = stor | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    Buf bXin = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    Buf bXn = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    Buf bBig = createBuf(c, (size_t)nB * chQkv * 4, stor, true);       // qkv | qfull
    Buf bMid = createBuf(c, (size_t)nB * dIn * 4, stor, true);         // z | qhat
    Buf bKin = createBuf(c, (size_t)nB * hKV * dh * 4, stor, true);
    Buf bVin = createBuf(c, (size_t)nB * hKV * dh * 4, stor, true);
    Buf bGb = createBuf(c, (size_t)nB * 2 * hV * 4, stor, true);
    Buf bConvOut = createBuf(c, (size_t)nB * chQkv * 4, stor, true);
    Buf bO = createBuf(c, (size_t)nB * dIn * 4, stor, true);
    Buf bAtt = createBuf(c, (size_t)nB * dIn * 4, stor, true);
    Buf bAttnOut = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    Buf bY = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    Buf bXn2 = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    Buf bML = createBuf(c, (size_t)nB * 256 * 4, stor, true);
    Buf bMH = createBuf(c, (size_t)nB * 9 * 512 * 4, stor, true);
    Buf bMSel = createBuf(c, (size_t)nB * 128, stor, true);
    Buf bMY = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    Buf bMY2 = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    Buf bOut = createBuf(c, nEmbd * 4, storSrc, true);
    Buf bONorm = createBuf(c, nEmbd * 4, stor, true);
    Buf bHeadW = createBuf(c, (size_t)vocab * rbE, stor, true);
    Buf bLogits = createBuf(c, (size_t)nB * vocab * 4, storSrc, true);
    Buf bEmbdW = createBuf(c, (size_t)vocab * rbE, stor, true);
    Buf bPids = createBuf(c, (size_t)promptIds.size() * 4, stor, true);
    Buf bAV = createBuf(c, (size_t)nB * 64 * 4, stor, true);
    Buf bAI = createBuf(c, (size_t)nB * 64 * 4, stor, true);
    Buf bTok = createBuf(c, (size_t)nB * 4, storSrc, true);
    // RoPE cos/sin precomputed on CPU: [pos][nRot/2][2]; removes pow/cos/sin from fa_prep.
    Buf bRope = createBuf(c, (size_t)tmax * (nRot / 2) * 2 * 4, stor, true);
    Buf bRb;  // generated-id readback, host-cached so CPU reads are fast
    {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = (size_t)tmax * nB * 4;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(c.dev, &bci, nullptr, &bRb.buf));
        bRb.size = bci.size;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(c.dev, bRb.buf, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = findMemType(
            c.mp, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
            mai.memoryTypeIndex = findMemType(
                c.mp, req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(c.dev, &mai, nullptr, &bRb.mem));
        VK_CHECK(vkBindBufferMemory(c.dev, bRb.buf, bRb.mem, 0));
    }
    Buf stage = createBuf(c, 160u << 20,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);

    auto begin = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    };
    auto submitWait = [&]() {
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    };
    void* mapped;
    VK_CHECK(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    size_t vramMB = 0;
    auto upload = [&](Buf& dst, const void* src, size_t n) {
        for (size_t off = 0; off < n; off += (140u << 20)) {
            size_t chunk = std::min(n - off, (size_t)(140u << 20));
            if (src) memcpy(mapped, (const uint8_t*)src + off, chunk);
            else memset(mapped, 0, chunk);
            begin();
            VkBufferCopy cp{0, off, chunk};
            vkCmdCopyBuffer(c.cb, stage.buf, dst.buf, 1, &cp);
            submitWait();
        }
        vramMB += n >> 20;
    };
    upload(bONorm, tONorm->data, nEmbd * 4);
    {   // precompute RoPE cos/sin table (NeoX partial rope, theta = pos*base^(-2j/nRot))
        const uint32_t half = nRot / 2;
        std::vector<float> rope((size_t)tmax * half * 2);
        for (uint32_t p = 0; p < tmax; p++)
            for (uint32_t j = 0; j < half; j++) {
                float th = (float)p * std::pow(kFreqBase, -2.f * (float)j / (float)nRot);
                rope[2 * ((size_t)p * half + j)]     = std::cos(th);
                rope[2 * ((size_t)p * half + j) + 1] = std::sin(th);
            }
        upload(bRope, rope.data(), rope.size() * 4);
    }
    upload(bHeadW, tHead->data, (size_t)vocab * rbE);
    upload(bEmbdW, tEmbd->data, (size_t)vocab * rbE);
    {
        std::vector<uint32_t> pi(promptIds.begin(), promptIds.end());
        upload(bPids, pi.data(), pi.size() * 4);
    }

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1024;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &dpool));
    auto mkSet = [&](Pipe& pp, std::vector<VkBuffer> bufs) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &pp.dsl;
        VkDescriptorSet ds;
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &ai, &ds));
        std::vector<VkDescriptorBufferInfo> dbi(bufs.size());
        std::vector<VkWriteDescriptorSet> wr(bufs.size());
        for (size_t i = 0; i < bufs.size(); i++) {
            dbi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[i].dstSet = ds;
            wr[i].dstBinding = (uint32_t)i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(c.dev, (uint32_t)wr.size(), wr.data(), 0, nullptr);
        return ds;
    };

    struct Layer {
        bool rec = false, downQ6 = false;
        std::vector<Buf> bufs;  // owned weight+state buffers
        VkDescriptorSet sRms, sP1, sP2, sP3, sAb, sConv, sStep, sGate, sWo, sAddN,
            sPrep, sAttn, sMoeL, sMoeS, sMoeGU, sMoeGUs, sMoeDn, sMoeDns, sAdd3;
        VkBuffer aNormBuf = VK_NULL_HANDLE;
        // per-slot recurrent state, snapshot-able for prefix caching:
        // rec layer -> (conv window, delta-rule S); attn layer -> (K cache, V cache).
        VkBuffer st1 = VK_NULL_HANDLE, st2 = VK_NULL_HANDLE;
        size_t sz1 = 0, sz2 = 0;
    };
    std::vector<Layer> layers(nLayer);
    char nb[128];

    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        auto T = [&](const char* suffix) -> const GgufTensor* {
            snprintf(nb, sizeof nb, "blk.%u.%s", il, suffix);
            return g.find(nb);
        };
        auto W = [&](const GgufTensor* t, size_t n) -> VkBuffer {
            // storSrc: every layer buffer is copy-source-capable so recurrent
            // state can be snapshotted for prefix caching (transfer-src is free).
            L.bufs.push_back(createBuf(c, n, storSrc, true));
            upload(L.bufs.back(), t ? t->data : nullptr, n);
            return L.bufs.back().buf;
        };
        MoeT moe;
        if (!loadMoeT(g, il, moe)) return false;
        L.downQ6 = moe.downQ6;
        L.rec = T("ssm_a") != nullptr;

        VkBuffer aNorm = W(T("attn_norm.weight"), nEmbd * 4);
        VkBuffer pn = W(T("post_attention_norm.weight"), nEmbd * 4);
        L.aNormBuf = aNorm;
        if (L.rec) {
            VkBuffer qkvW = W(T("attn_qkv.weight"), (size_t)chQkv * rbQ8e);
            VkBuffer zW = W(T("attn_gate.weight"), (size_t)dIn * rbQ8e);
            VkBuffer alW = W(T("ssm_alpha.weight"), (size_t)hV * nEmbd * 4);
            VkBuffer beW = W(T("ssm_beta.weight"), (size_t)hV * nEmbd * 4);
            VkBuffer dt = W(T("ssm_dt.bias"), hV * 4);
            VkBuffer av = W(T("ssm_a"), hV * 4);
            VkBuffer ker = W(T("ssm_conv1d.weight") ? T("ssm_conv1d.weight") : T("ssm_conv1d"),
                             (size_t)chQkv * 4 * 4);
            VkBuffer sn = W(T("ssm_norm.weight"), dS * 4);
            VkBuffer outW = W(T("ssm_out.weight"), (size_t)nEmbd * rbQ8i);
            VkBuffer convSt = W(nullptr, (size_t)nB * chQkv * 3 * 4);
            VkBuffer S = W(nullptr, (size_t)nB * hV * dS * dS * 4);
            L.st1 = convSt; L.sz1 = (size_t)nB * chQkv * 3 * 4;
            L.st2 = S;      L.sz2 = (size_t)nB * hV * dS * dS * 4;
            L.sRms = mkSet(pRms, {bXin.buf, aNorm, bXn.buf});
            L.sP1 = mkSet(pGemvA, {qkvW, bXn.buf, bBig.buf});
            L.sP2 = mkSet(pGemvA, {zW, bXn.buf, bMid.buf});
            L.sAb = mkSet(pAb, {bXn.buf, alW, beW, dt, av, bGb.buf});
            L.sConv = mkSet(pConvN, {convSt, bBig.buf, ker, bConvOut.buf});
            L.sStep = mkSet(pStep, {bConvOut.buf, bGb.buf, S, bO.buf});
            L.sGate = mkSet(pGate, {bO.buf, sn, bMid.buf, bAtt.buf});
            L.sWo = mkSet(pGemvO, {outW, bAtt.buf, bAttnOut.buf});
        } else {
            VkBuffer wq = W(T("attn_q.weight"), (size_t)chQkv * rbQ8e);
            VkBuffer wk = W(T("attn_k.weight"), (size_t)hKV * dh * rbQ8e);
            VkBuffer wv = W(T("attn_v.weight"), (size_t)hKV * dh * rbQ8e);
            VkBuffer qn = W(T("attn_q_norm.weight"), dh * 4);
            VkBuffer kn = W(T("attn_k_norm.weight"), dh * 4);
            VkBuffer wo = W(T("attn_output.weight"), (size_t)nEmbd * rbQ8i);
            VkBuffer kc = W(nullptr, (size_t)nB * hKV * tmax * dh * 4);
            VkBuffer vc = W(nullptr, (size_t)nB * hKV * tmax * dh * 4);
            L.st1 = kc; L.sz1 = (size_t)nB * hKV * tmax * dh * 4;
            L.st2 = vc; L.sz2 = (size_t)nB * hKV * tmax * dh * 4;
            L.sRms = mkSet(pRms, {bXin.buf, aNorm, bXn.buf});
            L.sP1 = mkSet(pGemvA, {wq, bXn.buf, bBig.buf});
            L.sP2 = mkSet(pGemvA, {wk, bXn.buf, bKin.buf});
            L.sP3 = mkSet(pGemvA, {wv, bXn.buf, bVin.buf});
            L.sPrep = mkSet(pPrep, {bBig.buf, bKin.buf, bVin.buf, qn, kn, bMid.buf, kc, vc, bRope.buf});
            L.sAttn = mkSet(pAttn, {bMid.buf, kc, vc, bBig.buf, bAtt.buf});
            L.sWo = mkSet(pGemvO, {wo, bAtt.buf, bAttnOut.buf});
        }
        VkBuffer mgi = W(moe.gi, (size_t)moe.nExp * nEmbd * 4);
        VkBuffer mgis = W(moe.gis, nEmbd * 4);
        VkBuffer mge = W(moe.ge, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        VkBuffer mue = W(moe.ue, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        VkBuffer mde = W(moe.de, (size_t)moe.nExp * moe.nEmbd * moe.rbDE);
        VkBuffer mgs = W(moe.gs, (size_t)moe.nFf * moe.rbGS);
        VkBuffer mus = W(moe.us, (size_t)moe.nFf * moe.rbGS);
        VkBuffer mds = W(moe.ds, (size_t)moe.nEmbd * moe.rbDS);
        L.sAddN = mkSet(pAddN, {bXin.buf, bAttnOut.buf, pn, bY.buf, bXn2.buf});
        L.sMoeL = mkSet(pMoeL, {mgi, bXn2.buf, bML.buf});
        L.sMoeS = mkSet(pMoeS, {bML.buf, mgis, bXn2.buf, bMSel.buf});
        L.sMoeGU = mkSet(pMoeGU, {mge, mue, bXn2.buf, bMSel.buf, bMH.buf});
        L.sMoeGUs = mkSet(pMoeGUs, {mgs, mus, bXn2.buf, bMH.buf});
        L.sMoeDn = mkSet(L.downQ6 ? pMoeDn6 : pMoeDn4, {mde, bMH.buf, bMSel.buf, bMY.buf});
        L.sMoeDns = mkSet(pMoeDnsB, {mds, bMH.buf, bMSel.buf, bMY2.buf});
        if (il % 8 == 7) printf("  loaded %u/40 layers (%zu MB in VRAM)\n", il + 1, vramMB);
    }
    // layer tail = add + NEXT layer's norm (output_norm after the last layer)
    for (uint32_t il = 0; il < nLayer; il++)
        layers[il].sAdd3 = mkSet(pAdd3, {bY.buf, bMY.buf, bMY2.buf,
                                         il + 1 < nLayer ? layers[il + 1].aNormBuf : bONorm.buf,
                                         bXin.buf, bXn.buf});
    VkDescriptorSet sHead = mkSet(pHead, {bHeadW.buf, bXn.buf, bLogits.buf});
    VkDescriptorSet sAm1 = mkSet(pAm1, {bLogits.buf, bAV.buf, bAI.buf});
    VkDescriptorSet sAm2 = mkSet(pAm2, {bAV.buf, bAI.buf, bTok.buf});
    VkDescriptorSet sEmbPre = mkSet(pEmb, {bEmbdW.buf, bPids.buf, bXin.buf});
    VkDescriptorSet sEmbDec = mkSet(pEmb, {bEmbdW.buf, bTok.buf, bXin.buf});
    printf("model resident: ~%zu MB\n", vramMB);

    VkCommandBuffer rcb = c.cb;  // recording target (switched to prebuilt CBs below)
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    auto barrier = [&]() {
        vkCmdPipelineBarrier(rcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    auto dispatchB = [&](Pipe& pp, VkDescriptorSet ds, uint32_t wgs, const void* pc, uint32_t pcSize) {
        vkCmdBindPipeline(rcb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.p);
        vkCmdBindDescriptorSets(rcb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(rcb, pp.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
        uint32_t gx = std::min(wgs, c.props.limits.maxComputeWorkGroupCount[0]);
        uint32_t gy = (wgs + gx - 1) / gx;
        vkCmdDispatch(rcb, gx, gy, nB);
    };

    struct { uint32_t n; float e; } pcRms{nEmbd, eps};
    struct { uint32_t m, k; } pcQkv{chQkv, nEmbd}, pcZ{dIn, nEmbd}, pcKV{hKV * dh, nEmbd},
        pcWo{nEmbd, dIn}, pcHead{vocab, nEmbd};
    struct { uint32_t n, h; } pcAb{nEmbd, hV};
    struct { uint32_t ch, d, qkch; float e; } pcConvN{chQkv, dS, 2 * hK * dS, eps};
    struct { uint32_t d, hk, hv; } pcStep{dS, hK, hV};
    struct { uint32_t d, hv; float e; } pcGate{dS, hV, eps};
    struct { uint32_t n; } pcAdd{nEmbd};
    struct { uint32_t a, b, cc, d; } pcv{nEmbd, 512, 256, 8};
    struct { uint32_t pos, tmax, dh, nRot, hQ, hKV_; float eps, fb; }
        pcFa{0, tmax, dh, nRot, hQ, hKV, eps, kFreqBase};

    auto recordToken = [&](uint32_t pos, bool withHead) {
        pcFa.pos = pos;
        dispatchB(pRms, layers[0].sRms, 1, &pcRms, 8);  // layer-0 input norm
        barrier();
        for (uint32_t il = 0; il < nLayer; il++) {
            Layer& L = layers[il];
            if (L.rec) {
                dispatchB(pGemvA, L.sP1, (chQkv + 3) / 4, &pcQkv, 8);
                dispatchB(pGemvA, L.sP2, (dIn + 3) / 4, &pcZ, 8);
                dispatchB(pAb, L.sAb, 2 * hV, &pcAb, 8);
                barrier();
                dispatchB(pConvN, L.sConv, chQkv / dS, &pcConvN, 16);
                barrier();
                dispatchB(pStep, L.sStep, hV, &pcStep, 12);
                barrier();
                dispatchB(pGate, L.sGate, hV, &pcGate, 12);
                barrier();
                dispatchB(pGemvO, L.sWo, (nEmbd + 1) / 2, &pcWo, 8);
            } else {
                dispatchB(pGemvA, L.sP1, (chQkv + 3) / 4, &pcQkv, 8);
                dispatchB(pGemvA, L.sP2, (hKV * dh + 3) / 4, &pcKV, 8);
                dispatchB(pGemvA, L.sP3, (hKV * dh + 3) / 4, &pcKV, 8);
                barrier();
                dispatchB(pPrep, L.sPrep, hQ + 2 * hKV, &pcFa, 32);
                barrier();
                dispatchB(pAttn, L.sAttn, hQ, &pcFa, 32);
                barrier();
                dispatchB(pGemvO, L.sWo, (nEmbd + 1) / 2, &pcWo, 8);
            }
            barrier();
            dispatchB(pAddN, L.sAddN, 1, &pcRms, 8);
            barrier();
            dispatchB(pMoeL, L.sMoeL, 256, &pcv, 16);
            dispatchB(pMoeGUs, L.sMoeGUs, 512, &pcv, 16);
            barrier();
            dispatchB(pMoeS, L.sMoeS, 1, &pcv, 16);
            barrier();
            dispatchB(pMoeGU, L.sMoeGU, 8 * 512, &pcv, 16);
            barrier();
            dispatchB(L.downQ6 ? pMoeDn6 : pMoeDn4, L.sMoeDn, nEmbd, &pcv, 16);
            dispatchB(pMoeDnsB, L.sMoeDns, nEmbd, &pcv, 16);  // disjoint outputs: concurrent
            barrier();
            dispatchB(pAdd3, L.sAdd3, 1, &pcRms, 8);  // residual sum + next layer's norm
            barrier();
        }
        if (withHead) {
            // bXn already holds output_norm(x) from the last layer's add_rms3
            dispatchB(pHead, sHead, (vocab + 1) / 2, &pcHead, 8);
        }
    };

    // Pre-record everything. Prefill CBs embed from the prompt-id buffer;
    // decode CBs run GPU argmax of the previous logits -> embed -> layers.
    // Each phase then executes as ONE queue submission: the sampling loop is
    // fully GPU-resident and the host only reads the generated ids at the end.
    const uint32_t nPrompt = (uint32_t)promptIds.size();
    const uint32_t amWgs = (vocab + 4095) / 4096;
    struct { uint32_t n, span; } pcAm{vocab, 4096};
    struct { uint32_t m; } pcAm2{amWgs};

    std::vector<VkCommandBuffer> cbPre(nPrompt), cbDec(tmax);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = c.pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = nPrompt;
    VK_CHECK(vkAllocateCommandBuffers(c.dev, &cbai, cbPre.data()));
    cbai.commandBufferCount = tmax;
    VK_CHECK(vkAllocateCommandBuffers(c.dev, &cbai, cbDec.data()));

    auto tr = std::chrono::steady_clock::now();
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    for (uint32_t p = 0; p < nPrompt; p++) {
        rcb = cbPre[p];
        VK_CHECK(vkBeginCommandBuffer(rcb, &cbbi));
        barrier();  // order against the previous CB in the queue
        struct { uint32_t k, idx, pr; } pcE{nEmbd, p, 0};
        dispatchB(pEmb, sEmbPre, 1, &pcE, 12);
        barrier();
        recordToken(p, p + 1 == nPrompt);
        VK_CHECK(vkEndCommandBuffer(rcb));
    }
    for (uint32_t p = nPrompt; p < tmax; p++) {
        rcb = cbDec[p];
        VK_CHECK(vkBeginCommandBuffer(rcb, &cbbi));
        // order against previous CB's compute AND its bTok->bRb transfer read
        VkMemoryBarrier m0{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m0.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        m0.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(rcb,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m0, 0, nullptr, 0, nullptr);
        dispatchB(pAm1, sAm1, amWgs, &pcAm, 8);
        barrier();
        dispatchB(pAm2, sAm2, 1, &pcAm2, 4);
        VkMemoryBarrier m2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        m2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(rcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &m2, 0, nullptr, 0, nullptr);
        struct { uint32_t k, idx, pr; } pcE{nEmbd, 0, 1};
        dispatchB(pEmb, sEmbDec, 1, &pcE, 12);
        VkBufferCopy ct{0, (size_t)p * nB * 4, (size_t)nB * 4};
        vkCmdCopyBuffer(rcb, bTok.buf, bRb.buf, 1, &ct);
        barrier();
        recordToken(p, true);
        VK_CHECK(vkEndCommandBuffer(rcb));
    }
    rcb = c.cb;
    printf("pre-recorded %u command buffers in %.0f ms\n", nPrompt + tmax - nPrompt + (tmax - nPrompt),
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count());

    void* rbMap;
    VK_CHECK(vkMapMemory(c.dev, bRb.mem, 0, VK_WHOLE_SIZE, 0, &rbMap));
    auto submitBatch = [&](VkCommandBuffer* list, uint32_t n) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = n;
        si.pCommandBuffers = list;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    };

    // ---- prefix-state cache: snapshot/restore all per-slot recurrent state ----
    // A real server computes a shared prefix's state once, then clones it into
    // each new request's slot instead of re-running prefill. Here we demo that
    // by snapshotting after prefill and proving a warm restore reproduces the
    // cold token stream while skipping prefill entirely.
    std::vector<Buf> cache1, cache2;
    Buf cacheLogits;
    auto copyState = [&](bool save) {
        begin();
        for (uint32_t il = 0; il < nLayer; il++) {
            Layer& L = layers[il];
            VkBufferCopy a{0, 0, L.sz1}, b{0, 0, L.sz2};
            if (save) {
                vkCmdCopyBuffer(c.cb, L.st1, cache1[il].buf, 1, &a);
                vkCmdCopyBuffer(c.cb, L.st2, cache2[il].buf, 1, &b);
            } else {
                vkCmdCopyBuffer(c.cb, cache1[il].buf, L.st1, 1, &a);
                vkCmdCopyBuffer(c.cb, cache2[il].buf, L.st2, 1, &b);
            }
        }
        VkBufferCopy lg{0, 0, (size_t)nB * vocab * 4};
        if (save) vkCmdCopyBuffer(c.cb, bLogits.buf, cacheLogits.buf, 1, &lg);
        else      vkCmdCopyBuffer(c.cb, cacheLogits.buf, bLogits.buf, 1, &lg);
        submitWait();
    };
    if (warmDemo) {
        const VkBufferUsageFlags cp =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        size_t cacheMB = 0;
        for (uint32_t il = 0; il < nLayer; il++) {
            cache1.push_back(createBuf(c, layers[il].sz1, cp, true));
            cache2.push_back(createBuf(c, layers[il].sz2, cp, true));
            cacheMB += (layers[il].sz1 + layers[il].sz2) >> 20;
        }
        cacheLogits = createBuf(c, (size_t)nB * vocab * 4, cp, true);
        printf("prefix-cache: %zu MB of per-slot recurrent state (KV + delta-rule S + conv)\n", cacheMB);
    }

    auto t0 = std::chrono::steady_clock::now();
    submitBatch(cbPre.data(), nPrompt);
    double prefillMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
    printf("prefill: %u tokens in %.1f ms (%.2f ms/token, one submission)\n", nPrompt,
           prefillMs, prefillMs / nPrompt);

    double snapMs = 0;
    if (warmDemo) {
        auto ts = std::chrono::steady_clock::now();
        copyState(true);  // snapshot state at pos = nPrompt
        snapMs = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - ts).count();
    }

    t0 = std::chrono::steady_clock::now();
    submitBatch(&cbDec[nPrompt], nGen);
    double genMs = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();

    std::vector<int> genIds(nGen);
    const uint32_t* rb = (const uint32_t*)rbMap;
    bool streamsEq = true;
    for (uint32_t i = 0; i < nGen; i++) {
        genIds[i] = (int)rb[(size_t)(nPrompt + i) * nB];
        for (uint32_t r = 1; r < nB; r++)
            if (rb[(size_t)(nPrompt + i) * nB + r] != (uint32_t)genIds[i]) streamsEq = false;
    }
    printf("decode: %u steps x %u streams in %.1f ms -> %.2f ms/step | per-stream %.1f tok/s | aggregate %.1f tok/s\n",
           nGen, nB, genMs, genMs / nGen, nGen * 1000.0 / genMs, (double)nGen * nB * 1000.0 / genMs);
    if (nB > 1) printf("streams identical: %s\n", streamsEq ? "YES" : "NO");
    printf("GEN:");
    for (int id : genIds) printf(" %d", id);
    printf("\n");

    if (warmDemo) {
        std::vector<int> cold = genIds;
        auto tr2 = std::chrono::steady_clock::now();
        copyState(false);  // restore snapshot: a "new request" reuses the prefix state
        double restoreMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - tr2).count();
        auto tw = std::chrono::steady_clock::now();
        submitBatch(&cbDec[nPrompt], nGen);  // decode with NO prefill
        double warmMs = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - tw).count();
        std::vector<int> warm(nGen);
        for (uint32_t i = 0; i < nGen; i++) warm[i] = (int)rb[(size_t)(nPrompt + i) * nB];
        bool eq = warm == cold;
        printf("\n--- prefix-cache warm-start demo (prompt = shared prefix) ---\n");
        printf("cold start: prefill %u tokens = %.1f ms before first token\n", nPrompt, prefillMs);
        printf("warm start: restore cached state = %.1f ms before first token"
               " (snapshot %.1f ms, paid once when the prefix is first seen)\n", restoreMs, snapMs);
        printf("setup before first token: %.1f ms -> %.1f ms  (%.0fx faster TTFT for a cached prefix)\n",
               prefillMs, restoreMs, prefillMs / restoreMs);
        printf("warm decode: %.1f ms for %u tokens; warm stream identical to cold: %s\n",
               warmMs, nGen, eq ? "YES" : "NO");
        if (!eq) {
            printf("WARM:");
            for (int id : warm) printf(" %d", id);
            printf("\n");
        }
    }
    vkUnmapMemory(c.dev, bRb.mem);

    vkUnmapMemory(c.dev, stage.mem);
    vkDestroyDescriptorPool(c.dev, dpool, nullptr);
    for (Pipe* pp : {&pRms, &pGemvA, &pAb, &pConvN, &pStep, &pGate, &pGemvO, &pAddN, &pAdd,
                     &pPrep, &pAttn, &pMoeL, &pMoeS, &pMoeGU, &pMoeGUs, &pMoeDn4, &pMoeDn6,
                     &pMoeDnsB, &pAdd3, &pHead, &pAm1, &pAm2, &pEmb})
        destroyPipe(c, *pp);
    for (auto& L : layers)
        for (auto& b : L.bufs) destroyBuf(c, b);
    for (auto& b : cache1) destroyBuf(c, b);
    for (auto& b : cache2) destroyBuf(c, b);
    if (warmDemo) destroyBuf(c, cacheLogits);
    for (Buf* b : {&bXin, &bXn, &bBig, &bMid, &bKin, &bVin, &bGb, &bConvOut, &bO, &bAtt,
                   &bAttnOut, &bY, &bXn2, &bML, &bMH, &bMSel, &bMY, &bMY2, &bOut, &bONorm,
                   &bHeadW, &bLogits, &bEmbdW, &bPids, &bAV, &bAI, &bTok, &bRb, &stage})
        destroyBuf(c, *b);
    return true;
}

// ===================== qk.h C ABI: persistent per-slot engine =====================
// Server-oriented engine. N slots each hold their own sequence (prompt cursor +
// position + sampled token); one batched dispatch (z = slot) advances all of
// them. Positions come from a per-slot buffer (fa_*_srv shaders) and the host
// drives one step at a time, so slots at different positions — some prefilling,
// some decoding, started/finished at different times — coexist. This is the
// execution model a real server needs, distinct from `qk token`'s GPU-resident,
// baked-position path.

struct qk_engine {
    VkCtx c{};
    Gguf g;
    uint32_t nSlots = 0, nCtx = 0, chunkN = 0;
    uint32_t vocab = 0, eosTok = 248046, bosTok = 248044;
    static constexpr uint32_t nEmbd = 2048, chQkv = 8192, dIn = 4096, hV = 32, dS = 128, hK = 16;
    static constexpr uint32_t dh = 256, hQ = 16, hKV = 2, nRot = 64, nLayer = 40;
    float eps = 1e-6f;

    VkDescriptorPool dpool = VK_NULL_HANDLE;
    Pipe pRms, pGemvA, pAb, pConvN, pStep, pGate, pGemvO, pAddN, pPrep, pAttn, pMoeL,
        pMoeS, pMoeGU, pMoeGUs, pMoeDn4, pMoeDn6, pMoeDnsB, pAdd3, pHead, pAm1, pAm2, pEmb;

    struct Layer {
        bool rec = false, downQ6 = false;
        std::vector<Buf> bufs;
        VkDescriptorSet sRms, sP1, sP2, sP3, sAb, sConv, sStep, sGate, sWo, sAddN,
            sPrep, sAttn, sMoeL, sMoeS, sMoeGU, sMoeGUs, sMoeDn, sMoeDns, sAdd3;
        VkBuffer aNormBuf = VK_NULL_HANDLE;
        // per-slot recurrent state (must be zeroed when a slot is reused):
        // rec -> (conv window, delta-rule S); attn -> (K cache, V cache).
        VkBuffer st1 = VK_NULL_HANDLE, st2 = VK_NULL_HANDLE;
        size_t ps1 = 0, ps2 = 0;  // per-slot byte stride
    };
    std::vector<Layer> layers;

    void resetSlot(uint32_t slot);  // zero one slot's recurrent state before reuse

    Buf bXin, bXn, bBig, bMid, bKin, bVin, bGb, bConvOut, bO, bAtt, bAttnOut, bY, bXn2,
        bML, bMH, bMSel, bMY, bMY2, bONorm, bHeadW, bLogits, bEmbdW, bRope, bAV, bAI,
        bTok, bSlotIn, bSlotPos, bSamp, stage;
    VkDescriptorSet sHead, sAm1, sAm2, sEmb;
    // One pre-recorded step CB per dispatch depth: stepCBs[z-1] dispatches z
    // slots. Submitting the one matching the highest active slot avoids paying
    // the (bandwidth-bound) weight re-reads for idle slots on light load.
    std::vector<VkCommandBuffer> stepCBs;
    uint32_t *slotInMap = nullptr, *slotPosMap = nullptr, *sampMap = nullptr;

    struct Slot {
        bool active = false;
        std::vector<uint32_t> prompt;      // full prompt of the current request
        std::vector<uint32_t> genTokens;   // tokens generated so far (for the cache key)
        uint32_t cursor = 0, pos = 0, gen = 0, maxGen = 0, last = 0;
    };
    std::vector<Slot> slots;

    // Prefix cache: after a request finishes, its recurrent state (delta-rule S +
    // conv + K/V for positions [0,pos)) is snapshotted to a host-resident buffer
    // keyed by the full token sequence. A new request whose prompt starts with a
    // cached sequence restores it and prefills only the divergent suffix — so a
    // multi-turn conversation re-processes only each new turn, not the whole history.
    struct CacheEntry {
        std::vector<uint32_t> tokens;  // full processed sequence (prompt + generated)
        Buf snap;                      // host-visible copy of all st1/st2 slot stripes
        uint64_t lru = 0;
        bool valid = false;
    };
    std::vector<CacheEntry> pcache;
    std::vector<size_t> snapOff1, snapOff2;  // per-layer byte offset of st1/st2 in the snapshot
    size_t snapSize = 0;
    uint64_t lruClock = 0;

    bool open(const char* path, const qk_config& cfg, char* err, size_t errLen);
    int stepChunk(uint32_t* outTok, uint32_t* outCnt, uint32_t* outFin);
    void snapshotSlot(uint32_t slot);           // save slot state -> LRU cache entry
    int matchPrefix(const uint32_t* prompt, uint32_t n);  // longest cached prefix, or -1
    void restoreInto(uint32_t slot, int cacheIdx);        // cache entry -> slot state
    void copyStripes(uint32_t slot, VkBuffer snapBuf, bool save);  // stripes <-> snapshot
    ~qk_engine();
};

qk_engine::~qk_engine() {
    if (c.dev == VK_NULL_HANDLE) return;  // never fully opened
    vkDeviceWaitIdle(c.dev);
    for (auto& L : layers)
        for (auto& b : L.bufs) destroyBuf(c, b);
    for (auto& e : pcache) destroyBuf(c, e.snap);
    for (Buf* b : {&bXin, &bXn, &bBig, &bMid, &bKin, &bVin, &bGb, &bConvOut, &bO, &bAtt,
                   &bAttnOut, &bY, &bXn2, &bML, &bMH, &bMSel, &bMY, &bMY2, &bONorm, &bHeadW,
                   &bLogits, &bEmbdW, &bRope, &bAV, &bAI, &bTok, &bSlotIn, &bSlotPos, &bSamp, &stage})
        destroyBuf(c, *b);
    if (dpool) vkDestroyDescriptorPool(c.dev, dpool, nullptr);
    for (Pipe* pp : {&pRms, &pGemvA, &pAb, &pConvN, &pStep, &pGate, &pGemvO, &pAddN, &pPrep,
                     &pAttn, &pMoeL, &pMoeS, &pMoeGU, &pMoeGUs, &pMoeDn4, &pMoeDn6, &pMoeDnsB,
                     &pAdd3, &pHead, &pAm1, &pAm2, &pEmb})
        destroyPipe(c, *pp);
    if (c.pool) vkDestroyCommandPool(c.dev, c.pool, nullptr);
    if (c.pl) vkDestroyPipelineLayout(c.dev, c.pl, nullptr);
    if (c.dsl) vkDestroyDescriptorSetLayout(c.dev, c.dsl, nullptr);
    vkDestroyDevice(c.dev, nullptr);
    if (c.inst) vkDestroyInstance(c.inst, nullptr);
    c.dev = VK_NULL_HANDLE;
}

// Zero one slot's recurrent state (delta-rule S, conv window, K/V cache) so a
// reused slot does not inherit the previous request's state.
void qk_engine::resetSlot(uint32_t slot) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    for (auto& L : layers) {
        vkCmdFillBuffer(c.cb, L.st1, (VkDeviceSize)slot * L.ps1, L.ps1, 0u);
        vkCmdFillBuffer(c.cb, L.st2, (VkDeviceSize)slot * L.ps2, L.ps2, 0u);
    }
    VK_CHECK(vkEndCommandBuffer(c.cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cb;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
}

// Copy every layer's per-slot state stripe between the slot and a snapshot
// buffer (save=true: slot -> snapshot; save=false: snapshot -> slot).
void qk_engine::copyStripes(uint32_t slot, VkBuffer snapBuf, bool save) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        VkBufferCopy a, b;
        if (save) {
            a = {(VkDeviceSize)slot * L.ps1, snapOff1[il], L.ps1};
            b = {(VkDeviceSize)slot * L.ps2, snapOff2[il], L.ps2};
            vkCmdCopyBuffer(c.cb, L.st1, snapBuf, 1, &a);
            vkCmdCopyBuffer(c.cb, L.st2, snapBuf, 1, &b);
        } else {
            a = {snapOff1[il], (VkDeviceSize)slot * L.ps1, L.ps1};
            b = {snapOff2[il], (VkDeviceSize)slot * L.ps2, L.ps2};
            vkCmdCopyBuffer(c.cb, snapBuf, L.st1, 1, &a);
            vkCmdCopyBuffer(c.cb, snapBuf, L.st2, 1, &b);
        }
    }
    VK_CHECK(vkEndCommandBuffer(c.cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cb;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
}

// Snapshot a finished slot's state into an LRU cache entry, keyed by its full
// token sequence (prompt + generated).
void qk_engine::snapshotSlot(uint32_t slot) {
    if (pcache.empty()) return;
    Slot& sl = slots[slot];
    // The state reflects exactly `pos` processed tokens: prompt + the generated
    // tokens that have been fed back in (all but the last, unless a decode-step
    // EOS fed it). Key the snapshot by exactly those `pos` tokens.
    size_t fedGen = sl.pos > sl.prompt.size() ? sl.pos - sl.prompt.size() : 0;
    if (fedGen > sl.genTokens.size()) fedGen = sl.genTokens.size();
    if (sl.pos < 8 || sl.pos > nCtx) return;  // too short to cache, or beyond capacity
    int idx = 0;
    for (uint32_t i = 0; i < pcache.size(); i++) {
        if (!pcache[i].valid) { idx = (int)i; break; }
        if (pcache[i].lru < pcache[idx].lru) idx = (int)i;
    }
    CacheEntry& e = pcache[idx];
    e.tokens = sl.prompt;
    e.tokens.insert(e.tokens.end(), sl.genTokens.begin(), sl.genTokens.begin() + fedGen);
    e.lru = ++lruClock;
    e.valid = true;
    copyStripes(slot, e.snap.buf, /*save=*/true);
}

// Longest cached sequence that is a strict prefix of prompt[0,n); -1 if none.
int qk_engine::matchPrefix(const uint32_t* prompt, uint32_t n) {
    int best = -1;
    uint32_t bestLen = 8;  // require at least 8 shared tokens to bother
    for (uint32_t i = 0; i < pcache.size(); i++) {
        if (!pcache[i].valid) continue;
        const auto& tk = pcache[i].tokens;
        uint32_t L = (uint32_t)tk.size();
        if (L >= n || L <= bestLen) continue;  // need L < n (>=1 to prefill) and longer than best
        bool ok = true;
        for (uint32_t j = 0; j < L; j++)
            if (tk[j] != prompt[j]) { ok = false; break; }
        if (ok) { best = (int)i; bestLen = L; }
    }
    return best;
}

void qk_engine::restoreInto(uint32_t slot, int cacheIdx) {
    pcache[cacheIdx].lru = ++lruClock;
    copyStripes(slot, pcache[cacheIdx].snap.buf, /*save=*/false);
}

bool qk_engine::open(const char* path, const qk_config& cfg, char* err, size_t errLen) {
    auto fail = [&](const char* m) { if (err && errLen) snprintf(err, errLen, "%s", m); return false; };
    nSlots = cfg.n_slots; nCtx = cfg.n_ctx; chunkN = cfg.chunk;
    // fa_attn_srv is flash-attention (tiled) now, so nCtx is bounded by KV-cache
    // VRAM (~32K at 4 slots on a 20 GB card), not shared memory.
    if (nSlots < 1 || nSlots > 16 || nCtx < 64 || nCtx > 32768 || chunkN < 1 || chunkN > 32)
        return fail("qk_open: bad config");
    initVk(c, "libqk");  // shader dir resolved via QK_SHADER_DIR
    if (!g.open(path)) return fail("qk_open: cannot open GGUF");
    const GgufTensor* tEmbd = g.find("token_embd.weight");
    const GgufTensor* tONorm = g.find("output_norm.weight");
    const GgufTensor* tHead = g.find("output.weight");
    if (!tEmbd || !tONorm || !tHead || tEmbd->type != GGML_Q6_K || tHead->type != GGML_Q6_K)
        return fail("qk_open: missing/unexpected embd/head tensors");
    vocab = (uint32_t)tHead->ne[1];
    const size_t rbQ8e = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbQ8i = ggmlRowBytes(GGML_Q8_0, dIn);
    const size_t rbE = ggmlRowBytes(GGML_Q6_K, nEmbd);
    const uint32_t nB = nSlots, tmax = nCtx;

    pRms = makePipe(c, "rmsnorm.spv", 3, 8);
    pGemvA = makePipe(c, "gemv_q8_0.spv", 3, 8, 64);
    pAb = makePipe(c, "dn_ab.spv", 6, 8);
    pConvN = makePipe(c, "dn_convn.spv", 4, 16);
    pStep = makePipe(c, "dn_step.spv", 4, 12);
    pGate = makePipe(c, "dn_gate.spv", 4, 12);
    pGemvO = makePipe(c, "gemv_q8_0.spv", 3, 8, 128);
    pAddN = makePipe(c, "add_rmsnorm.spv", 5, 8);
    pPrep = makePipe(c, "fa_prep_srv.spv", 10, 32);
    pAttn = makePipe(c, "fa_attn_srv.spv", 6, 32);
    pMoeL = makePipe(c, "moe_logits.spv", 3, 16);
    pMoeS = makePipe(c, "moe_select.spv", 4, 16);
    pMoeGU = makePipe(c, "moe_gateup_iq3.spv", 5, 16);
    pMoeGUs = makePipe(c, "moe_gateup_q8.spv", 4, 16);
    pMoeDn4 = makePipe(c, "moe_down_iq4.spv", 4, 16);
    pMoeDn6 = makePipe(c, "moe_down_q6k.spv", 4, 16);
    pMoeDnsB = makePipe(c, "moe_down_q8b.spv", 4, 16);
    pAdd3 = makePipe(c, "add_rms3.spv", 6, 8);
    pHead = makePipe(c, "gemv_q6_k.spv", 3, 8, 128);
    pAm1 = makePipe(c, "argmax1.spv", 3, 8);
    pAm2 = makePipe(c, "argmax2.spv", 3, 4);
    pEmb = makePipe(c, "embed_q6k.spv", 3, 12);

    const VkBufferUsageFlags stor = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags storSrc = stor | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bXin = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bXn = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bBig = createBuf(c, (size_t)nB * chQkv * 4, stor, true);
    bMid = createBuf(c, (size_t)nB * dIn * 4, stor, true);
    bKin = createBuf(c, (size_t)nB * hKV * dh * 4, stor, true);
    bVin = createBuf(c, (size_t)nB * hKV * dh * 4, stor, true);
    bGb = createBuf(c, (size_t)nB * 2 * hV * 4, stor, true);
    bConvOut = createBuf(c, (size_t)nB * chQkv * 4, stor, true);
    bO = createBuf(c, (size_t)nB * dIn * 4, stor, true);
    bAtt = createBuf(c, (size_t)nB * dIn * 4, stor, true);
    bAttnOut = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bY = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bXn2 = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bML = createBuf(c, (size_t)nB * 256 * 4, stor, true);
    bMH = createBuf(c, (size_t)nB * 9 * 512 * 4, stor, true);
    bMSel = createBuf(c, (size_t)nB * 128, stor, true);
    bMY = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bMY2 = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bONorm = createBuf(c, nEmbd * 4, stor, true);
    bHeadW = createBuf(c, (size_t)vocab * rbE, stor, true);
    bLogits = createBuf(c, (size_t)nB * vocab * 4, storSrc, true);
    bEmbdW = createBuf(c, (size_t)vocab * rbE, stor, true);
    bRope = createBuf(c, (size_t)tmax * (nRot / 2) * 2 * 4, stor, true);
    bAV = createBuf(c, (size_t)nB * 64 * 4, stor, true);
    bAI = createBuf(c, (size_t)nB * 64 * 4, stor, true);
    bTok = createBuf(c, (size_t)nB * 4, storSrc, true);
    bSlotIn = createBuf(c, (size_t)nB * 4, stor, false);   // host-visible, host-written each step
    bSlotPos = createBuf(c, (size_t)nB * 4, stor, false);
    bSamp = createBuf(c, (size_t)nB * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);  // readback
    stage = createBuf(c, 160u << 20,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    VK_CHECK(vkMapMemory(c.dev, bSlotIn.mem, 0, VK_WHOLE_SIZE, 0, (void**)&slotInMap));
    VK_CHECK(vkMapMemory(c.dev, bSlotPos.mem, 0, VK_WHOLE_SIZE, 0, (void**)&slotPosMap));
    VK_CHECK(vkMapMemory(c.dev, bSamp.mem, 0, VK_WHOLE_SIZE, 0, (void**)&sampMap));
    for (uint32_t s = 0; s < nB; s++) { slotInMap[s] = 0; slotPosMap[s] = 0; }

    void* mapped;
    VK_CHECK(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    auto begin = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    };
    auto submitWait = [&]() {
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    };
    auto upload = [&](Buf& dst, const void* src, size_t n) {
        for (size_t off = 0; off < n; off += (140u << 20)) {
            size_t chunk = std::min(n - off, (size_t)(140u << 20));
            if (src) memcpy(mapped, (const uint8_t*)src + off, chunk); else memset(mapped, 0, chunk);
            begin();
            VkBufferCopy cp{0, off, chunk};
            vkCmdCopyBuffer(c.cb, stage.buf, dst.buf, 1, &cp);
            submitWait();
        }
    };
    upload(bONorm, tONorm->data, nEmbd * 4);
    {
        const uint32_t half = nRot / 2;
        std::vector<float> rope((size_t)tmax * half * 2);
        for (uint32_t p = 0; p < tmax; p++)
            for (uint32_t j = 0; j < half; j++) {
                float th = (float)p * std::pow(kFreqBase, -2.f * (float)j / (float)nRot);
                rope[2 * ((size_t)p * half + j)] = std::cos(th);
                rope[2 * ((size_t)p * half + j) + 1] = std::sin(th);
            }
        upload(bRope, rope.data(), rope.size() * 4);
    }
    upload(bHeadW, tHead->data, (size_t)vocab * rbE);
    upload(bEmbdW, tEmbd->data, (size_t)vocab * rbE);

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1024; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &dpool));
    auto mkSet = [&](Pipe& pp, std::vector<VkBuffer> bufs) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &pp.dsl;
        VkDescriptorSet ds;
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &ai, &ds));
        std::vector<VkDescriptorBufferInfo> dbi(bufs.size());
        std::vector<VkWriteDescriptorSet> wr(bufs.size());
        for (size_t i = 0; i < bufs.size(); i++) {
            dbi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[i].dstSet = ds; wr[i].dstBinding = (uint32_t)i; wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(c.dev, (uint32_t)wr.size(), wr.data(), 0, nullptr);
        return ds;
    };

    layers.resize(nLayer);
    char nb[128];
    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        auto T = [&](const char* suffix) -> const GgufTensor* {
            snprintf(nb, sizeof nb, "blk.%u.%s", il, suffix); return g.find(nb);
        };
        auto W = [&](const GgufTensor* t, size_t n) -> VkBuffer {
            L.bufs.push_back(createBuf(c, n, stor, true));
            upload(L.bufs.back(), t ? t->data : nullptr, n);
            return L.bufs.back().buf;
        };
        MoeT moe;
        if (!loadMoeT(g, il, moe)) return fail("qk_open: MoE tensors missing");
        L.downQ6 = moe.downQ6;
        L.rec = T("ssm_a") != nullptr;
        VkBuffer aNorm = W(T("attn_norm.weight"), nEmbd * 4);
        VkBuffer pn = W(T("post_attention_norm.weight"), nEmbd * 4);
        L.aNormBuf = aNorm;
        if (L.rec) {
            VkBuffer qkvW = W(T("attn_qkv.weight"), (size_t)chQkv * rbQ8e);
            VkBuffer zW = W(T("attn_gate.weight"), (size_t)dIn * rbQ8e);
            VkBuffer alW = W(T("ssm_alpha.weight"), (size_t)hV * nEmbd * 4);
            VkBuffer beW = W(T("ssm_beta.weight"), (size_t)hV * nEmbd * 4);
            VkBuffer dt = W(T("ssm_dt.bias"), hV * 4);
            VkBuffer av = W(T("ssm_a"), hV * 4);
            VkBuffer ker = W(T("ssm_conv1d.weight") ? T("ssm_conv1d.weight") : T("ssm_conv1d"),
                             (size_t)chQkv * 4 * 4);
            VkBuffer sn = W(T("ssm_norm.weight"), dS * 4);
            VkBuffer outW = W(T("ssm_out.weight"), (size_t)nEmbd * rbQ8i);
            VkBuffer convSt = W(nullptr, (size_t)nB * chQkv * 3 * 4);
            VkBuffer S = W(nullptr, (size_t)nB * hV * dS * dS * 4);
            L.st1 = convSt; L.ps1 = (size_t)chQkv * 3 * 4;
            L.st2 = S;      L.ps2 = (size_t)hV * dS * dS * 4;
            L.sRms = mkSet(pRms, {bXin.buf, aNorm, bXn.buf});
            L.sP1 = mkSet(pGemvA, {qkvW, bXn.buf, bBig.buf});
            L.sP2 = mkSet(pGemvA, {zW, bXn.buf, bMid.buf});
            L.sAb = mkSet(pAb, {bXn.buf, alW, beW, dt, av, bGb.buf});
            L.sConv = mkSet(pConvN, {convSt, bBig.buf, ker, bConvOut.buf});
            L.sStep = mkSet(pStep, {bConvOut.buf, bGb.buf, S, bO.buf});
            L.sGate = mkSet(pGate, {bO.buf, sn, bMid.buf, bAtt.buf});
            L.sWo = mkSet(pGemvO, {outW, bAtt.buf, bAttnOut.buf});
        } else {
            VkBuffer wq = W(T("attn_q.weight"), (size_t)chQkv * rbQ8e);
            VkBuffer wk = W(T("attn_k.weight"), (size_t)hKV * dh * rbQ8e);
            VkBuffer wv = W(T("attn_v.weight"), (size_t)hKV * dh * rbQ8e);
            VkBuffer qn = W(T("attn_q_norm.weight"), dh * 4);
            VkBuffer kn = W(T("attn_k_norm.weight"), dh * 4);
            VkBuffer wo = W(T("attn_output.weight"), (size_t)nEmbd * rbQ8i);
            VkBuffer kc = W(nullptr, (size_t)nB * hKV * tmax * dh * 4);
            VkBuffer vc = W(nullptr, (size_t)nB * hKV * tmax * dh * 4);
            L.st1 = kc; L.ps1 = (size_t)hKV * tmax * dh * 4;
            L.st2 = vc; L.ps2 = (size_t)hKV * tmax * dh * 4;
            L.sRms = mkSet(pRms, {bXin.buf, aNorm, bXn.buf});
            L.sP1 = mkSet(pGemvA, {wq, bXn.buf, bBig.buf});
            L.sP2 = mkSet(pGemvA, {wk, bXn.buf, bKin.buf});
            L.sP3 = mkSet(pGemvA, {wv, bXn.buf, bVin.buf});
            L.sPrep = mkSet(pPrep, {bBig.buf, bKin.buf, bVin.buf, qn, kn, bMid.buf, kc, vc,
                                    bRope.buf, bSlotPos.buf});
            L.sAttn = mkSet(pAttn, {bMid.buf, kc, vc, bBig.buf, bAtt.buf, bSlotPos.buf});
            L.sWo = mkSet(pGemvO, {wo, bAtt.buf, bAttnOut.buf});
        }
        VkBuffer mgi = W(moe.gi, (size_t)moe.nExp * nEmbd * 4);
        VkBuffer mgis = W(moe.gis, nEmbd * 4);
        VkBuffer mge = W(moe.ge, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        VkBuffer mue = W(moe.ue, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        VkBuffer mde = W(moe.de, (size_t)moe.nExp * moe.nEmbd * moe.rbDE);
        VkBuffer mgs = W(moe.gs, (size_t)moe.nFf * moe.rbGS);
        VkBuffer mus = W(moe.us, (size_t)moe.nFf * moe.rbGS);
        VkBuffer mds = W(moe.ds, (size_t)moe.nEmbd * moe.rbDS);
        L.sAddN = mkSet(pAddN, {bXin.buf, bAttnOut.buf, pn, bY.buf, bXn2.buf});
        L.sMoeL = mkSet(pMoeL, {mgi, bXn2.buf, bML.buf});
        L.sMoeS = mkSet(pMoeS, {bML.buf, mgis, bXn2.buf, bMSel.buf});
        L.sMoeGU = mkSet(pMoeGU, {mge, mue, bXn2.buf, bMSel.buf, bMH.buf});
        L.sMoeGUs = mkSet(pMoeGUs, {mgs, mus, bXn2.buf, bMH.buf});
        L.sMoeDn = mkSet(L.downQ6 ? pMoeDn6 : pMoeDn4, {mde, bMH.buf, bMSel.buf, bMY.buf});
        L.sMoeDns = mkSet(pMoeDnsB, {mds, bMH.buf, bMSel.buf, bMY2.buf});
    }
    for (uint32_t il = 0; il < nLayer; il++)
        layers[il].sAdd3 = mkSet(pAdd3, {bY.buf, bMY.buf, bMY2.buf,
                                         il + 1 < nLayer ? layers[il + 1].aNormBuf : bONorm.buf,
                                         bXin.buf, bXn.buf});
    sHead = mkSet(pHead, {bHeadW.buf, bXn.buf, bLogits.buf});
    sAm1 = mkSet(pAm1, {bLogits.buf, bAV.buf, bAI.buf});
    sAm2 = mkSet(pAm2, {bAV.buf, bAI.buf, bTok.buf});
    sEmb = mkSet(pEmb, {bEmbdW.buf, bSlotIn.buf, bXin.buf});

    // ---- prefix cache: host-resident snapshots of the per-slot state stripes ----
    {
        snapOff1.resize(nLayer);
        snapOff2.resize(nLayer);
        size_t off = 0;
        for (uint32_t il = 0; il < nLayer; il++) {
            snapOff1[il] = off; off += layers[il].ps1;
            snapOff2[il] = off; off += layers[il].ps2;
        }
        snapSize = off;
        const uint32_t PCACHE_N = 3;  // conversation snapshots kept (LRU)
        pcache.resize(PCACHE_N);
        for (auto& e : pcache)
            e.snap = createBuf(c, snapSize,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               false);  // host-visible
    }

    // ---- record one re-submittable step CB per dispatch depth (z = 1..nSlots) ----
    stepCBs.resize(nSlots);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = c.pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = nSlots;
    VK_CHECK(vkAllocateCommandBuffers(c.dev, &cbai, stepCBs.data()));
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VkCommandBuffer rcb = VK_NULL_HANDLE;  // set per z below
    uint32_t zdim = nB;                    // dispatch depth for the current recording
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    auto barrier = [&]() {
        vkCmdPipelineBarrier(rcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    auto disp = [&](Pipe& pp, VkDescriptorSet ds, uint32_t wgs, const void* pc, uint32_t pcSize) {
        vkCmdBindPipeline(rcb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.p);
        vkCmdBindDescriptorSets(rcb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(rcb, pp.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
        uint32_t gx = std::min(wgs, c.props.limits.maxComputeWorkGroupCount[0]);
        uint32_t gy = (wgs + gx - 1) / gx;
        vkCmdDispatch(rcb, gx, gy, zdim);
    };
    struct { uint32_t n; float e; } pcRms{nEmbd, eps};
    struct { uint32_t m, k; } pcQkv{chQkv, nEmbd}, pcZ{dIn, nEmbd}, pcKV{hKV * dh, nEmbd},
        pcWo{nEmbd, dIn}, pcHead{vocab, nEmbd};
    struct { uint32_t n, h; } pcAb{nEmbd, hV};
    struct { uint32_t ch, d, qkch; float e; } pcConvN{chQkv, dS, 2 * hK * dS, eps};
    struct { uint32_t d, hk, hv; } pcStep{dS, hK, hV};
    struct { uint32_t d, hv; float e; } pcGate{dS, hV, eps};
    struct { uint32_t a, b, cc, d; } pcv{nEmbd, 512, 256, 8};
    struct { uint32_t pos, tmax, dh, nRot, hQ, hKV_; float eps, fb; }
        pcFa{0, tmax, dh, nRot, hQ, hKV, eps, kFreqBase};
    const uint32_t amWgs = (vocab + 4095) / 4096;
    struct { uint32_t n, span; } pcAm{vocab, 4096};
    struct { uint32_t m; } pcAm2{amWgs};
    struct { uint32_t k, idx, pr; } pcE{nEmbd, 0, 1};  // idx0 + perReq1 -> ids[rq] = slotInput[rq]

    for (uint32_t z = 1; z <= nSlots; z++) {
    zdim = z;
    rcb = stepCBs[z - 1];
    VK_CHECK(vkBeginCommandBuffer(rcb, &cbbi));
    barrier();
    disp(pEmb, sEmb, 1, &pcE, 12);
    barrier();
    disp(pRms, layers[0].sRms, 1, &pcRms, 8);
    barrier();
    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        if (L.rec) {
            disp(pGemvA, L.sP1, (chQkv + 3) / 4, &pcQkv, 8);
            disp(pGemvA, L.sP2, (dIn + 3) / 4, &pcZ, 8);
            disp(pAb, L.sAb, 2 * hV, &pcAb, 8);
            barrier();
            disp(pConvN, L.sConv, chQkv / dS, &pcConvN, 16);
            barrier();
            disp(pStep, L.sStep, hV, &pcStep, 12);
            barrier();
            disp(pGate, L.sGate, hV, &pcGate, 12);
            barrier();
            disp(pGemvO, L.sWo, (nEmbd + 1) / 2, &pcWo, 8);
        } else {
            disp(pGemvA, L.sP1, (chQkv + 3) / 4, &pcQkv, 8);
            disp(pGemvA, L.sP2, (hKV * dh + 3) / 4, &pcKV, 8);
            disp(pGemvA, L.sP3, (hKV * dh + 3) / 4, &pcKV, 8);
            barrier();
            disp(pPrep, L.sPrep, hQ + 2 * hKV, &pcFa, 32);
            barrier();
            disp(pAttn, L.sAttn, hQ, &pcFa, 32);
            barrier();
            disp(pGemvO, L.sWo, (nEmbd + 1) / 2, &pcWo, 8);
        }
        barrier();
        disp(pAddN, L.sAddN, 1, &pcRms, 8);
        barrier();
        disp(pMoeL, L.sMoeL, 256, &pcv, 16);
        disp(pMoeGUs, L.sMoeGUs, 512, &pcv, 16);
        barrier();
        disp(pMoeS, L.sMoeS, 1, &pcv, 16);
        barrier();
        disp(pMoeGU, L.sMoeGU, 8 * 512, &pcv, 16);
        barrier();
        disp(L.downQ6 ? pMoeDn6 : pMoeDn4, L.sMoeDn, nEmbd, &pcv, 16);
        disp(pMoeDnsB, L.sMoeDns, nEmbd, &pcv, 16);
        barrier();
        disp(pAdd3, L.sAdd3, 1, &pcRms, 8);
        barrier();
    }
    disp(pHead, sHead, (vocab + 1) / 2, &pcHead, 8);
    barrier();
    disp(pAm1, sAm1, amWgs, &pcAm, 8);
    barrier();
    disp(pAm2, sAm2, 1, &pcAm2, 4);
    VkMemoryBarrier m2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    m2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(rcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &m2, 0, nullptr, 0, nullptr);
    VkBufferCopy ct{0, 0, (size_t)nB * 4};
    vkCmdCopyBuffer(rcb, bTok.buf, bSamp.buf, 1, &ct);
    VK_CHECK(vkEndCommandBuffer(rcb));
    }  // for z = 1..nSlots

    slots.resize(nSlots);
    return true;
}

// Advance every active slot by up to chunkN steps. Returns active-at-entry.
int qk_engine::stepChunk(uint32_t* outTok, uint32_t* outCnt, uint32_t* outFin) {
    for (uint32_t s = 0; s < nSlots; s++) outCnt[s] = 0;
    *outFin = 0;
    int activeAtEntry = 0;
    for (uint32_t s = 0; s < nSlots; s++) if (slots[s].active) activeAtEntry++;
    if (!activeAtEntry) return 0;

    for (uint32_t step = 0; step < chunkN; step++) {
        int nAct = 0;
        uint32_t maxZ = 0;  // highest active slot index + 1 = needed dispatch depth
        for (uint32_t s = 0; s < nSlots; s++) {
            Slot& sl = slots[s];
            if (!sl.active) { slotInMap[s] = 0; slotPosMap[s] = 0; continue; }
            nAct++; maxZ = s + 1;
            if (sl.cursor < sl.prompt.size()) {         // prefill: feed prompt[cursor] at position cursor
                slotInMap[s] = sl.prompt[sl.cursor];
                slotPosMap[s] = sl.cursor;
            } else {                                    // decode: feed last sampled at pos
                slotInMap[s] = sl.last;
                slotPosMap[s] = sl.pos;
            }
        }
        if (!nAct) break;

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &stepCBs[maxZ - 1];  // dispatch only up to the top active slot
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));

        for (uint32_t s = 0; s < nSlots; s++) {
            Slot& sl = slots[s];
            if (!sl.active) continue;
            uint32_t sampled = sampMap[s];
            bool prefilling = sl.cursor < sl.prompt.size();
            if (prefilling && sl.cursor + 1 < sl.prompt.size()) {
                sl.cursor++;  // still consuming prompt; ignore this logit
                continue;
            }
            // this step produced a real generated token (last prompt token, or a decode step)
            if (prefilling) { sl.cursor = (uint32_t)sl.prompt.size(); sl.pos = (uint32_t)sl.prompt.size(); }
            if (sampled == eosTok) {
                if (!prefilling) sl.pos++;  // a decode-step EOS still fed a token (its K/V is written)
                snapshotSlot(s);            // cache before the next step overwrites this slot
                sl.active = false; *outFin |= 1u << s;
                continue;
            }
            outTok[s * chunkN + outCnt[s]++] = sampled;
            sl.genTokens.push_back(sampled);
            sl.last = sampled;
            sl.gen++;
            if (!prefilling) sl.pos++;
            if (sl.pos >= nCtx || sl.gen >= sl.maxGen) {
                snapshotSlot(s);
                sl.active = false; *outFin |= 1u << s;
            }
        }
    }
    return activeAtEntry;
}

extern "C" {

__attribute__((visibility("default")))
qk_engine* qk_open(const char* gguf_path, const qk_config* cfg, char* err, size_t err_len) {
    if (!gguf_path || !cfg) { if (err && err_len) snprintf(err, err_len, "qk_open: null arg"); return nullptr; }
    qk_engine* e = new (std::nothrow) qk_engine();
    if (!e) { if (err && err_len) snprintf(err, err_len, "qk_open: oom"); return nullptr; }
    if (!e->open(gguf_path, *cfg, err, err_len)) { delete e; return nullptr; }
    return e;
}

__attribute__((visibility("default"))) void qk_close(qk_engine* e) { delete e; }
__attribute__((visibility("default"))) uint32_t qk_n_vocab(const qk_engine* e) { return e->vocab; }
__attribute__((visibility("default"))) uint32_t qk_n_ctx(const qk_engine* e) { return e->nCtx; }
__attribute__((visibility("default"))) uint32_t qk_n_slots(const qk_engine* e) { return e->nSlots; }
__attribute__((visibility("default"))) uint32_t qk_chunk(const qk_engine* e) { return e->chunkN; }
__attribute__((visibility("default"))) uint32_t qk_eos_token(const qk_engine* e) { return e->eosTok; }
__attribute__((visibility("default"))) uint32_t qk_bos_token(const qk_engine* e) { return e->bosTok; }

__attribute__((visibility("default")))
int qk_slot_start(qk_engine* e, uint32_t slot, const uint32_t* prompt, uint32_t n_prompt, uint32_t max_gen) {
    if (!e || slot >= e->nSlots) return -1;
    if (e->slots[slot].active) return -2;
    if (!prompt || n_prompt < 1 || n_prompt + max_gen > e->nCtx) return -3;
    for (uint32_t i = 0; i < n_prompt; i++) if (prompt[i] >= e->vocab) return -4;
    qk_engine::Slot& s = e->slots[slot];
    int cidx = e->matchPrefix(prompt, n_prompt);
    if (cidx >= 0) {
        // Reuse a cached prefix: restore its state and prefill only the suffix.
        e->restoreInto(slot, cidx);
        uint32_t L = (uint32_t)e->pcache[cidx].tokens.size();
        s.cursor = L; s.pos = L;
    } else {
        e->resetSlot(slot);  // clear any prior occupant's recurrent state
        s.cursor = 0; s.pos = 0;
    }
    s.active = true; s.prompt.assign(prompt, prompt + n_prompt);
    s.genTokens.clear();
    s.gen = 0; s.maxGen = max_gen; s.last = 0;
    return 0;
}

__attribute__((visibility("default")))
void qk_slot_cancel(qk_engine* e, uint32_t slot) {
    if (e && slot < e->nSlots) { e->slots[slot].active = false; e->slots[slot].prompt.clear(); }
}

__attribute__((visibility("default")))
int qk_step_chunk(qk_engine* e, uint32_t* out_tokens, uint32_t* out_counts, uint32_t* out_finished) {
    if (!e || !out_tokens || !out_counts || !out_finished) return -1;
    return e->stepChunk(out_tokens, out_counts, out_finished);
}

}  // extern "C"

// Batched-prefill GEMM validation: Y[N,M] = X[N,K]·W[M,K]^T vs a CPU reference.
static bool caseBGemm(VkCtx& c, uint32_t M, uint32_t K, uint32_t N) {
    printf("\n== batched GEMM  Y[%u,%u] = X[%u,%u] . W[%u,%u]^T  (Q8_0, 16x16 tile) ==\n",
           N, M, N, K, M, K);
    if (K % 32) { fprintf(stderr, "K must be a multiple of 32\n"); return false; }
    size_t nb = (size_t)M * (K / 32);
    std::vector<block_q8_0> W(nb);
    std::mt19937 rng(7);
    for (auto& b : W) {
        b.d = qk_f32_to_f16(0.005f + 0.02f * (rng() & 0xFFFF) / 65536.0f);
        for (auto& q : b.qs) q = (int8_t)((int)(rng() % 255) - 127);
    }
    std::vector<float> X((size_t)N * K);
    for (auto& v : X) v = -1.f + 2.f * (rng() & 0xFFFF) / 65536.0f;

    std::vector<float> Yref((size_t)N * M), tmp(K);
    for (uint32_t m = 0; m < M; m++) {
        dequant_row_q8_0(&W[(size_t)m * (K / 32)], tmp.data(), K);
        for (uint32_t n = 0; n < N; n++) {
            double a = 0;
            const float* xr = &X[(size_t)n * K];
            for (uint32_t k = 0; k < K; k++) a += (double)tmp[k] * xr[k];
            Yref[(size_t)n * M + m] = (float)a;
        }
    }

    Pipe p = makePipe(c, "gemm_q8_0.spv", 3, 12);
    const VkBufferUsageFlags stor = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    Buf bW = createBuf(c, nb * sizeof(block_q8_0), stor, true);
    Buf bX = createBuf(c, (size_t)N * K * 4, stor, true);
    Buf bY = createBuf(c, (size_t)N * M * 4, stor | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    size_t maxN = std::max({nb * sizeof(block_q8_0), (size_t)N * K * 4, (size_t)N * M * 4});
    Buf stage = createBuf(c, maxN, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);

    auto begin = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    };
    auto submitWait = [&]() {
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    };
    void* mapped;
    VK_CHECK(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    auto upload = [&](Buf& dst, const void* src, size_t n) {
        memcpy(mapped, src, n);
        begin();
        VkBufferCopy cp{0, 0, n};
        vkCmdCopyBuffer(c.cb, stage.buf, dst.buf, 1, &cp);
        submitWait();
    };
    upload(bW, W.data(), nb * sizeof(block_q8_0));
    upload(bX, X.data(), (size_t)N * K * 4);

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool pool;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &pool));
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &p.dsl;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(c.dev, &ai, &ds));
    VkBuffer bufs[3] = {bW.buf, bX.buf, bY.buf};
    VkDescriptorBufferInfo dbi[3];
    VkWriteDescriptorSet wr[3];
    for (int i = 0; i < 3; i++) {
        dbi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
        wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wr[i].dstSet = ds; wr[i].dstBinding = (uint32_t)i; wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(c.dev, 3, wr, 0, nullptr);

    struct { uint32_t M, K, N; } pc{M, K, N};
    begin();
    vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
    vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(c.cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pc);
    auto t0 = std::chrono::steady_clock::now();
    vkCmdDispatch(c.cb, (M + 15) / 16, 1, (N + 15) / 16);
    submitWait();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    begin();
    VkBufferCopy cp{0, 0, (size_t)N * M * 4};
    vkCmdCopyBuffer(c.cb, bY.buf, stage.buf, 1, &cp);
    submitWait();
    const float* Y = (const float*)mapped;
    double maxErr = 0, sumsq = 0;
    for (size_t i = 0; i < (size_t)N * M; i++) {
        maxErr = std::max(maxErr, std::fabs((double)Y[i] - Yref[i]));
        sumsq += (double)Yref[i] * Yref[i];
    }
    double rms = std::sqrt(sumsq / ((size_t)N * M));
    bool ok = maxErr < 1e-3 * rms;  // fp32 kernel vs double reference, scale-relative
    printf("  %u tokens in one batched pass: %.2f ms | max abs err %.3g / rms %.3g = %.2g -> %s\n",
           N, ms, maxErr, rms, maxErr / rms, ok ? "PASS" : "FAIL");
    vkUnmapMemory(c.dev, stage.mem);
    vkDestroyDescriptorPool(c.dev, pool, nullptr);
    destroyPipe(c, p);
    for (Buf* b : {&bW, &bX, &bY, &stage}) destroyBuf(c, *b);
    return ok;
}

static void listTensors(const std::string& filter) {
    Gguf g;
    if (!g.open(ggufPath())) return;
    for (const auto& [name, t] : g.tensors()) {
        if (!filter.empty() && name.find(filter) == std::string::npos) continue;
        printf("%-44s %-8s ne=[%llu,%llu,%llu]\n", name.c_str(), ggmlTypeName(t.type),
               (unsigned long long)t.ne[0], (unsigned long long)t.ne[1],
               (unsigned long long)t.ne[2]);
    }
}

#ifndef QK_LIBRARY
int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "suite";

    if (mode == "list") {
        listTensors(argc > 2 ? argv[2] : "");
        return 0;
    }

    if (mode == "serve-test") {
        // Drive the qk.h C ABI in-process: every slot runs the SAME prompt, so
        // all streams must match each other AND the external greedy reference.
        if (argc < 4) {
            fprintf(stderr, "usage: qk serve-test <ids-file> <nGen> [nSlots] [tmax]\n");
            return 1;
        }
        std::vector<uint32_t> prompt;
        {
            FILE* f = fopen(argv[2], "r");
            if (!f) { perror(argv[2]); return 1; }
            int v;
            while (fscanf(f, "%d%*[, \n]", &v) == 1) prompt.push_back((uint32_t)v);
            fclose(f);
        }
        uint32_t nGen = (uint32_t)atoi(argv[3]);
        uint32_t nSlots = argc > 4 ? (uint32_t)atoi(argv[4]) : 1;
        uint32_t tmax = argc > 5 ? (uint32_t)atoi(argv[5]) : 128;
        qk_config cfg{nSlots, tmax, 8};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
        for (uint32_t s = 0; s < nSlots; s++)
            qk_slot_start(e, s, prompt.data(), (uint32_t)prompt.size(), nGen);
        uint32_t ch = qk_chunk(e);
        std::vector<std::vector<uint32_t>> gen(nSlots);
        std::vector<uint32_t> outTok((size_t)nSlots * ch), outCnt(nSlots);
        uint32_t finMask = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (qk_step_chunk(e, outTok.data(), outCnt.data(), &finMask) > 0)
            for (uint32_t s = 0; s < nSlots; s++)
                for (uint32_t i = 0; i < outCnt[s]; i++) gen[s].push_back(outTok[s * ch + i]);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        bool allEq = true;
        for (uint32_t s = 1; s < nSlots; s++) if (gen[s] != gen[0]) allEq = false;
        printf("serve-test: %u slots x prompt %zu -> %zu tokens each in %.1f ms\n",
               nSlots, prompt.size(), gen[0].size(), ms);
        if (nSlots > 1) printf("all slots identical: %s\n", allEq ? "YES" : "NO");
        printf("GEN:");
        for (uint32_t t : gen[0]) printf(" %u", t);
        printf("\n");
        qk_close(e);
        return 0;
    }

    if (mode == "cachetest") {
        // Turn 1 populates the prefix cache; turn 2 (prompt = turn1 seq + more)
        // should hit the cache. Compare its output to a cold fresh-engine run of
        // the same turn-2 prompt: they must be token-identical.
        if (argc < 4) { fprintf(stderr, "usage: qk cachetest <ids-file> <nGen> [tmax]\n"); return 1; }
        std::vector<uint32_t> prompt;
        {
            FILE* f = fopen(argv[2], "r");
            if (!f) { perror(argv[2]); return 1; }
            int v;
            while (fscanf(f, "%d%*[, \n]", &v) == 1) prompt.push_back((uint32_t)v);
            fclose(f);
        }
        uint32_t nGen = (uint32_t)atoi(argv[3]);
        uint32_t tmax = argc > 4 ? (uint32_t)atoi(argv[4]) : 4096;
        auto run = [&](qk_engine* e, const std::vector<uint32_t>& ids, uint32_t g, double& ms) {
            std::vector<uint32_t> out;
            qk_slot_start(e, 0, ids.data(), (uint32_t)ids.size(), g);
            uint32_t ch = qk_chunk(e);
            std::vector<uint32_t> ot((size_t)ch), oc(1);
            uint32_t fin = 0;
            auto t0 = std::chrono::steady_clock::now();
            while (qk_step_chunk(e, ot.data(), oc.data(), &fin) > 0)
                for (uint32_t i = 0; i < oc[0]; i++) out.push_back(ot[i]);
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            return out;
        };
        char err[256] = {0};
        qk_config cfg{1, tmax, 8};
        qk_engine* eA = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!eA) { fprintf(stderr, "open: %s\n", err); return 1; }
        double t1;
        auto R = run(eA, prompt, nGen, t1);                       // turn 1 -> caches [prompt+R]
        std::vector<uint32_t> p2 = prompt;
        p2.insert(p2.end(), R.begin(), R.end());
        p2.insert(p2.end(), prompt.begin(), prompt.end());        // turn 2 = prompt + R + prompt
        double tw;
        auto warm = run(eA, p2, nGen, tw);                        // cache hit
        qk_close(eA);
        qk_engine* eB = qk_open(ggufPath(), &cfg, err, sizeof err);  // fresh engine, empty cache
        double tc;
        auto cold = run(eB, p2, nGen, tc);                        // full cold prefill
        qk_close(eB);
        printf("cachetest: turn-2 prompt %zu tokens, gen %u\n", p2.size(), nGen);
        printf("  cold (full prefill) %.1f ms  |  warm (cached prefix) %.1f ms  |  %.2fx faster\n",
               tc, tw, tw > 0 ? tc / tw : 0.0);
        bool eq = warm == cold;
        printf("  warm output identical to cold: %s\n", eq ? "YES" : "NO");
        if (!eq) {
            size_t d = 0;
            while (d < warm.size() && d < cold.size() && warm[d] == cold[d]) d++;
            printf("  first divergence at token %zu\n", d);
        }
        return 0;
    }

    VkCtx c;
    initVk(c, argv[0]);

    auto argU = [&](int i, uint32_t dflt) {
        return argc > i ? (uint32_t)atoi(argv[i]) : dflt;
    };

    bool ok = true;
    if (mode == "suite") {
        ok &= caseF16(c, 8192, 8192, 200);
        ok &= caseQ80(c, 8192, 8192, 200);
        ok &= caseQ6K(c, 8192, 8192, 200);
        ok &= caseIQ4XS(c, 8192, 8192, 200);
        ok &= caseIQ3XXS(c, 8192, 8192, 200);
        printf("\nreal-weight mode: qk gguf <tensor>   (see: qk list blk.0)\n");
    } else if (mode == "f16") {
        ok = caseF16(c, argU(2, 16384), argU(3, 8192), argU(4, 100));
    } else if (mode == "q8_0") {
        ok = caseQ80(c, argU(2, 16384), argU(3, 8192), argU(4, 100));
    } else if (mode == "q6_k") {
        ok = caseQ6K(c, argU(2, 16384), argU(3, 8192), argU(4, 100));
    } else if (mode == "iq4_xs") {
        ok = caseIQ4XS(c, argU(2, 16384), argU(3, 8192), argU(4, 100));
    } else if (mode == "iq3_xxs") {
        ok = caseIQ3XXS(c, argU(2, 16384), argU(3, 8192), argU(4, 100));
    } else if (mode == "bgemm") {
        ok = caseBGemm(c, argU(2, 8192), argU(3, 2048), argU(4, 256));  // batched-prefill GEMM
    } else if (mode == "gguf") {
        if (argc < 3) {
            fprintf(stderr, "usage: qk gguf <tensor> [iters]\n");
            return 1;
        }
        ok = caseGguf(c, argv[2], argU(3, 100));
    } else if (mode == "moe") {
        ok = caseMoe(c, argU(2, 0), argU(3, 200));
    } else if (mode == "block") {
        ok = caseBlock(c, argU(2, 0), argU(3, 3), argU(4, 200));
    } else if (mode == "ablock") {
        ok = caseABlock(c, argU(2, 3), argU(3, 3), argU(4, 200));
    } else if (mode == "token") {
        if (argc < 4) {
            fprintf(stderr, "usage: qk token <ids-file> <nGen> [tmax] [batch]\n");
            return 1;
        }
        ok = caseToken(c, argv[2], argU(3, 12), argU(4, 128), argU(5, 1));
    } else if (mode == "warm") {
        if (argc < 4) {
            fprintf(stderr, "usage: qk warm <ids-file> <nGen> [tmax]  (prefix-cache cold/warm demo)\n");
            return 1;
        }
        ok = caseToken(c, argv[2], argU(3, 12), argU(4, 128), 1, /*warmDemo=*/true);
    } else {
        fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
        return 1;
    }

    vkDeviceWaitIdle(c.dev);
    return ok ? 0 : 1;
}
#endif  // QK_LIBRARY
