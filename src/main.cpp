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
// Env: QK_DEVICE=<n> device index; QK_DEVICE_PCI=<bdf> (e.g. 1a:00.0, wins
//      over QK_DEVICE — enumeration order flips across boots, BDF doesn't);
//      QK_SHADER_DIR; QK_GGUF=<path>.

#include <vulkan/vulkan.h>

// pipeline-split harness (qk pipe / pipe-worker): plain TCP between stages
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <csignal>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
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

// BDF ("0000:1a:00.0") of a physical device via VK_EXT_pci_bus_info; ""
// when the driver doesn't expose it (lavapipe).
static std::string pciBdf(VkPhysicalDevice d) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(d, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> ext(n);
    vkEnumerateDeviceExtensionProperties(d, nullptr, &n, ext.data());
    bool has = false;
    for (const auto& e : ext)
        if (!strcmp(e.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) { has = true; break; }
    if (!has) return {};
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &pci;
    vkGetPhysicalDeviceProperties2(d, &p2);
    char buf[32];
    snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x",
             pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
    return buf;
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
    const char* wantPci = getenv("QK_DEVICE_PCI"); // BDF suffix, "1a:00.0" ok
    int pick = -1;
    std::string pickBdf;
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        VkPhysicalDeviceMemoryProperties m;
        vkGetPhysicalDeviceMemoryProperties(devs[i], &m);
        VkDeviceSize vram = 0;
        for (uint32_t h = 0; h < m.memoryHeapCount; h++)
            if (m.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                vram = std::max(vram, m.memoryHeaps[h].size);
        std::string bdf = pciBdf(devs[i]);
        bool discrete = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        fprintf(stderr, "vk device %u: %s [%s] vram=%zu MiB%s\n", i, p.deviceName,
                bdf.empty() ? "?" : bdf.c_str(), (size_t)(vram >> 20),
                discrete ? "" : " (not discrete)");
        if (wantPci && !bdf.empty() && bdf.size() >= strlen(wantPci) &&
            bdf.compare(bdf.size() - strlen(wantPci), strlen(wantPci), wantPci) == 0)
            pick = (int)i;
        if (!wantPci && pick < 0 && discrete) pick = (int)i;
    }
    if (wantPci && pick < 0) {
        fprintf(stderr, "QK_DEVICE_PCI=%s matched no device\n", wantPci);
        exit(1);
    }
    if (!wantPci)
        if (const char* e = getenv("QK_DEVICE")) pick = atoi(e);
    if (pick < 0) pick = 0;
    if (pick >= (int)ndev) {
        fprintf(stderr, "QK_DEVICE=%d out of range (%u devices)\n", pick, ndev);
        exit(1);
    }
    c.phys = devs[pick];
    vkGetPhysicalDeviceProperties(c.phys, &c.props);
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mp);
    pickBdf = pciBdf(c.phys);
    printf("device: %s [%s]\n", c.props.deviceName,
           pickBdf.empty() ? "?" : pickBdf.c_str());

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

// QK_STATS_FILE: mirror per-request stat lines ([pcache], [spec]) to an
// append-only file — pod logs reset on every restart, which kept wiping the
// tuning data these lines exist to accumulate.
static void qkStatsLine(const char* fmt, ...) {
    static FILE* f = [] {
        const char* p = getenv("QK_STATS_FILE");
        FILE* h = p ? fopen(p, "a") : nullptr;
        if (p && !h) fprintf(stderr, "QK_STATS_FILE: cannot open %s\n", p);
        return h;
    }();
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
    if (f) {
        va_start(ap, fmt);
        vfprintf(f, fmt, ap);
        va_end(ap);
        fflush(f);
    }
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
    const bool guIq4 = tGE->type == GGML_IQ4_XS && tUE->type == GGML_IQ4_XS;
    const bool guIq3 = tGE->type == GGML_IQ3_XXS && tUE->type == GGML_IQ3_XXS;
    if (tGI->type != GGML_F32 || tGIS->type != GGML_F32 ||
        (!guIq3 && !guIq4) ||
        (tDE->type != GGML_IQ4_XS && tDE->type != GGML_Q6_K) ||
        tGS->type != GGML_Q8_0 || tUS->type != GGML_Q8_0 || tDS->type != GGML_Q8_0) {
        fprintf(stderr, "layer %u tensor types don't match the compiled kernels\n", layer);
        return false;
    }
    const bool downQ6 = tDE->type == GGML_Q6_K;  // deep layers in the 35B GGUF

    const uint32_t nEmbd = (uint32_t)tGE->ne[0];
    const uint32_t nFf   = (uint32_t)tGE->ne[1];
    const uint32_t nExp  = (uint32_t)tGE->ne[2];
    const uint32_t nUsed =
        (uint32_t)g.kvInt(g.kvStr("general.architecture", "") + ".expert_used_count", 8);
    if (nExp > 512 || nUsed > 16) {
        fprintf(stderr, "n_expert %u / top-%u exceeds moe_select limits (512/16)\n", nExp, nUsed);
        return false;
    }
    printf("\n== moe blk.%u  n_embd=%u n_ff=%u experts=%u top-%u + shared  gate/up=%s ==\n",
           layer, nEmbd, nFf, nExp, nUsed, guIq4 ? "IQ4_XS" : "IQ3_XXS");

    auto x = randomX(nEmbd);
    const size_t rbGE = ggmlRowBytes(tGE->type, nEmbd);
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
    uint32_t ids[16];
    double wsel[16];
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
    auto dqGU = [&](const GgufTensor* t, uint32_t e, uint32_t r, float* out) {
        const uint8_t* row = t->data + ((size_t)e * nFf + r) * rbGE;
        if (guIq4) dequant_row_iq4_xs((const block_iq4_xs*)row, out, nEmbd);
        else       dequant_row_iq3_xxs((const block_iq3_xxs*)row, out, nEmbd);
    };
    for (uint32_t s = 0; s < nUsed; s++) {
        uint32_t e = ids[s];
        for (uint32_t r = 0; r < nFf; r++) {
            dqGU(tGE, e, r, tmpE.data());
            double ga = 0;
            for (uint32_t k = 0; k < nEmbd; k++) ga += (double)tmpE[k] * x[k];
            dqGU(tUE, e, r, tmpE.data());
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
    Pipe pGuIq3  = makePipe(c, guIq4 ? "moe_gateup_iq4.spv" : "moe_gateup_iq3.spv", 5, 16);
    Pipe pGuQ8   = makePipe(c, "moe_gateup_q8.spv", 4, 16);
    Pipe pDnIq4  = makePipe(c, downQ6 ? "moe_down_q6k.spv" : "moe_down_iq4.spv", 4, 16,
                            downQ6 ? 0 : 256);
    Pipe pDnQ8   = makePipe(c, "moe_down_q8.spv", 4, 16);

    const size_t szGI = (size_t)nExp * nEmbd * 4, szGIS = (size_t)nEmbd * 4;
    const size_t szGE = (size_t)nExp * nFf * rbGE, szDE = (size_t)nExp * nEmbd * rbDE;
    const size_t szGS = (size_t)nFf * rbGS, szDS = (size_t)nEmbd * rbDS;
    const size_t szX = (size_t)nEmbd * 4, szY = (size_t)nEmbd * 4;
    const size_t szH = (size_t)(nUsed + 1) * nFf * 4, szSel = 160, szL = (size_t)nExp * 4;

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
    struct SelOut { uint32_t ids[16]; float w[16]; float wShared; } selGpu;
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
    bool downQ6, guIq4;
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
    bool guIq3 = m.ge->type == GGML_IQ3_XXS && m.ue->type == GGML_IQ3_XXS;
    m.guIq4 = m.ge->type == GGML_IQ4_XS && m.ue->type == GGML_IQ4_XS;
    if (m.gi->type != GGML_F32 || m.gis->type != GGML_F32 ||
        (!guIq3 && !m.guIq4) ||
        (m.de->type != GGML_IQ4_XS && m.de->type != GGML_Q6_K) ||
        m.gs->type != GGML_Q8_0 || m.us->type != GGML_Q8_0 || m.ds->type != GGML_Q8_0) {
        fprintf(stderr, "layer %u: unexpected MoE tensor types\n", layer);
        return false;
    }
    m.nEmbd = (uint32_t)m.ge->ne[0];
    m.nFf = (uint32_t)m.ge->ne[1];
    m.nExp = (uint32_t)m.ge->ne[2];
    // top-k from the file (35B: 8, 80B: 10); kernels are sized for <= 16
    std::string arch = g.kvStr("general.architecture", "");
    m.nUsed = (uint32_t)g.kvInt(arch + ".expert_used_count", 8);
    if (m.nUsed < 1 || m.nUsed > 16 || m.nUsed > m.nExp) {
        fprintf(stderr, "layer %u: expert_used_count %u unsupported\n", layer, m.nUsed);
        return false;
    }
    m.downQ6 = m.de->type == GGML_Q6_K;
    m.rbGE = ggmlRowBytes(m.ge->type, m.nEmbd);
    m.rbDE = ggmlRowBytes(m.de->type, m.nFf);
    m.rbGS = ggmlRowBytes(GGML_Q8_0, m.nEmbd);
    m.rbDS = ggmlRowBytes(GGML_Q8_0, m.nFf);
    return true;
}

struct MoeRefSel {
    uint32_t ids[16];
    double w[16];
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
    auto dqGU = [&](const GgufTensor* t, uint32_t e, uint32_t r, float* out) {
        const uint8_t* row = t->data + ((size_t)e * m.nFf + r) * m.rbGE;
        if (m.guIq4) dequant_row_iq4_xs((const block_iq4_xs*)row, out, m.nEmbd);
        else         dequant_row_iq3_xxs((const block_iq3_xxs*)row, out, m.nEmbd);
    };
    for (uint32_t s = 0; s < m.nUsed; s++) {
        uint32_t e = sel.ids[s];
        for (uint32_t r = 0; r < m.nFf; r++) {
            dqGU(m.ge, e, r, tmpE.data());
            double ga = 0;
            for (uint32_t k = 0; k < m.nEmbd; k++) ga += (double)tmpE[k] * x[k];
            dqGU(m.ue, e, r, tmpE.data());
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
    Pipe pStep  = makePipe(c, "dn_step.spv", 4, 16);
    Pipe pGate  = makePipe(c, "dn_gate.spv", 4, 12);
    Pipe pGemvO = makePipe(c, "gemv_q8_0.spv", 3, 8, 128);   // K = d_inner
    Pipe pAddN  = makePipe(c, "add_rmsnorm.spv", 5, 8);
    Pipe pAdd   = makePipe(c, "vec_add.spv", 3, 4);
    Pipe pMoeL  = makePipe(c, "moe_logits.spv", 3, 16);
    Pipe pMoeS  = makePipe(c, "moe_select.spv", 4, 16);
    Pipe pMoeGU = makePipe(c, "moe_gateup_iq3.spv", 5, 16);
    Pipe pMoeGUs = makePipe(c, "moe_gateup_q8.spv", 4, 16);
    Pipe pMoeDn = makePipe(c, moe.downQ6 ? "moe_down_q6k.spv" : "moe_down_iq4.spv", 4, 16,
                           moe.downQ6 ? 0 : 256);
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
    struct { uint32_t d, hk, hv, kdiv; } pcStep{dS, hK, hV, 0};  // 35B harness: modulo tiling
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
        dispatchB(pStep, sStep, hV, &pcStep, 16);
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
    Pipe pMoeDn = makePipe(c, moe.downQ6 ? "moe_down_q6k.spv" : "moe_down_iq4.spv", 4, 16,
                           moe.downQ6 ? 0 : 256);
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
    Pipe pStep = makePipe(c, "dn_step.spv", 4, 16);
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
    Pipe pMoeDn4 = makePipe(c, "moe_down_iq4.spv", 4, 16, 256);
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
    struct { uint32_t d, hk, hv, kdiv; } pcStep{dS, hK, hV, 0};  // 35B harness: modulo tiling
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
                dispatchB(pStep, L.sStep, hV, &pcStep, 16);
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
    bool shareFork = false;  // QK_FORK: cache each fresh prompt's prefill so same-prompt requests fork it
    uint32_t vocab = 0, eosTok = 248046, bosTok = 248044;
    static constexpr uint32_t nEmbd = 2048, chQkv = 8192, dIn = 4096, hV = 32, dS = 128, hK = 16;
    static constexpr uint32_t dh = 256, hQ = 16, hKV = 2, nRot = 64;
    // Model-shape knobs that differ between Qwen3.6-35B (40/256/8) and
    // Qwen3-Next-80B (48/512/10); read from GGUF KVs / tensor shapes in open().
    // The tensor DIMS above are identical across both models by construction.
    uint32_t nLayer = 40, nExp = 256, nUsed = 8, ffE = 512;
    // DeltaNet GQA broadcast: 0 -> v-head h reads k-head h % hK (qwen35moe
    // ggml_repeat tiling); else h / dnKDiv (qwen3next consecutive pairs,
    // dnKDiv = hV/hK). Feeds the dn_step kernels' kDiv push constant.
    uint32_t dnKDiv = 0;
    bool guIq4 = false;   // routed gate/up experts are IQ4_XS (80B) vs IQ3_XXS (35B)
    bool embQ8 = false;   // token_embd is Q8_0 (80B repack) vs Q6_K (35B)
    float eps = 1e-6f;
    // Pipeline split (QK_LAYERS=a:b): this engine owns layers [lFirst, lEnd).
    // First stage also owns the embedding; last stage owns norm+head+argmax.
    // layers/blayers stay nLayer long (global indexing); unowned entries hold
    // no buffers and null descriptor sets — every walk skips them.
    uint32_t lFirst = 0, lEnd = nLayer;
    bool firstStage() const { return lFirst == 0; }
    bool lastStage() const { return lEnd == nLayer; }
    bool splitStage() const { return lFirst != 0 || lEnd != nLayer; }
    // Rows bbLogits held after the most recent head+argmax pass — stageTopK
    // reads the final row's logits from it (the sampling hook).
    uint32_t lastRunRows = 0;

    VkDescriptorPool dpool = VK_NULL_HANDLE;
    Pipe pRms, pGemvA, pAb, pConvN, pStep, pStepReg, pStepGate, pGate, pGemvO,
        pAddN, pAddNRoute,
        pPrep, pAttn, pMoeL, pMoeRS, pMoeRSHier, pMoeS, pMoeS256, pMoeGroupPairs,
        pMoeGU, pMoeGUGroup3,
        pMoeGUs, pMoeGUs64, pMoeDn4, pMoeDn4_128, pMoeDn6, pMoeDnsB,
        pMoeDnsB32, pMoeDnGroup4, pMoeDnGroup6, pAdd3, pAdd3Group, pHead,
        pAm1, pAm2, pEmb;
    Pipe pAttnS, pAttnSG2, pAttnSG4, pAttnSG8, pAttnR;
    // split-K decode attention (QK_NO_SPLITK=1 -> legacy pAttn); SG variants
    // reuse one KV read across 2/4/8 query heads.
    Pipe pPrepB, pAttnB, pAbB, pConvB, pStepB, pGateB, pGemmB;
    // IQ4_XS twins of the dense-projection / routed-gateup pipes (80B weights).
    // Identical bindings and push constants — only the in-shader dequant differs,
    // so descriptor sets stay layout-compatible with the Q8_0 pipes.
    Pipe pGemvA4, pGemvO4, pMoeGU4, pGemmB4;

    struct Layer {
        bool rec = false, downQ6 = false;
        // per-projection quant flags: true = IQ4_XS weight (dispatch the *4 pipe).
        // p1 = qkv|q, p2 = z|k, p3 = v, wo = ssm_out|attn_output.
        bool iq4P1 = false, iq4P2 = false, iq4P3 = false, iq4Wo = false;
        std::vector<Buf> bufs;
        VkDescriptorSet sRms, sP1, sP2, sP3, sAb, sConv, sStep, sStepReg, sStepGate,
            sGate, sWo, sAddN,
            sAddNRoute, sPrep, sAttn, sMoeL, sMoeRS, sMoeS, sMoeS256, sMoeGU,
            sMoeGUs, sMoeDn, sMoeDns, sAdd3;
        VkDescriptorSet sAttnS, sAttnR;  // split-K decode attention
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
        bTok, bSlotIn, bSlotPos, bAttnIndirect, bSamp, stage;
    Buf bAttScratch;           // split-K partials: nSlots*hQ*attnSplitMax*(dh+2) floats
    uint32_t attnChunk = 32;   // KV positions per split WG (QK_ATTN_CHUNK, 8..1024)
    uint32_t attnSplitMax = 0; // ceil(nCtx/attnChunk) WGs dispatched per q-head
    uint32_t attnGqaGroup = 1; // QK_ATTN_GQA_GROUP={2,4,8}
    uint32_t attnGqaThreshold = 2048; // QK_ATTN_GQA_THRESHOLD for AUTO 1->4 switch
    bool useSplitK = true;     // QK_NO_SPLITK=1 restores the legacy attention paths
    bool attnLiveDispatch = false; // QK_ATTN_LIVE_DISPATCH skips empty split-K WGs
    bool attnGqaAuto = false; // QK_ATTN_GQA_AUTO selects group1/4 by live context
    bool moeSelect256 = false; // QK_MOE_SELECT_FAST subgroup selector
    bool moeRouteFused = false; // QK_MOE_ROUTE_FUSED selects in the last router WG
    bool moeSelectHier = false; // QK_MOE_SELECT_HIER removes per-pick WG barriers
    bool moeSharedGu64 = false; // QK_MOE_SHARED_GU_64 uses only the 64 useful lanes
    bool moeDown128 = false; // QK_MOE_DOWN_128 shrinks underfilled routed IQ4 down WGs
    bool moeSharedDown32 = false; // QK_MOE_SHARED_DOWN_32 uses one useful wave
    bool moeGroupPrefill = false; // QK_MOE_GROUP_PREFILL expert-major routed MoE
    bool dnStepReg = false; // QK_DN_STEP_REG caches each state row across both passes
    bool dnStepGate = false; // QK_DN_STEP_GATE_FUSED removes the o[] round trip
    Buf bbPos;                 // 1-entry position buffer: the batch n==1 split-K
    uint32_t* bbPosMap = nullptr;  // case binds it where the srv shaders read slotPos
    VkDescriptorSet sHead, sAm1, sAm2, sEmb;

    uint32_t maxB = 0;
    Buf bbXin, bbXn, bbBig, bbMid, bbKin, bbVin, bbGb, bbConvOut, bbO, bbAtt, bbAttnOut,
        bbY, bbXn2, bbML, bbMH, bbMSel, bbMY, bbMY2, bbMOffsets, bbMPairs,
        bbMContrib, bbLogits, bbIds, bbCarry;
    struct BLayer {
        VkDescriptorSet sRms, sP1, sP2, sP3, sAb, sConv, sStep, sStepGate, sGate, sWo, sAddN,
            sAddNRoute, sPrep, sAttn, sMoeL, sMoeRS, sMoeS, sMoeS256, sMoeGU,
            sMoeGUs, sMoeDn, sMoeDns, sMoeGUGroup, sMoeDnGroup, sAdd3, sAdd3Group;
        VkDescriptorSet sAttnS, sAttnR;  // split-K for the n==1 (decode) batch case
    };
    std::vector<BLayer> blayers;
    VkDescriptorSet sbEmb, sbHead, sbMoeGroupPairs;
    uint32_t* bbIdsMap = nullptr;

    // Spec-decode verify (P0): per-position greedy argmax over bbLogits rows.
    // argmax1/argmax2 already z-batch (row rq at offset rq*vocab), so verify just
    // needs maxB-deep candidate buffers and a host-visible id readback.
    Buf bvAV, bvAI, bvTok, bvSamp;
    VkDescriptorSet svAm1, svAm2;
    uint32_t* bvSampMap = nullptr;

    // One pre-recorded step CB per dispatch depth: stepCBs[z-1] dispatches z
    // slots. Submitting the one matching the highest active slot avoids paying
    // the (bandwidth-bound) weight re-reads for idle slots on light load.
    std::vector<VkCommandBuffer> stepCBs;
    uint32_t *slotInMap = nullptr, *slotPosMap = nullptr, *attnIndirectMap = nullptr,
             *sampMap = nullptr;
    void* stageMap = nullptr;

    struct Slot {
        bool active = false;
        std::vector<uint32_t> prompt;      // full prompt of the current request
        std::vector<uint32_t> genTokens;   // tokens generated so far (for the cache key)
        uint32_t cursor = 0, pos = 0, gen = 0, maxGen = 0, last = 0;
        // Speculative decoding (QK_SPEC). A verify round can commit up to K
        // tokens at once; the ABI emits <= chunk per call, so the overflow waits
        // in outQ (a queue-draining call does no GPU work). finPending: the GPU
        // side finished (EOS / max_gen, state snapshotted) but queued tokens are
        // still being emitted; the slot frees when the queue drains.
        std::vector<uint32_t> outQ;
        size_t outQHead = 0;
        bool finPending = false;
        // prompt-lookup draft: hash of every specL-gram in (prompt ++ genTokens)
        // -> END index (exclusive) of its latest occurrence; built lazily
        std::unordered_map<uint64_t, uint32_t> ngram;
        uint32_t ngramBuilt = 0;
        uint32_t specRounds = 0, specFed = 0, specEmitted = 0, serialSteps = 0;
        void resetSpec() {
            outQ.clear(); outQHead = 0; finPending = false;
            ngram.clear(); ngramBuilt = 0;
            specRounds = specFed = specEmitted = serialSteps = 0;
        }
    };
    std::vector<Slot> slots;
    bool specOn = false;          // QK_SPEC=1: speculative decoding (single-active-slot v1)
    uint32_t specL = 6, specK = 8;  // QK_SPEC_L trigger n-gram length, QK_SPEC_K verify width
    std::vector<uint32_t> specToks, specAm;  // verify batch + per-position argmax scratch
    // Prompt-lookup draft for slot s: if the specL-gram suffix of its history
    // recurs earlier, fill specToks with [pending token, drafted continuation..]
    // and return the batch length n >= 2; 0 = no trigger (stay serial).
    uint32_t specDraft(uint32_t s);

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

    // QK_STEP_PROF: one-shot per-stage GPU timestamps baked into the z=1 step
    // CB (barriers serialize the chain, so consecutive deltas = stage cost).
    VkQueryPool profQ = VK_NULL_HANDLE;
    std::vector<const char*> profLbl;
    bool profPrinted = false;
    bool profArmed = true;  // QK_STEP_PROF_DEFER lets a harness arm after warmup/idle

    bool open(const char* path, const qk_config& cfg, char* err, size_t errLen);
    int stepChunk(uint32_t* outTok, uint32_t* outCnt, uint32_t* outFin);
    // base=0: from-empty (resets slot, zero conv carry). base>0: CONTINUE an existing
    // slot at position `base` — keeps the restored state, seeds each layer's conv carry
    // from the slot's conv window, and writes K/V at pos=base+n. Enables cache-hit suffix
    // batching and >maxB multi-chunk prefill.
    // argmaxOut (optional): also run the z-batched argmax over ALL n logit rows and
    // write the n greedy ids (the prediction FOLLOWING toks[i]) — the spec-decode
    // verify primitive. Costs one head pass over n rows + two small reductions.
    // scratchState: run the gated-DeltaNet layers against the SCRATCH stripe
    // (index nSlots) instead of the slot's live stripe, leaving the live recurrent
    // state untouched at `base` — the spec-decode verify mode. The caller seeds
    // scratch (copyDnStripes) first; K/V writes still go to the live cache, which
    // is safe under rejection (causal reads never pass the committed position).
    // hiddenIn (split stages after the first): n*nEmbd residual rows from the
    // previous stage, injected in place of the embedding (toks may be null).
    // hiddenOut (split stages before the last): the residual rows after this
    // stage's final layer are read back for the next stage.
    void prefillBatchLast(const uint32_t* toks, uint32_t n, uint32_t slot,
                          std::vector<float>& logits, bool wantLogits = true, uint32_t base = 0,
                          uint32_t* argmaxOut = nullptr, bool scratchState = false,
                          const float* hiddenIn = nullptr, float* hiddenOut = nullptr);
    // Pipeline stage driver (see qk_stage_run in qk.h): n positions from `base`,
    // chunked at maxB; ids in / hidden out on the first stage, hidden in / ids
    // out on the last. base==0 resets the slot.
    int stageRun(uint32_t slot, const uint32_t* toks, const float* hiddenIn, uint32_t n,
                 uint32_t base, float* hiddenOut, uint32_t* idsOut);
    // Top-k (ids, logits) of the final position's row after a last-stage
    // stageRun — the split driver's sampling hook (see qk_stage_topk in qk.h).
    int stageTopK(uint32_t k, uint32_t* idsOut, float* valsOut);
    // Copy the 30 gated-DeltaNet layers' (conv window, delta-rule S) stripes
    // between stripe indices (a slot, or the scratch stripe = nSlots). ~63 MB,
    // one submit.
    void copyDnStripes(uint32_t fromStripe, uint32_t toStripe);
    // One spec-decode verify round: batched forward of n draft tokens from `base`
    // on the scratch stripe + per-position greedy ids into outIds. Live state
    // stays at `base`; K/V beyond the eventually-accepted prefix is garbage that
    // is never read. Caller then either promotes (full accept) or re-runs the
    // accepted prefix in live mode (commit pass).
    void verifyRound(const uint32_t* toks, uint32_t n, uint32_t slot, uint32_t base,
                     uint32_t* outIds);
    void promoteScratch(uint32_t slot) { copyDnStripes(nSlots, slot); }
    // Serial reference: drive the recorded per-token step CB over n prompt tokens on
    // `slot`, then read back that slot's raw logit row (the prediction after the last
    // prompt token). Returns the greedy argmax (robust to EOS, unlike the slot API).
    uint32_t serialPrefillLogits(const uint32_t* toks, uint32_t n, uint32_t slot,
                                 std::vector<float>& logits);
    void snapshotSlot(uint32_t slot);           // save slot state -> LRU cache entry
    int matchPrefix(const uint32_t* prompt, uint32_t n);  // longest cached prefix, or -1
    void restoreInto(uint32_t slot, int cacheIdx);        // cache entry -> slot state
    void copyStripes(uint32_t slot, VkBuffer snapBuf, bool save, uint32_t nTok = 0);  // stripes <-> snapshot (nTok bounds KV rows)
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
                   &bLogits, &bEmbdW, &bRope, &bAV, &bAI, &bTok, &bSlotIn, &bSlotPos,
                   &bAttnIndirect, &bSamp, &stage, &bAttScratch})
        destroyBuf(c, *b);
    for (Buf* b : {&bbXin, &bbXn, &bbBig, &bbMid, &bbKin, &bbVin, &bbGb, &bbConvOut, &bbO,
                   &bbAtt, &bbAttnOut, &bbY, &bbXn2, &bbML, &bbMH, &bbMSel, &bbMY, &bbMY2,
                   &bbMOffsets, &bbMPairs, &bbMContrib,
                   &bbLogits, &bbIds, &bbCarry, &bbPos, &bvAV, &bvAI, &bvTok, &bvSamp})
        destroyBuf(c, *b);
    if (profQ) vkDestroyQueryPool(c.dev, profQ, nullptr);
    if (dpool) vkDestroyDescriptorPool(c.dev, dpool, nullptr);
    for (Pipe* pp : {&pRms, &pGemvA, &pGemvA4,
                     &pAb, &pConvN, &pStep, &pStepReg, &pStepGate,
                     &pGate, &pGemvO,
                     &pGemvO4, &pAddN, &pAddNRoute, &pPrep,
                     &pAttn, &pAttnS, &pAttnSG2, &pAttnSG4, &pAttnSG8, &pAttnR,
                     &pMoeL, &pMoeRS, &pMoeRSHier, &pMoeS, &pMoeS256, &pMoeGroupPairs,
                     &pMoeGU, &pMoeGUGroup3,
                     &pMoeGUs, &pMoeGUs64, &pMoeDn4, &pMoeDn4_128, &pMoeDn6,
                     &pMoeDnsB, &pMoeDnsB32, &pMoeDnGroup4, &pMoeDnGroup6,
                     &pMoeGU4, &pAdd3, &pAdd3Group, &pHead, &pAm1, &pAm2, &pEmb, &pPrepB, &pAttnB,
                     &pAbB, &pConvB, &pStepB, &pGateB, &pGemmB, &pGemmB4})
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
        if (!L.st1) continue;  // layer not owned by this pipeline stage
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
void qk_engine::copyStripes(uint32_t slot, VkBuffer snapBuf, bool save, uint32_t nTok) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        if (!L.st1) continue;  // layer not owned by this pipeline stage
        // Attention KV stripes are [hKV][tmax][dh]: with a live token count,
        // copy only each kv-head's first nTok rows — snapshots then touch
        // (and keep resident) pages proportional to the conversation, not to
        // capacity. Offsets stay capacity-strided so save/restore layouts
        // match regardless of nTok; positions >= nTok are never read.
        // Recurrent (DeltaNet/conv) stripes are position-independent state
        // and always copy whole.
        if (!L.rec && nTok && nTok < nCtx) {
            VkDeviceSize headBytes = L.ps1 / hKV;
            VkDeviceSize liveBytes = (VkDeviceSize)nTok * dh * 4;
            VkBufferCopy cp1[8], cp2[8];
            for (uint32_t h = 0; h < hKV; h++) {
                VkDeviceSize off = (VkDeviceSize)h * headBytes;
                if (save) {
                    cp1[h] = {(VkDeviceSize)slot * L.ps1 + off, snapOff1[il] + off, liveBytes};
                    cp2[h] = {(VkDeviceSize)slot * L.ps2 + off, snapOff2[il] + off, liveBytes};
                } else {
                    cp1[h] = {snapOff1[il] + off, (VkDeviceSize)slot * L.ps1 + off, liveBytes};
                    cp2[h] = {snapOff2[il] + off, (VkDeviceSize)slot * L.ps2 + off, liveBytes};
                }
            }
            vkCmdCopyBuffer(c.cb, save ? L.st1 : snapBuf, save ? snapBuf : L.st1, hKV, cp1);
            vkCmdCopyBuffer(c.cb, save ? L.st2 : snapBuf, save ? snapBuf : L.st2, hKV, cp2);
            continue;
        }
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

void qk_engine::copyDnStripes(uint32_t fromStripe, uint32_t toStripe) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    for (auto& L : layers) {
        if (!L.rec || !L.st1) continue;
        VkBufferCopy a{(VkDeviceSize)fromStripe * L.ps1, (VkDeviceSize)toStripe * L.ps1, L.ps1};
        VkBufferCopy b{(VkDeviceSize)fromStripe * L.ps2, (VkDeviceSize)toStripe * L.ps2, L.ps2};
        vkCmdCopyBuffer(c.cb, L.st1, L.st1, 1, &a);
        vkCmdCopyBuffer(c.cb, L.st2, L.st2, 1, &b);
    }
    VK_CHECK(vkEndCommandBuffer(c.cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cb;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
}

void qk_engine::verifyRound(const uint32_t* toks, uint32_t n, uint32_t slot, uint32_t base,
                            uint32_t* outIds) {
    copyDnStripes(slot, nSlots);  // seed scratch = live state at `base`
    std::vector<float> dummy;
    prefillBatchLast(toks, n, slot, dummy, /*wantLogits=*/false, base, outIds,
                     /*scratchState=*/true);
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
    // Key by exactly the `pos` processed tokens: the first min(pos, |prompt|) prompt
    // tokens plus the fedGen generated tokens. (For an on-finish snapshot pos >= |prompt|
    // so this is the whole prompt + fed gens; for a prefill-boundary snapshot pos < |prompt|
    // so it is the prefilled prefix — which stays consistent with the captured state.)
    size_t pp = std::min((size_t)sl.pos, sl.prompt.size());
    e.tokens.assign(sl.prompt.begin(), sl.prompt.begin() + pp);
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
    shareFork = getenv("QK_FORK") != nullptr;
    // fa_attn_srv is flash-attention (tiled) now, so nCtx is bounded by KV-cache
    // VRAM, not shared memory: ~134 MB/attn-layer/slot at 32K, linear in nCtx.
    // 65536 fits the split-80B shape on both boxes (head 3 attn layers ~1.6 GB,
    // worker 9 layers ~4.8 GB at 2 slots); single-box 35B stays at 32768 by
    // deployment config (measured VRAM fit on the 20 GB card).
    if (nSlots < 1 || nSlots > 16 || nCtx < 64 || nCtx > 65536 || chunkN < 1 || chunkN > 32)
        return fail("qk_open: bad config");
    initVk(c, "libqk");  // shader dir resolved via QK_SHADER_DIR
    if (!g.open(path)) return fail("qk_open: cannot open GGUF");
    // Model-shape KVs (arch-prefixed): 35B = qwen35moe 40/8, 80B = qwen3next 48/10.
    const std::string arch = g.kvStr("general.architecture", "");
    nLayer = (uint32_t)g.kvInt(arch + ".block_count", nLayer);
    if (nLayer < 1 || nLayer > 256) return fail("qk_open: bad block_count");
    nUsed = (uint32_t)g.kvInt(arch + ".expert_used_count", nUsed);
    if (nUsed < 1 || nUsed > 16) return fail("qk_open: expert_used_count > 16 unsupported");
    // qwen3next pairs v-heads consecutively with k-heads (k-head g serves
    // v-heads 2g, 2g+1); qwen35moe tiles (h % hK). Verified against llama.cpp
    // per-op dumps — the two archs genuinely differ here.
    if (arch == "qwen3next") dnKDiv = hV / hK;
    eosTok = (uint32_t)g.kvInt("tokenizer.ggml.eos_token_id", eosTok);
    bosTok = (uint32_t)g.kvInt("tokenizer.ggml.bos_token_id", bosTok);
    lEnd = nLayer;  // full model by default (the member init used the pre-open default)
    // QK_LAYERS=a:b — pipeline-split stage owning layers [a,b). Driven via
    // qk_stage_run only (slot_start/step_chunk are disabled on a split engine).
    if (const char* v = getenv("QK_LAYERS")) {
        unsigned a = 0, b = 0;
        if (sscanf(v, "%u:%u", &a, &b) == 2 && a < b && b <= nLayer) { lFirst = a; lEnd = b; }
        else {
            snprintf(err, errLen, "qk_open: bad QK_LAYERS (want a:b with 0 <= a < b <= %u)", nLayer);
            return false;
        }
    }
    const GgufTensor* tEmbd = g.find("token_embd.weight");
    const GgufTensor* tONorm = g.find("output_norm.weight");
    const GgufTensor* tHead = g.find("output.weight");
    if (!tEmbd || !tONorm || !tHead ||
        (tEmbd->type != GGML_Q6_K && tEmbd->type != GGML_Q8_0) || tHead->type != GGML_Q6_K)
        return fail("qk_open: missing/unexpected embd/head tensors");
    embQ8 = tEmbd->type == GGML_Q8_0;
    vocab = (uint32_t)tHead->ne[1];
    // Routed-expert counts from the first owned layer's tensors (uniform per file).
    {
        char nb0[96];
        snprintf(nb0, sizeof nb0, "blk.%u.ffn_gate_inp.weight", lFirst);
        const GgufTensor* tgi0 = g.find(nb0);
        snprintf(nb0, sizeof nb0, "blk.%u.ffn_gate_exps.weight", lFirst);
        const GgufTensor* tge0 = g.find(nb0);
        if (!tgi0 || !tge0) return fail("qk_open: missing router tensors");
        nExp = (uint32_t)tgi0->ne[1];
        ffE = (uint32_t)tge0->ne[1];
        if (nExp < nUsed || nExp > 512 || ffE % 256 != 0)
            return fail("qk_open: expert shape unsupported (n_expert <= 512, n_ff_exp % 256)");
    }
    const size_t rbQ8e = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbQ8i = ggmlRowBytes(GGML_Q8_0, dIn);
    const size_t rbE = ggmlRowBytes(GGML_Q6_K, nEmbd);        // head rows (always Q6_K)
    const size_t rbEmb = ggmlRowBytes(tEmbd->type, nEmbd);    // embd rows (Q6_K or Q8_0)
    const uint32_t nB = nSlots, tmax = nCtx;

    pRms = makePipe(c, "rmsnorm.spv", 3, 8);
    pGemvA = makePipe(c, "gemv_q8_0.spv", 3, 8, 64);
    pGemvA4 = makePipe(c, "gemv_iq4_xs.spv", 3, 8, 64);
    pAb = makePipe(c, "dn_ab.spv", 6, 8);
    pConvN = makePipe(c, "dn_convn.spv", 4, 16);
    pStep = makePipe(c, "dn_step.spv", 4, 16);
    pStepReg = makePipe(c, "dn_step_reg.spv", 4, 16);
    pStepGate = makePipe(c, "dn_step_gate.spv", 6, 20);
    pGate = makePipe(c, "dn_gate.spv", 4, 12);
    pGemvO = makePipe(c, "gemv_q8_0.spv", 3, 8, 128);
    pGemvO4 = makePipe(c, "gemv_iq4_xs.spv", 3, 8, 128);
    pAddN = makePipe(c, "add_rmsnorm.spv", 5, 8);
    pAddNRoute = makePipe(c, "add_rmsnorm_route.spv", 6, 8);
    pPrep = makePipe(c, "fa_prep_srv.spv", 10, 32);
    pAttn = makePipe(c, "fa_attn_srv.spv", 6, 32);
    pAttnS = makePipe(c, "fa_attn_srv_split.spv", 5, 40);
    pAttnSG2 = makePipe(c, "fa_attn_srv_split_gqa.spv", 5, 40, 2);
    pAttnSG4 = makePipe(c, "fa_attn_srv_split_gqa.spv", 5, 40, 4);
    pAttnSG8 = makePipe(c, "fa_attn_srv_split_gqa.spv", 5, 40, 8);
    pAttnR = makePipe(c, "fa_attn_srv_reduce.spv", 4, 40);
    pMoeL = makePipe(c, "moe_logits.spv", 3, 16);
    pMoeRS = makePipe(c, "moe_route_select.spv", 5, 16);
    pMoeRSHier = makePipe(c, "moe_route_select_hier.spv", 5, 16,
                          nUsed <= 8 ? 2 : 4);
    pMoeS = makePipe(c, "moe_select.spv", 4, 16);
    pMoeS256 = makePipe(c, "moe_select_256.spv", 4, 16);
    pMoeGroupPairs = makePipe(c, "moe_group_pairs.spv", 3, 12);
    pMoeGU = makePipe(c, "moe_gateup_iq3.spv", 5, 16);
    pMoeGUGroup3 = makePipe(c, "moe_gateup_iq3_grouped.spv", 6, 16);
    pMoeGU4 = makePipe(c, "moe_gateup_iq4.spv", 5, 16);
    pMoeGUs = makePipe(c, "moe_gateup_q8.spv", 4, 16);
    pMoeGUs64 = makePipe(c, "moe_gateup_q8_64.spv", 4, 16);
    pMoeDn4 = makePipe(c, "moe_down_iq4.spv", 4, 16, 256);
    pMoeDn4_128 = makePipe(c, "moe_down_iq4.spv", 4, 16, 128);
    pMoeDn6 = makePipe(c, "moe_down_q6k.spv", 4, 16);
    pMoeDnsB = makePipe(c, "moe_down_q8b.spv", 4, 16);
    pMoeDnsB32 = makePipe(c, "moe_down_q8b_32.spv", 4, 16);
    pMoeDnGroup4 = makePipe(c, "moe_down_iq4_grouped.spv", 6, 16);
    pMoeDnGroup6 = makePipe(c, "moe_down_q6k_grouped.spv", 6, 16);
    pAdd3 = makePipe(c, "add_rms3.spv", 6, 8);
    pAdd3Group = makePipe(c, "add_rms3_grouped.spv", 6, 12);
    pHead = makePipe(c, "gemv_q6_k.spv", 3, 8, 128);
    pAm1 = makePipe(c, "argmax1.spv", 3, 8);
    pAm2 = makePipe(c, "argmax2.spv", 3, 4);
    pEmb = makePipe(c, embQ8 ? "embed_q8_0.spv" : "embed_q6k.spv", 3, 12);
    pPrepB = makePipe(c, "fa_prep_batch.spv", 9, 40);
    pAttnB = makePipe(c, "fa_attn_batch.spv", 5, 40);
    pAbB = makePipe(c, "dn_ab_batch.spv", 6, 12);
    pConvB = makePipe(c, "dn_conv_batch.spv", 5, 20);   // +conv-window state out (decode handoff)
    pStepB = makePipe(c, "dn_step_batch.spv", 4, 20);   // +delta-rule S seed/persist (decode handoff)
    pGateB = makePipe(c, "dn_gate_batch.spv", 4, 16);
    pGemmB = makePipe(c, "gemm_q8_0.spv", 3, 12);  // batched projections (weight reads amortized)
    pGemmB4 = makePipe(c, "gemm_iq4_xs.spv", 3, 12);
    static_assert(dS <= 128, "dn_step_batch srow[32] holds dState/4 vec4s; dState must be <= 128");

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
    if (const char* v = getenv("QK_ATTN_CHUNK")) {
        long x = atol(v);
        if (x >= 8 && x <= 1024) attnChunk = (uint32_t)x;
    }
    attnSplitMax = (tmax + attnChunk - 1) / attnChunk;
    bAttScratch = createBuf(c, (size_t)nB * hQ * attnSplitMax * (dh + 2) * 4, stor, true);
    bAttnOut = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bY = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bXn2 = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bML = createBuf(c, (size_t)nB * nExp * 4, stor, true);
    bMH = createBuf(c, (size_t)nB * (nUsed + 1) * ffE * 4, stor, true);
    bMSel = createBuf(c, (size_t)nB * 160, stor, true);   // sizeof(SelT): ids[16]+w[16]+wShared+pad[7]
    bMY = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    bMY2 = createBuf(c, (size_t)nB * nEmbd * 4, stor, true);
    // bONorm always exists (a non-last stage binds it as the throwaway next-norm
    // of its final add_rms3); the real weight is only uploaded on the last stage.
    // Embed/head weights and the logits/argmax buffers are stage-gated — each is
    // O(vocab) (~400 MB weights, ~1 MB/row logits), the split's whole point.
    bONorm = createBuf(c, nEmbd * 4, stor, true);
    if (lastStage()) {
        bHeadW = createBuf(c, (size_t)vocab * rbE, stor, true);
        bLogits = createBuf(c, (size_t)nB * vocab * 4, storSrc, true);
        bAV = createBuf(c, (size_t)nB * 64 * 4, stor, true);
        bAI = createBuf(c, (size_t)nB * 64 * 4, stor, true);
        bTok = createBuf(c, (size_t)nB * 4, storSrc, true);
        bSamp = createBuf(c, (size_t)nB * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);  // readback
    }
    if (firstStage()) bEmbdW = createBuf(c, (size_t)vocab * rbEmb, stor, true);
    bRope = createBuf(c, (size_t)tmax * (nRot / 2) * 2 * 4, stor, true);
    bSlotIn = createBuf(c, (size_t)nB * 4, stor, false);   // host-visible, host-written each step
    bSlotPos = createBuf(c, (size_t)nB * 4, stor, false);
    bAttnIndirect = createBuf(c, 6 * sizeof(uint32_t),
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false);
    stage = createBuf(c, 160u << 20,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    VK_CHECK(vkMapMemory(c.dev, bSlotIn.mem, 0, VK_WHOLE_SIZE, 0, (void**)&slotInMap));
    VK_CHECK(vkMapMemory(c.dev, bSlotPos.mem, 0, VK_WHOLE_SIZE, 0, (void**)&slotPosMap));
    VK_CHECK(vkMapMemory(c.dev, bAttnIndirect.mem, 0, VK_WHOLE_SIZE, 0,
                         (void**)&attnIndirectMap));
    if (lastStage()) VK_CHECK(vkMapMemory(c.dev, bSamp.mem, 0, VK_WHOLE_SIZE, 0, (void**)&sampMap));
    for (uint32_t s = 0; s < nB; s++) { slotInMap[s] = 0; slotPosMap[s] = 0; }
    attnIndirectMap[0] = attnSplitMax;
    attnIndirectMap[1] = hQ;
    attnIndirectMap[2] = 1;
    attnIndirectMap[3] = 0;
    attnIndirectMap[4] = hQ / 4;
    attnIndirectMap[5] = 1;

    VK_CHECK(vkMapMemory(c.dev, stage.mem, 0, VK_WHOLE_SIZE, 0, &stageMap));
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
            if (src) memcpy(stageMap, (const uint8_t*)src + off, chunk); else memset(stageMap, 0, chunk);
            begin();
            VkBufferCopy cp{0, off, chunk};
            vkCmdCopyBuffer(c.cb, stage.buf, dst.buf, 1, &cp);
            submitWait();
        }
    };
    upload(bONorm, lastStage() ? tONorm->data : nullptr, nEmbd * 4);
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
    if (lastStage()) upload(bHeadW, tHead->data, (size_t)vocab * rbE);
    if (firstStage()) upload(bEmbdW, tEmbd->data, (size_t)vocab * rbEmb);

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 2048; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
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

    // QK_MAXB: batch-prefill chunk width (tokens per prefillBatchLast submit).
    // Every bb* activation buffer below scales linearly with it — bbLogits
    // dominates at vocab*4 ≈ 1 MB per token of width — so the default stays 128;
    // raise via env for experiments.
    uint32_t cap = 128;
    if (const char* v = getenv("QK_MAXB")) {
        long x = atol(v);
        if (x >= 16 && x <= 1024) cap = (uint32_t)x;
        else fprintf(stderr, "QK_MAXB=%s out of range [16,1024]; using %u\n", v, cap);
    }
    maxB = cap;
    // storSrc: a non-last pipeline stage copies its final residual rows out of
    // bbXin for the next stage (prefillBatchLast hiddenOut).
    bbXin = createBuf(c, (size_t)cap * nEmbd * 4, storSrc, true);
    bbXn = createBuf(c, (size_t)cap * nEmbd * 4, stor, true);
    bbBig = createBuf(c, (size_t)cap * chQkv * 4, stor, true);
    bbMid = createBuf(c, (size_t)cap * dIn * 4, stor, true);
    bbKin = createBuf(c, (size_t)cap * hKV * dh * 4, stor, true);
    bbVin = createBuf(c, (size_t)cap * hKV * dh * 4, stor, true);
    bbGb = createBuf(c, (size_t)cap * 2 * hV * 4, stor, true);
    bbConvOut = createBuf(c, (size_t)cap * chQkv * 4, stor, true);
    bbO = createBuf(c, (size_t)cap * dIn * 4, stor, true);
    bbAtt = createBuf(c, (size_t)cap * dIn * 4, stor, true);
    bbAttnOut = createBuf(c, (size_t)cap * nEmbd * 4, stor, true);
    bbY = createBuf(c, (size_t)cap * nEmbd * 4, stor, true);
    bbXn2 = createBuf(c, (size_t)cap * nEmbd * 4, stor, true);
    bbML = createBuf(c, (size_t)cap * nExp * 4, stor, true);
    bbMH = createBuf(c, (size_t)cap * (nUsed + 1) * ffE * 4, stor, true);
    bbMSel = createBuf(c, (size_t)cap * 160, stor, true);
    bbMY = createBuf(c, (size_t)cap * nEmbd * 4, stor, true);
    bbMY2 = createBuf(c, (size_t)cap * nEmbd * 4, stor, true);
    bbMOffsets = createBuf(c, (size_t)(nExp + 1) * 4, stor, true);
    bbMPairs = createBuf(c, (size_t)cap * nUsed * 4, stor, true);
    bbMContrib = createBuf(c, (size_t)cap * nUsed * nEmbd * 4, stor, true);
    if (lastStage()) bbLogits = createBuf(c, (size_t)cap * vocab * 4, storSrc, true);
    bbIds = createBuf(c, (size_t)cap * 4, stor, false);
    bbPos = createBuf(c, 4, stor, false);
    VK_CHECK(vkMapMemory(c.dev, bbPos.mem, 0, VK_WHOLE_SIZE, 0, (void**)&bbPosMap));
    // Per-layer conv carry (the 3 tokens before a chunk): one [chQkv*3] slice per layer,
    // so a seeded (base>0) chunk can seed each deltanet layer's carry from its own conv
    // window. Zero for from-empty chunks. storSrc so it can be a copy target.
    bbCarry = createBuf(c, (size_t)nLayer * chQkv * 3 * 4, storSrc, true);
    upload(bbCarry, nullptr, (size_t)nLayer * chQkv * 3 * 4);
    VK_CHECK(vkMapMemory(c.dev, bbIds.mem, 0, VK_WHOLE_SIZE, 0, (void**)&bbIdsMap));
    if (lastStage()) {
        bvAV = createBuf(c, (size_t)cap * 64 * 4, stor, true);
        bvAI = createBuf(c, (size_t)cap * 64 * 4, stor, true);
        bvTok = createBuf(c, (size_t)cap * 4, storSrc, true);
        bvSamp = createBuf(c, (size_t)cap * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        VK_CHECK(vkMapMemory(c.dev, bvSamp.mem, 0, VK_WHOLE_SIZE, 0, (void**)&bvSampMap));
    }

    layers.resize(nLayer);
    blayers.resize(nLayer);
    char nb[128];
    for (uint32_t il = lFirst; il < lEnd; il++) {
        Layer& L = layers[il];
        BLayer& BL = blayers[il];
        auto T = [&](const char* suffix) -> const GgufTensor* {
            snprintf(nb, sizeof nb, "blk.%u.%s", il, suffix); return g.find(nb);
        };
        auto W = [&](const GgufTensor* t, size_t n) -> VkBuffer {
            L.bufs.push_back(createBuf(c, n, storSrc, true));  // storSrc: state stripes are copied from
            upload(L.bufs.back(), t ? t->data : nullptr, n);
            return L.bufs.back().buf;
        };
        auto Wp = [&](const void* p, size_t n) -> VkBuffer {  // host-built weights (ba de-interleave)
            L.bufs.push_back(createBuf(c, n, storSrc, true));
            upload(L.bufs.back(), p, n);
            return L.bufs.back().buf;
        };
        // Dense projection: Q8_0 (35B) or IQ4_XS (80B; attn_v stays Q8_0 there).
        // Missing tensor or any other type is a hard error — W(null) would
        // silently upload zeros.
        bool denseBad = false;
        auto Wd = [&](const char* suffix, uint32_t K, size_t rows, bool& iq4) -> VkBuffer {
            const GgufTensor* t = T(suffix);
            if (!t || (t->type != GGML_Q8_0 && t->type != GGML_IQ4_XS)) {
                fprintf(stderr, "blk.%u.%s: missing or unsupported dense type\n", il, suffix);
                denseBad = true;
                iq4 = false;
                return W(nullptr, rows * ggmlRowBytes(GGML_Q8_0, K));
            }
            iq4 = t->type == GGML_IQ4_XS;
            return W(t, rows * ggmlRowBytes(t->type, K));
        };
        MoeT moe;
        if (!loadMoeT(g, il, moe)) return fail("qk_open: MoE tensors missing");
        L.downQ6 = moe.downQ6;
        guIq4 = moe.guIq4;   // uniform across layers (checked per layer by loadMoeT)
        L.rec = T("ssm_a") != nullptr;
        VkBuffer aNorm = W(T("attn_norm.weight"), nEmbd * 4);
        VkBuffer pn = W(T("post_attention_norm.weight"), nEmbd * 4);
        L.aNormBuf = aNorm;
        if (L.rec) {
            VkBuffer qkvW = Wd("attn_qkv.weight", nEmbd, chQkv, L.iq4P1);
            VkBuffer zW = Wd("attn_gate.weight", nEmbd, dIn, L.iq4P2);
            VkBuffer alW, beW;
            if (T("ssm_alpha.weight")) {
                alW = W(T("ssm_alpha.weight"), (size_t)hV * nEmbd * 4);
                beW = W(T("ssm_beta.weight"), (size_t)hV * nEmbd * 4);
            } else {
                // qwen3next fuses the two: ssm_ba [nEmbd, 2*hV], interleaved per
                // k-head group g: rows g*4+{0,1} = beta (v-heads 2g, 2g+1),
                // rows g*4+{2,3} = alpha (llama.cpp qwen3next view split). The
                // tensor is tiny — dequant to F32 on the host and de-interleave
                // into the engine's split alpha/beta layout; dn_ab is unchanged.
                const GgufTensor* tBa = T("ssm_ba.weight");
                if (!tBa || tBa->ne[0] != nEmbd || tBa->ne[1] != 2 * hV)
                    return fail("qk_open: missing ssm_alpha/ssm_beta/ssm_ba");
                std::vector<float> row(nEmbd), al((size_t)hV * nEmbd), be((size_t)hV * nEmbd);
                size_t rbBa = ggmlRowBytes(tBa->type, nEmbd);
                for (uint32_t r = 0; r < 2 * hV; r++) {
                    const uint8_t* src = tBa->data + (size_t)r * rbBa;
                    if (tBa->type == GGML_F32) memcpy(row.data(), src, nEmbd * 4);
                    else if (tBa->type == GGML_IQ4_XS)
                        dequant_row_iq4_xs((const block_iq4_xs*)src, row.data(), nEmbd);
                    else if (tBa->type == GGML_Q8_0)
                        dequant_row_q8_0((const block_q8_0*)src, row.data(), nEmbd);
                    else return fail("qk_open: ssm_ba type unsupported");
                    uint32_t grp = r >> 2, sub = r & 3;
                    float* dst = (sub < 2 ? be.data() : al.data()) +
                                 ((size_t)grp * 2 + (sub & 1)) * nEmbd;
                    memcpy(dst, row.data(), (size_t)nEmbd * 4);
                }
                alW = Wp(al.data(), al.size() * 4);
                beW = Wp(be.data(), be.size() * 4);
            }
            VkBuffer dt = W(T("ssm_dt.bias"), hV * 4);
            VkBuffer av = W(T("ssm_a"), hV * 4);
            VkBuffer ker = W(T("ssm_conv1d.weight") ? T("ssm_conv1d.weight") : T("ssm_conv1d"),
                             (size_t)chQkv * 4 * 4);
            VkBuffer sn = W(T("ssm_norm.weight"), dS * 4);
            VkBuffer outW = Wd("ssm_out.weight", dIn, nEmbd, L.iq4Wo);
            // +1 stripe: the spec-decode verify scratch (stripe index nSlots).
            // Verify rounds are engine-thread-serial, so one shared scratch
            // stripe serves every slot (~63 MB total across the 30 rec layers).
            VkBuffer convSt = W(nullptr, (size_t)(nB + 1) * chQkv * 3 * 4);
            VkBuffer S = W(nullptr, (size_t)(nB + 1) * hV * dS * dS * 4);
            L.st1 = convSt; L.ps1 = (size_t)chQkv * 3 * 4;
            L.st2 = S;      L.ps2 = (size_t)hV * dS * dS * 4;
            L.sRms = mkSet(pRms, {bXin.buf, aNorm, bXn.buf});
            L.sP1 = mkSet(pGemvA, {qkvW, bXn.buf, bBig.buf});
            L.sP2 = mkSet(pGemvA, {zW, bXn.buf, bMid.buf});
            L.sAb = mkSet(pAb, {bXn.buf, alW, beW, dt, av, bGb.buf});
            L.sConv = mkSet(pConvN, {convSt, bBig.buf, ker, bConvOut.buf});
            L.sStep = mkSet(pStep, {bConvOut.buf, bGb.buf, S, bO.buf});
            L.sStepReg = mkSet(pStepReg, {bConvOut.buf, bGb.buf, S, bO.buf});
            L.sStepGate = mkSet(pStepGate, {bConvOut.buf, bGb.buf, S, sn, bMid.buf, bAtt.buf});
            L.sGate = mkSet(pGate, {bO.buf, sn, bMid.buf, bAtt.buf});
            L.sWo = mkSet(pGemvO, {outW, bAtt.buf, bAttnOut.buf});
            BL.sRms = mkSet(pRms, {bbXin.buf, aNorm, bbXn.buf});
            BL.sP1 = mkSet(pGemvA, {qkvW, bbXn.buf, bbBig.buf});
            BL.sP2 = mkSet(pGemvA, {zW, bbXn.buf, bbMid.buf});
            BL.sAb = mkSet(pAbB, {bbXn.buf, alW, beW, dt, av, bbGb.buf});
            BL.sConv = mkSet(pConvB, {bbCarry.buf, bbBig.buf, ker, bbConvOut.buf, convSt});
            {   // point this layer's conv carry at its own [chQkv*3] slice of bbCarry
                VkDescriptorBufferInfo dbi{bbCarry.buf, (VkDeviceSize)il * chQkv * 3 * 4,
                                           (VkDeviceSize)chQkv * 3 * 4};
                VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                wr.dstSet = BL.sConv; wr.dstBinding = 0; wr.descriptorCount = 1;
                wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr.pBufferInfo = &dbi;
                vkUpdateDescriptorSets(c.dev, 1, &wr, 0, nullptr);
            }
            BL.sStep = mkSet(pStepB, {bbConvOut.buf, bbGb.buf, bbO.buf, S});
            BL.sStepGate = mkSet(pStepGate,
                                 {bbConvOut.buf, bbGb.buf, S, sn, bbMid.buf, bbAtt.buf});
            BL.sGate = mkSet(pGateB, {bbO.buf, sn, bbMid.buf, bbAtt.buf});
            BL.sWo = mkSet(pGemvO, {outW, bbAtt.buf, bbAttnOut.buf});
        } else {
            VkBuffer wq = Wd("attn_q.weight", nEmbd, chQkv, L.iq4P1);
            VkBuffer wk = Wd("attn_k.weight", nEmbd, (size_t)hKV * dh, L.iq4P2);
            VkBuffer wv = Wd("attn_v.weight", nEmbd, (size_t)hKV * dh, L.iq4P3);
            VkBuffer qn = W(T("attn_q_norm.weight"), dh * 4);
            VkBuffer kn = W(T("attn_k_norm.weight"), dh * 4);
            VkBuffer wo = Wd("attn_output.weight", dIn, nEmbd, L.iq4Wo);
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
        L.sAttnS = mkSet(pAttnS, {bMid.buf, kc, vc, bAttScratch.buf, bSlotPos.buf});
        L.sAttnR = mkSet(pAttnR, {bAttScratch.buf, bBig.buf, bAtt.buf, bSlotPos.buf});
            L.sWo = mkSet(pGemvO, {wo, bAtt.buf, bAttnOut.buf});
            BL.sRms = mkSet(pRms, {bbXin.buf, aNorm, bbXn.buf});
            BL.sP1 = mkSet(pGemvA, {wq, bbXn.buf, bbBig.buf});
            BL.sP2 = mkSet(pGemvA, {wk, bbXn.buf, bbKin.buf});
            BL.sP3 = mkSet(pGemvA, {wv, bbXn.buf, bbVin.buf});
            BL.sPrep = mkSet(pPrepB, {bbBig.buf, bbKin.buf, bbVin.buf, qn, kn, bbMid.buf, kc, vc,
                                      bRope.buf});
            BL.sAttn = mkSet(pAttnB, {bbMid.buf, kc, vc, bbBig.buf, bbAtt.buf});
            BL.sAttnS = mkSet(pAttnS, {bbMid.buf, kc, vc, bAttScratch.buf, bbPos.buf});
            BL.sAttnR = mkSet(pAttnR, {bAttScratch.buf, bbBig.buf, bbAtt.buf, bbPos.buf});
            BL.sWo = mkSet(pGemvO, {wo, bbAtt.buf, bbAttnOut.buf});
        }
        if (denseBad) return fail("qk_open: dense projection tensor missing or bad type");
        VkBuffer mgi = W(moe.gi, (size_t)moe.nExp * nEmbd * 4);
        VkBuffer mgis = W(moe.gis, nEmbd * 4);
        VkBuffer mge = W(moe.ge, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        VkBuffer mue = W(moe.ue, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        VkBuffer mde = W(moe.de, (size_t)moe.nExp * moe.nEmbd * moe.rbDE);
        VkBuffer mgs = W(moe.gs, (size_t)moe.nFf * moe.rbGS);
        VkBuffer mus = W(moe.us, (size_t)moe.nFf * moe.rbGS);
        VkBuffer mds = W(moe.ds, (size_t)moe.nEmbd * moe.rbDS);
        L.sAddN = mkSet(pAddN, {bXin.buf, bAttnOut.buf, pn, bY.buf, bXn2.buf});
        L.sAddNRoute = mkSet(pAddNRoute,
                             {bXin.buf, bAttnOut.buf, pn, bY.buf, bXn2.buf, bMSel.buf});
        L.sMoeL = mkSet(pMoeL, {mgi, bXn2.buf, bML.buf});
        L.sMoeRS = mkSet(pMoeRS, {mgi, bXn2.buf, bML.buf, mgis, bMSel.buf});
        L.sMoeS = mkSet(pMoeS, {bML.buf, mgis, bXn2.buf, bMSel.buf});
        L.sMoeS256 = mkSet(pMoeS256, {bML.buf, mgis, bXn2.buf, bMSel.buf});
        L.sMoeGU = mkSet(guIq4 ? pMoeGU4 : pMoeGU, {mge, mue, bXn2.buf, bMSel.buf, bMH.buf});
        L.sMoeGUs = mkSet(pMoeGUs, {mgs, mus, bXn2.buf, bMH.buf});
        L.sMoeDn = mkSet(L.downQ6 ? pMoeDn6 : pMoeDn4, {mde, bMH.buf, bMSel.buf, bMY.buf});
        L.sMoeDns = mkSet(pMoeDnsB, {mds, bMH.buf, bMSel.buf, bMY2.buf});
        BL.sAddN = mkSet(pAddN, {bbXin.buf, bbAttnOut.buf, pn, bbY.buf, bbXn2.buf});
        BL.sAddNRoute = mkSet(pAddNRoute,
                              {bbXin.buf, bbAttnOut.buf, pn, bbY.buf, bbXn2.buf, bbMSel.buf});
        BL.sMoeL = mkSet(pMoeL, {mgi, bbXn2.buf, bbML.buf});
        BL.sMoeRS = mkSet(pMoeRS, {mgi, bbXn2.buf, bbML.buf, mgis, bbMSel.buf});
        BL.sMoeS = mkSet(pMoeS, {bbML.buf, mgis, bbXn2.buf, bbMSel.buf});
        BL.sMoeS256 = mkSet(pMoeS256, {bbML.buf, mgis, bbXn2.buf, bbMSel.buf});
        BL.sMoeGU = mkSet(guIq4 ? pMoeGU4 : pMoeGU, {mge, mue, bbXn2.buf, bbMSel.buf, bbMH.buf});
        BL.sMoeGUs = mkSet(pMoeGUs, {mgs, mus, bbXn2.buf, bbMH.buf});
        BL.sMoeDn = mkSet(L.downQ6 ? pMoeDn6 : pMoeDn4, {mde, bbMH.buf, bbMSel.buf, bbMY.buf});
        BL.sMoeDns = mkSet(pMoeDnsB, {mds, bbMH.buf, bbMSel.buf, bbMY2.buf});
        BL.sMoeGUGroup = mkSet(pMoeGUGroup3,
                               {mge, mue, bbXn2.buf, bbMOffsets.buf, bbMPairs.buf,
                                bbMH.buf});
        BL.sMoeDnGroup = mkSet(L.downQ6 ? pMoeDnGroup6 : pMoeDnGroup4,
                               {mde, bbMH.buf, bbMSel.buf, bbMOffsets.buf,
                                bbMPairs.buf, bbMContrib.buf});
    }
    // A stage's LAST layer pre-norms with the next stage's first attn_norm in the
    // full model; here it binds bONorm instead — real weight on the last stage
    // (the final-norm handoff, unchanged), zeros elsewhere. That xn output is
    // dead on a non-last stage: the residual (bXin) crosses the stage boundary
    // and the next stage re-norms it with its own first layer's sRms.
    for (uint32_t il = lFirst; il < lEnd; il++)
        layers[il].sAdd3 = mkSet(pAdd3, {bY.buf, bMY.buf, bMY2.buf,
                                         il + 1 < lEnd ? layers[il + 1].aNormBuf : bONorm.buf,
                                         bXin.buf, bXn.buf});
    for (uint32_t il = lFirst; il < lEnd; il++)
        blayers[il].sAdd3 = mkSet(pAdd3, {bbY.buf, bbMY.buf, bbMY2.buf,
                                          il + 1 < lEnd ? layers[il + 1].aNormBuf : bONorm.buf,
                                          bbXin.buf, bbXn.buf});
    for (uint32_t il = lFirst; il < lEnd; il++)
        blayers[il].sAdd3Group = mkSet(pAdd3Group,
                                      {bbY.buf, bbMContrib.buf, bbMY2.buf,
                                       il + 1 < lEnd ? layers[il + 1].aNormBuf : bONorm.buf,
                                       bbXin.buf, bbXn.buf});
    if (lastStage()) {
        sHead = mkSet(pHead, {bHeadW.buf, bXn.buf, bLogits.buf});
        sAm1 = mkSet(pAm1, {bLogits.buf, bAV.buf, bAI.buf});
        sAm2 = mkSet(pAm2, {bAV.buf, bAI.buf, bTok.buf});
        sbHead = mkSet(pHead, {bHeadW.buf, bbXn.buf, bbLogits.buf});
        svAm1 = mkSet(pAm1, {bbLogits.buf, bvAV.buf, bvAI.buf});
        svAm2 = mkSet(pAm2, {bvAV.buf, bvAI.buf, bvTok.buf});
    }
    if (firstStage()) {
        sEmb = mkSet(pEmb, {bEmbdW.buf, bSlotIn.buf, bXin.buf});
        sbEmb = mkSet(pEmb, {bEmbdW.buf, bbIds.buf, bbXin.buf});
    }
    sbMoeGroupPairs = mkSet(pMoeGroupPairs,
                            {bbMSel.buf, bbMOffsets.buf, bbMPairs.buf});

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
        // Conversation snapshots kept (LRU), HOST-visible buffers. NOTE: raising
        // this does NOT help agentic multi-turn throughput — measured 0% hit at
        // both 3 and 16. Cross-turn reuse misses for a different reason: the
        // snapshot is keyed by the full rendered prompt, which ends with the
        // generation scaffold "<|im_start|>assistant\n<think></think>", so turn
        // N's snapshot is never a prefix of turn N+1 (the scaffold is replaced by
        // the real reply). The real fix is to snapshot at the history boundary
        // before that cue (see cross-turn-reuse task). Cache size only helps the
        // identical-prompt fork case. QK_PCACHE tunable kept for experiments.
        uint32_t PCACHE_N = 3;
        if (const char* pcn = getenv("QK_PCACHE")) {
            long v = strtol(pcn, nullptr, 10);
            if (v >= 1 && v <= 256) PCACHE_N = (uint32_t)v;
        }
        pcache.resize(PCACHE_N);
        for (auto& e : pcache)
            e.snap = createBuf(c, snapSize,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               false);  // host-visible
    }

    // ---- record one re-submittable step CB per dispatch depth (z = 1..nSlots) ----
    specOn = getenv("QK_SPEC") != nullptr;
    if (const char* v = getenv("QK_SPEC_L")) {
        long x = atol(v);
        if (x >= 2 && x <= 64) specL = (uint32_t)x;
    }
    if (const char* v = getenv("QK_SPEC_K")) {
        long x = atol(v);
        if (x >= 2 && (uint32_t)x <= maxB) specK = (uint32_t)x;
    }
    specToks.resize(maxB);
    specAm.resize(maxB);

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
    bool prof = getenv("QK_STEP_PROF") != nullptr && c.hasTimestamps && !splitStage();
    profArmed = getenv("QK_STEP_PROF_DEFER") == nullptr;
    if (prof) {
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = 1024;
        VK_CHECK(vkCreateQueryPool(c.dev, &qci, nullptr, &profQ));
    }
    auto stamp = [&](const char* lbl) {  // after a barrier: everything prior has drained
        if (!prof || zdim != 1 || profLbl.size() >= 1024) return;
        vkCmdWriteTimestamp(rcb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profQ, (uint32_t)profLbl.size());
        profLbl.push_back(lbl);
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
    struct { uint32_t d, hk, hv, kdiv; } pcStep{dS, hK, hV, dnKDiv};
    struct { uint32_t d, hk, hv, kdiv; float e; } pcStepGate{dS, hK, hV, dnKDiv, eps};
    struct { uint32_t d, hv; float e; } pcGate{dS, hV, eps};
    struct { uint32_t a, b, cc, d; } pcv{nEmbd, ffE, nExp, nUsed};
    struct { uint32_t pos, tmax, dh, nRot, hQ, hKV_; float eps, fb; uint32_t nSplit, pad; }
        pcFa{0, tmax, dh, nRot, hQ, hKV, eps, kFreqBase, attnSplitMax, attnChunk};
    useSplitK = getenv("QK_NO_SPLITK") == nullptr;  // split-K decode attention
    attnLiveDispatch = getenv("QK_ATTN_LIVE_DISPATCH") != nullptr;
    attnGqaAuto = getenv("QK_ATTN_GQA_AUTO") != nullptr;
    if (const char* v = getenv("QK_ATTN_GQA_GROUP")) {
        long x = atol(v);
        if (x == 2 || x == 4 || x == 8) attnGqaGroup = (uint32_t)x;
    }
    if (const char* v = getenv("QK_ATTN_GQA_THRESHOLD")) {
        long x = atol(v);
        if (x >= 256 && (uint32_t)x <= nCtx) attnGqaThreshold = (uint32_t)x;
    }
    moeSelect256 = getenv("QK_MOE_SELECT_FAST") != nullptr;
    moeRouteFused = getenv("QK_MOE_ROUTE_FUSED") != nullptr;
    moeSelectHier = getenv("QK_MOE_SELECT_HIER") != nullptr;
    moeSharedGu64 = getenv("QK_MOE_SHARED_GU_64") != nullptr;
    moeDown128 = getenv("QK_MOE_DOWN_128") != nullptr;
    moeSharedDown32 = getenv("QK_MOE_SHARED_DOWN_32") != nullptr;
    moeGroupPrefill = getenv("QK_MOE_GROUP_PREFILL") != nullptr;
    dnStepReg = getenv("QK_DN_STEP_REG") != nullptr;
    dnStepGate = getenv("QK_DN_STEP_GATE_FUSED") != nullptr;
    const uint32_t amWgs = (vocab + 4095) / 4096;
    struct { uint32_t n, span; } pcAm{vocab, 4096};
    struct { uint32_t m; } pcAm2{amWgs};
    struct { uint32_t k, idx, pr; } pcE{nEmbd, 0, 1};  // idx0 + perReq1 -> ids[rq] = slotInput[rq]

    // A split stage lacks embed/head, so the fused per-token step CBs cannot be
    // recorded; every forward goes through prefillBatchLast via stageRun instead
    // (the slot_start/step_chunk API is rejected on split engines).
    for (uint32_t z = 1; z <= (splitStage() ? 0 : nSlots); z++) {
    zdim = z;
    rcb = stepCBs[z - 1];
    VK_CHECK(vkBeginCommandBuffer(rcb, &cbbi));
    if (prof && z == 1) vkCmdResetQueryPool(rcb, profQ, 0, 1024);
    barrier();
    stamp("t0");
    disp(pEmb, sEmb, 1, &pcE, 12);
    barrier();
    stamp("emb");
    disp(pRms, layers[0].sRms, 1, &pcRms, 8);
    barrier();
    stamp("rms0");
    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        if (L.rec) {
            disp(L.iq4P1 ? pGemvA4 : pGemvA, L.sP1, (chQkv + 3) / 4, &pcQkv, 8);
            disp(L.iq4P2 ? pGemvA4 : pGemvA, L.sP2, (dIn + 3) / 4, &pcZ, 8);
            disp(pAb, L.sAb, 2 * hV, &pcAb, 8);
            barrier();
            stamp("dn.proj");
            disp(pConvN, L.sConv, chQkv / dS, &pcConvN, 16);
            barrier();
            stamp("dn.conv");
            if (dnStepGate) {
                disp(pStepGate, L.sStepGate, hV, &pcStepGate, 20);
                barrier();
                stamp("dn.s+g");
            } else {
                disp(dnStepReg ? pStepReg : pStep,
                     dnStepReg ? L.sStepReg : L.sStep, hV, &pcStep, 16);
                barrier();
                stamp("dn.step");
                disp(pGate, L.sGate, hV, &pcGate, 12);
                barrier();
                stamp("dn.gate");
            }
            disp(L.iq4Wo ? pGemvO4 : pGemvO, L.sWo, (nEmbd + 1) / 2, &pcWo, 8);
        } else {
            disp(L.iq4P1 ? pGemvA4 : pGemvA, L.sP1, (chQkv + 3) / 4, &pcQkv, 8);
            disp(L.iq4P2 ? pGemvA4 : pGemvA, L.sP2, (hKV * dh + 3) / 4, &pcKV, 8);
            disp(L.iq4P3 ? pGemvA4 : pGemvA, L.sP3, (hKV * dh + 3) / 4, &pcKV, 8);
            barrier();
            stamp("at.proj");
            disp(pPrep, L.sPrep, hQ + 2 * hKV, &pcFa, 32);
            barrier();
            stamp("at.prep");
            if (useSplitK) {
                auto bindSplit = [&](Pipe& ps) {
                    vkCmdBindPipeline(rcb, VK_PIPELINE_BIND_POINT_COMPUTE, ps.p);
                    vkCmdBindDescriptorSets(rcb, VK_PIPELINE_BIND_POINT_COMPUTE, ps.pl,
                                            0, 1, &L.sAttnS, 0, nullptr);
                    vkCmdPushConstants(rcb, ps.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 40, &pcFa);
                };
                Pipe& ps = attnGqaGroup == 8 ? pAttnSG8 :
                           attnGqaGroup == 4 ? pAttnSG4 :
                           attnGqaGroup == 2 ? pAttnSG2 : pAttnS;
                if (attnLiveDispatch && attnGqaAuto) {
                    bindSplit(pAttnS);
                    vkCmdDispatchIndirect(rcb, bAttnIndirect.buf, 0);
                    bindSplit(pAttnSG4);
                    vkCmdDispatchIndirect(rcb, bAttnIndirect.buf, 3 * sizeof(uint32_t));
                } else if (attnLiveDispatch) {
                    bindSplit(ps);
                    vkCmdDispatchIndirect(rcb, bAttnIndirect.buf, 0);
                } else {
                    bindSplit(ps);
                    vkCmdDispatch(rcb, attnSplitMax, hQ / attnGqaGroup, zdim);
                }
                barrier();
                stamp("at.split");
                disp(pAttnR, L.sAttnR, hQ, &pcFa, 40);
            } else {
                disp(pAttn, L.sAttn, hQ, &pcFa, 32);
            }
            barrier();
            stamp("at.attn");
            disp(L.iq4Wo ? pGemvO4 : pGemvO, L.sWo, (nEmbd + 1) / 2, &pcWo, 8);
        }
        barrier();
        stamp("wo");
        disp(moeRouteFused ? pAddNRoute : pAddN,
             moeRouteFused ? L.sAddNRoute : L.sAddN, 1, &pcRms, 8);
        barrier();
        stamp("addN");
        disp(moeRouteFused ? (moeSelectHier ? pMoeRSHier : pMoeRS) : pMoeL,
             moeRouteFused ? L.sMoeRS : L.sMoeL, nExp, &pcv, 16);
        disp(moeSharedGu64 ? pMoeGUs64 : pMoeGUs, L.sMoeGUs, ffE, &pcv, 16);
        barrier();
        stamp(moeRouteFused ? "moe.r+s" : "moe.route");
        if (!moeRouteFused) {
            disp(moeSelect256 ? pMoeS256 : pMoeS,
                 moeSelect256 ? L.sMoeS256 : L.sMoeS, 1, &pcv, 16);
            barrier();
            stamp("moe.sel");
        }
        disp(guIq4 ? pMoeGU4 : pMoeGU, L.sMoeGU, nUsed * ffE, &pcv, 16);
        barrier();
        stamp("moe.gu");
        disp(L.downQ6 ? pMoeDn6 : (moeDown128 ? pMoeDn4_128 : pMoeDn4),
             L.sMoeDn, nEmbd, &pcv, 16);
        disp(moeSharedDown32 ? pMoeDnsB32 : pMoeDnsB,
             L.sMoeDns, nEmbd, &pcv, 16);
        barrier();
        stamp("moe.dn");
        disp(pAdd3, L.sAdd3, 1, &pcRms, 8);
        barrier();
        stamp("add3");
    }
    disp(pHead, sHead, (vocab + 1) / 2, &pcHead, 8);
    barrier();
    stamp("head");
    disp(pAm1, sAm1, amWgs, &pcAm, 8);
    barrier();
    stamp("am1");
    disp(pAm2, sAm2, 1, &pcAm2, 4);
    VkMemoryBarrier m2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    m2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(rcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &m2, 0, nullptr, 0, nullptr);
    VkBufferCopy ct{0, 0, (size_t)nB * 4};
    vkCmdCopyBuffer(rcb, bTok.buf, bSamp.buf, 1, &ct);
    stamp("am2+copy");
    VK_CHECK(vkEndCommandBuffer(rcb));
    }  // for z = 1..nSlots

    slots.resize(nSlots);
    return true;
}

uint32_t qk_engine::specDraft(uint32_t s) {
    Slot& sl = slots[s];
    const uint32_t L = specL;
    size_t H = sl.prompt.size() + sl.genTokens.size();
    if (H < (size_t)L + 1) return 0;
    auto tok = [&](size_t i) {
        return i < sl.prompt.size() ? sl.prompt[i] : sl.genTokens[i - sl.prompt.size()];
    };
    auto gramHash = [&](size_t end) {  // FNV-1a over hist[end-L, end)
        uint64_t h = 1469598103934665603ull;
        for (size_t i = end - L; i < end; i++) { h ^= tok(i); h *= 1099511628211ull; }
        return h;
    };
    // Index grams ending strictly before H, so the query (the gram ending AT H)
    // never matches itself; it becomes indexable once the history grows past it.
    if (sl.ngramBuilt < L) sl.ngramBuilt = L;
    for (size_t end = sl.ngramBuilt; end < H; end++) sl.ngram[gramHash(end)] = (uint32_t)end;
    sl.ngramBuilt = (uint32_t)H;
    auto it = sl.ngram.find(gramHash(H));
    if (it == sl.ngram.end()) return 0;  // no trigger: stay serial (zero GPU cost)
    uint32_t e = it->second;             // suffix also ends at e; continuation = hist[e..]
    uint32_t maxN = std::min(std::min(specK, nCtx - sl.pos), sl.maxGen - sl.gen);
    if (maxN < 2) return 0;
    uint32_t nDraft = std::min<uint32_t>(maxN - 1, (uint32_t)(H - e));
    specToks[0] = tok(H - 1);            // == sl.last: sampled but not yet fed
    for (uint32_t j = 0; j < nDraft; j++) specToks[1 + j] = tok(e + j);
    return 1 + nDraft;
    // A hash collision only fakes a trigger; acceptance still compares real
    // argmaxes, so output stays exact — the round is just wasted work.
}

// Advance every active slot by up to chunkN steps. Returns active-at-entry.
// With QK_SPEC: drain queued spec tokens first (no GPU), then run gated verify
// rounds (single active slot), then serial steps for any remaining budget.
int qk_engine::stepChunk(uint32_t* outTok, uint32_t* outCnt, uint32_t* outFin) {
    for (uint32_t s = 0; s < nSlots; s++) outCnt[s] = 0;
    *outFin = 0;
    int activeAtEntry = 0;
    for (uint32_t s = 0; s < nSlots; s++) if (slots[s].active) activeAtEntry++;
    if (!activeAtEntry) return 0;

    auto finish = [&](uint32_t s) {  // GPU-side finished; free once the queue drains
        Slot& sl = slots[s];
        if (sl.finPending && sl.outQHead >= sl.outQ.size()) {
            if (specOn && sl.specRounds && getenv("QK_SPEC_LOG")) {
                qkStatsLine("[spec] slot=%u rounds=%u fed=%u emitted=%u serial=%u avg_accept=%.2f\n",
                            s, sl.specRounds, sl.specFed, sl.specEmitted, sl.serialSteps,
                            (double)sl.specFed / sl.specRounds);
            }
            sl.resetSpec();
            sl.active = false;
            *outFin |= 1u << s;
        }
    };
    auto emitTok = [&](uint32_t s, uint32_t t) {  // ABI cap: overflow waits in outQ
        if (outCnt[s] < chunkN) outTok[s * chunkN + outCnt[s]++] = t;
        else slots[s].outQ.push_back(t);
    };

    // ---- phase 1: drain queued spec tokens (no GPU work) ----
    for (uint32_t s = 0; s < nSlots; s++) {
        Slot& sl = slots[s];
        if (!sl.active) continue;
        while (sl.outQHead < sl.outQ.size() && outCnt[s] < chunkN)
            outTok[s * chunkN + outCnt[s]++] = sl.outQ[sl.outQHead++];
        if (sl.outQHead >= sl.outQ.size()) { sl.outQ.clear(); sl.outQHead = 0; }
        finish(s);
    }

    // ---- phase 2: speculative verify rounds ----
    // v1 fires only with a single active slot: a round blocks the engine thread
    // for c(K) ~ one serial chunk, but advances just one slot — fine alone, a
    // fairness question with concurrent slots (revisit with P3 replay numbers).
    if (specOn) {
        int only = -1, nAct = 0;
        for (uint32_t s = 0; s < nSlots; s++)
            if (slots[s].active && !slots[s].finPending) { nAct++; only = (int)s; }
        if (nAct == 1) {
            Slot& sl = slots[only];
            while (sl.active && !sl.finPending && sl.cursor >= sl.prompt.size() &&
                   outCnt[only] < chunkN) {
                uint32_t n = specDraft((uint32_t)only);
                if (n < 2) break;
                verifyRound(specToks.data(), n, (uint32_t)only, sl.pos, specAm.data());
                uint32_t k = 1;
                while (k < n && specToks[k] == specAm[k - 1]) k++;
                // an accepted greedy EOS ends the request: commit the tokens FED
                // up to it (its K/V is written, like a serial decode-step EOS) but
                // emit only the tokens before it
                uint32_t emitN = k;
                bool eosHit = false;
                for (uint32_t i = 0; i < k; i++)
                    if (specAm[i] == eosTok) { eosHit = true; emitN = i; k = i + 1; break; }
                if (k == n) promoteScratch((uint32_t)only);  // scratch state IS post-k
                else {
                    // rollback: live state is still at pos; commit the k accepted
                    // tokens by re-running them in live mode (exact — same input,
                    // same state as the verify round computed)
                    std::vector<float> dummy;
                    prefillBatchLast(specToks.data(), k, (uint32_t)only, dummy, false, sl.pos);
                }
                sl.pos += k;
                for (uint32_t i = 0; i < emitN; i++) {
                    sl.genTokens.push_back(specAm[i]);
                    sl.last = specAm[i];
                    sl.gen++;
                    emitTok((uint32_t)only, specAm[i]);
                }
                sl.specRounds++;
                sl.specFed += k;
                sl.specEmitted += emitN;
                if (eosHit || sl.gen >= sl.maxGen || sl.pos >= nCtx) {
                    snapshotSlot((uint32_t)only);
                    sl.finPending = true;
                    finish((uint32_t)only);  // may free the slot (and reset finPending)
                    break;
                }
            }
        }
    }

    // ---- phase 3: serial steps for whatever emission budget remains ----
    for (uint32_t step = 0; step < chunkN; step++) {
        int nAct = 0;
        uint32_t maxZ = 0;  // highest active slot index + 1 = needed dispatch depth
        bool need = false;  // someone still owes prefill progress or emitted tokens
        for (uint32_t s = 0; s < nSlots; s++) {
            Slot& sl = slots[s];
            if (!sl.active || sl.finPending) { slotInMap[s] = 0; slotPosMap[s] = 0; continue; }
            nAct++; maxZ = s + 1;
            if (sl.cursor < sl.prompt.size()) {         // prefill: feed prompt[cursor] at position cursor
                slotInMap[s] = sl.prompt[sl.cursor];
                slotPosMap[s] = sl.cursor;
                need = true;
            } else {                                    // decode: feed last sampled at pos
                slotInMap[s] = sl.last;
                slotPosMap[s] = sl.pos;
                if (outCnt[s] < chunkN) need = true;
            }
        }
        if (!nAct || !need) break;

        if (attnLiveDispatch) {
            uint32_t maxLive = 1;
            for (uint32_t s = 0; s < maxZ; s++)
                if (slots[s].active && !slots[s].finPending)
                    maxLive = std::max(maxLive, slotPosMap[s] + 1);
            uint32_t liveSplits = std::min(attnSplitMax,
                                           (maxLive + attnChunk - 1) / attnChunk);
            bool grouped = attnGqaAuto && maxLive >= attnGqaThreshold;
            attnIndirectMap[0] = grouped ? 0 : liveSplits;
            attnIndirectMap[1] = hQ / (attnGqaAuto ? 1 : attnGqaGroup);
            attnIndirectMap[2] = maxZ;
            attnIndirectMap[3] = grouped ? liveSplits : 0;
            attnIndirectMap[4] = hQ / 4;
            attnIndirectMap[5] = maxZ;
        }

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &stepCBs[maxZ - 1];  // dispatch only up to the top active slot
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));

        // QK_STEP_PROF: dump the per-stage timing of the first pure-decode step.
        if (profQ && profArmed && !profPrinted && maxZ == 1 &&
            slots[0].cursor >= slots[0].prompt.size()) {
            std::vector<uint64_t> ts(profLbl.size());
            if (!ts.empty() &&
                vkGetQueryPoolResults(c.dev, profQ, 0, (uint32_t)ts.size(), ts.size() * 8,
                                      ts.data(), 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                double per = c.props.limits.timestampPeriod / 1000.0;  // ticks -> us
                std::unordered_map<std::string, std::pair<uint32_t, double>> agg;
                for (size_t i = 1; i < ts.size(); i++) {
                    auto& a = agg[profLbl[i]];
                    a.first++;
                    a.second += (double)(ts[i] - ts[i - 1]) * per;
                }
                double tot = (double)(ts.back() - ts.front()) * per;
                std::vector<std::pair<double, std::string>> rows;
                for (auto& kv : agg) rows.push_back({kv.second.second, kv.first});
                std::sort(rows.rbegin(), rows.rend());
                fprintf(stderr, "[prof] decode step: %.0f us on-GPU across %zu stages\n",
                        tot, ts.size() - 1);
                for (auto& r : rows) {
                    auto& a = agg[r.second];
                    fprintf(stderr, "[prof]   %-10s n=%-3u tot=%8.1f us  avg=%6.2f us  %4.1f%%\n",
                            r.second.c_str(), a.first, a.second, a.second / a.first,
                            100.0 * a.second / tot);
                }
                if (getenv("QK_STEP_PROF_DETAIL")) {
                    uint32_t layer = 0;
                    std::unordered_map<std::string, uint32_t> occurrence;
                    for (size_t i = 1; i < ts.size(); i++) {
                        const char* lbl = profLbl[i];
                        uint32_t occ = occurrence[lbl]++;
                        double us = (double)(ts[i] - ts[i - 1]) * per;
                        fprintf(stderr,
                                "[prof-detail] seq=%-3zu layer=%-2u occ=%-2u %-10s %8.2f us\n",
                                i, layer, occ, lbl, us);
                        if (strcmp(lbl, "add3") == 0) layer++;
                    }
                }
                profPrinted = true;
            }
        }

        for (uint32_t s = 0; s < nSlots; s++) {
            Slot& sl = slots[s];
            if (!sl.active || sl.finPending) continue;
            uint32_t sampled = sampMap[s];
            bool prefilling = sl.cursor < sl.prompt.size();
            if (prefilling && sl.cursor + 1 < sl.prompt.size()) {
                sl.cursor++;  // still consuming prompt; ignore this logit
                continue;
            }
            // this step produced a real generated token (last prompt token, or a decode step)
            if (prefilling) { sl.cursor = (uint32_t)sl.prompt.size(); sl.pos = (uint32_t)sl.prompt.size(); }
            sl.serialSteps++;
            if (sampled == eosTok) {
                if (!prefilling) sl.pos++;  // a decode-step EOS still fed a token (its K/V is written)
                snapshotSlot(s);            // cache before the next step overwrites this slot
                sl.finPending = true;
                finish(s);                  // queue empty -> frees now (pre-spec behavior)
                continue;
            }
            emitTok(s, sampled);
            sl.genTokens.push_back(sampled);
            sl.last = sampled;
            sl.gen++;
            if (!prefilling) sl.pos++;
            if (sl.pos >= nCtx || sl.gen >= sl.maxGen) {
                snapshotSlot(s);
                sl.finPending = true;
                finish(s);
            }
        }
    }
    return activeAtEntry;
}

// Batch-prefill tokens [base, base+n) of `slot`. base=0 starts from empty (resets the
// slot, zero conv carry); base>0 CONTINUES a slot already holding state at position
// `base` (from a restore or a previous chunk) — the slot's delta-rule S/conv window/K/V
// are kept, each layer's conv carry is seeded from its slot conv window, and K/V is
// written at pos=base+n. Leaves the slot's state exactly as n serial steps from `base`
// would, so serial decode (or the next chunk) continues from base+n. Projections use the
// tiled GEMM for n>=48 (accumulation order differs from serial GEMV by ~1e-7, argmax-
// stable; review R1).
void qk_engine::prefillBatchLast(const uint32_t* toks, uint32_t n, uint32_t slot,
                                 std::vector<float>& logits, bool wantLogits, uint32_t base,
                                 uint32_t* argmaxOut, bool scratchState,
                                 const float* hiddenIn, float* hiddenOut) {
    if ((!toks && !hiddenIn) || n < 1 || n > maxB || slot >= nSlots || (size_t)base + n > nCtx ||
        (hiddenIn && firstStage()) || (!hiddenIn && !firstStage()) ||
        ((wantLogits || argmaxOut) && !lastStage())) {
        fprintf(stderr, "prefillBatchLast: bad args n=%u slot=%u base=%u stage=[%u,%u)\n",
                n, slot, base, lFirst, lEnd);
        exit(1);
    }
    if (!hiddenIn) memcpy(bbIdsMap, toks, (size_t)n * 4);
    if (wantLogits) logits.resize(vocab);
    // Hidden-row I/O rides the host-visible staging buffer at fixed offsets:
    // [0, vocab*4) stays the wantLogits readback; hidden-in and hidden-out get
    // disjoint maxB-row regions above it (no transfer/transfer hazard).
    const size_t hidInOff = 1u << 20, hidOutOff = (1u << 20) + (size_t)maxB * nEmbd * 4;
    if (hiddenIn) memcpy((uint8_t*)stageMap + hidInOff, hiddenIn, (size_t)n * nEmbd * 4);
    // base=0: zero the slot's recurrent state so dn_step's seed reads 0 and the conv
    // window starts clean (idempotent). base>0: keep the existing state to continue from.
    if (base == 0) resetSlot(slot);
    // Point the batched sets' per-slot STATE bindings at this slot's stripe (offset
    // slot*ps). The activation bindings are shared (one prefill at a time). Prefill is
    // synchronous, so rebinding the shared sets before re-recording the CB is safe.
    {
        VkDescriptorBufferInfo dbi; VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wr.descriptorCount = 1; wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr.pBufferInfo = &dbi;
        auto rebind = [&](VkDescriptorSet ds, uint32_t binding, VkBuffer buf, size_t ps, uint32_t stripe) {
            dbi = {buf, (VkDeviceSize)stripe * ps, (VkDeviceSize)ps};
            wr.dstSet = ds; wr.dstBinding = binding;
            vkUpdateDescriptorSets(c.dev, 1, &wr, 0, nullptr);
        };
        // Verify mode: the DeltaNet scan seeds from and persists to the scratch
        // stripe (pre-seeded by the caller); the live stripe keeps the state at
        // `base`. K/V stays on the live slot — rejected positions are never read.
        uint32_t dnStripe = scratchState ? nSlots : slot;
        for (uint32_t il = lFirst; il < lEnd; il++) {
            Layer& L = layers[il]; BLayer& BL = blayers[il];
            if (L.rec) {                                   // st1=convSt, st2=S
                rebind(BL.sConv, 4, L.st1, L.ps1, dnStripe);
                rebind(BL.sStep, 3, L.st2, L.ps2, dnStripe);
                rebind(BL.sStepGate, 2, L.st2, L.ps2, dnStripe);
            } else {                                       // st1=kc, st2=vc
                rebind(BL.sPrep, 6, L.st1, L.ps1, slot);
                rebind(BL.sPrep, 7, L.st2, L.ps2, slot);
                rebind(BL.sAttn, 1, L.st1, L.ps1, slot);
                rebind(BL.sAttn, 2, L.st2, L.ps2, slot);
                rebind(BL.sAttnS, 1, L.st1, L.ps1, slot);
                rebind(BL.sAttnS, 2, L.st2, L.ps2, slot);
            }
        }
    }

    VK_CHECK(vkResetCommandBuffer(c.cb, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));

    uint32_t zdim = n;
    // Atomic last-router selection is a decode optimization. At batch widths
    // above one, the standalone selector is already amortized and avoids one
    // device-scope atomic per expert/token row.
    bool fuseRoute = moeRouteFused && n == 1;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    auto barrier = [&]() {
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    // amdgpu kills any single submission that runs past its ~10 s gfx-ring timeout
    // (ring reset -> VK_ERROR_DEVICE_LOST). A whole-model chunk (nLayer layers x
    // maxB tokens) in ONE submit can cross that line when the GPU is contended or
    // thermally clamped, so flush the command buffer to the queue every
    // QK_SUBMIT_LAYERS layers (default 8). Submission order + waitIdle + the
    // fresh barrier keep execution and visibility identical to one big submit.
    static const uint32_t flushEvery = [] {
        const char* v = getenv("QK_SUBMIT_LAYERS");
        long x = v ? atol(v) : 8;
        return (uint32_t)(x < 1 ? 1 : x);
    }();
    // Per-dispatch attention-work ceiling (queries * keys). fa_attn_batch cost is
    // n*(base+n); at a long base a single full-chunk attention dispatch exceeds
    // amdgpu's ~10 s gfx-ring timeout -> VK_ERROR_DEVICE_LOST. Tile the (fully
    // independent) query axis so each attention dispatch stays under this budget.
    // The budget is a WALL-TIME proxy: the query-blocked kernel does ~16x more
    // q*k per second than its naive predecessor (whose measured limits set the
    // old 128k default), so the default scales by the same factor — 2M keeps
    // the original safety margin, and measured 10k-prefill is flat from 2M up.
    // Tunable via env.
    static const uint64_t attnBudget = [] {
        const char* v = getenv("QK_ATTN_BUDGET");
        long x = v ? atol(v) : 2097152;
        return (uint64_t)(x < 4096 ? 4096 : x);
    }();
    auto flushCB = [&]() {
        VK_CHECK(vkEndCommandBuffer(c.cb));
        VkSubmitInfo fsi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        fsi.commandBufferCount = 1;
        fsi.pCommandBuffers = &c.cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &fsi, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
        VK_CHECK(vkResetCommandBuffer(c.cb, 0));
        VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
        barrier();
    };
    auto disp = [&](Pipe& pp, VkDescriptorSet ds, uint32_t wgs, const void* pc, uint32_t pcSize) {
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(c.cb, pp.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
        uint32_t gx = std::min(wgs, c.props.limits.maxComputeWorkGroupCount[0]);
        uint32_t gy = (wgs + gx - 1) / gx;
        vkCmdDispatch(c.cb, gx, gy, zdim);
    };
    // Batched Q8_0 GEMM for the dense projections: Y[n][M] = X[n][K] . W[M][K]^T.
    // Reuses the gemv projection sets ({W,X,Y} storage buffers) — layout-compatible.
    // Reads each weight ONCE for all n tokens (vs gemv's n re-reads). K%32==0 (holds
    // for nEmbd=2048, dIn=4096). grid (ceil(M/128),1,ceil(n/64)) per gemm_q8_0.comp.
    auto gemmProj = [&](VkDescriptorSet ds, uint32_t M, uint32_t K, bool iq4) {
        struct { uint32_t M, K, N; } pcg{M, K, n};
        Pipe& pg = iq4 ? pGemmB4 : pGemmB;
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pg.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pg.pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(c.cb, pg.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pcg);
        vkCmdDispatch(c.cb, (M + 127) / 128, 1, (n + 63) / 64);
    };
    // Optimal projection per chunk width: the 128x64-tiled GEMM amortizes weight
    // reads and wins for wide chunks, but wastes work when n fills <1 N-tile; the
    // per-token GEMV (dispatched z=n) is faster below the measured ~48-token
    // crossover. isOut picks the wo pipe (tpr=128) vs the qkv/kv pipe (tpr=64).
    // iq4 routes to the IQ4_XS twin of the same-shape pipe (80B dense weights).
    struct { uint32_t m, k; } pcProj;
    auto proj = [&](VkDescriptorSet ds, uint32_t M, uint32_t K, bool isOut, bool iq4) {
        if (n >= 48) { gemmProj(ds, M, K, iq4); return; }
        pcProj = {M, K};
        if (isOut) disp(iq4 ? pGemvO4 : pGemvO, ds, (M + 1) / 2, &pcProj, 8);
        else       disp(iq4 ? pGemvA4 : pGemvA, ds, (M + 3) / 4, &pcProj, 8);
    };

    struct { uint32_t n; float e; } pcRms{nEmbd, eps};
    struct { uint32_t m, k; } pcQkv{chQkv, nEmbd}, pcZ{dIn, nEmbd}, pcKV{hKV * dh, nEmbd},
        pcWo{nEmbd, dIn}, pcHead{vocab, nEmbd};
    struct { uint32_t n, hv, Tn; } pcAbB{nEmbd, hV, n};
    struct { uint32_t channels, dState, qkCh; float e; uint32_t Tn; }
        pcConvB{chQkv, dS, 2 * hK * dS, eps, n};
    struct { uint32_t dState, hK_, hV_, Tn, kDiv; } pcStepB{dS, hK, hV, n, dnKDiv};
    struct { uint32_t d, hk, hv, kdiv; float e; } pcStepGate{dS, hK, hV, dnKDiv, eps};
    struct { uint32_t dState, hV_; float e; uint32_t Tn; } pcGateB{dS, hV, eps, n};
    struct { uint32_t a, b, cc, d; } pcv{nEmbd, ffE, nExp, nUsed};
    struct { uint32_t tmax, dh_, nRot_, hQ_, hKV_; float e, fb; uint32_t base, Tn, qbase; }
        pcFaB{nCtx, dh, nRot, hQ, hKV, eps, kFreqBase, base, n, 0};
    struct { uint32_t k, idx, pr; } pcE{nEmbd, 0, 1};

    // Seed each deltanet layer's conv carry: zero for a from-empty chunk (base=0), else
    // the slot's conv window (the 3 tokens ending at base-1) so the causal conv is
    // continuous across the boundary. carry and the slot conv window share [channels][3]
    // layout, so this is a plain copy.
    if (base == 0) {
        vkCmdFillBuffer(c.cb, bbCarry.buf, 0, (VkDeviceSize)nLayer * chQkv * 3 * 4, 0u);
    } else {
        for (uint32_t il = lFirst; il < lEnd; il++) {
            if (!layers[il].rec) continue;
            VkBufferCopy cc{(VkDeviceSize)slot * layers[il].ps1, (VkDeviceSize)il * chQkv * 3 * 4,
                            (VkDeviceSize)chQkv * 3 * 4};
            vkCmdCopyBuffer(c.cb, layers[il].st1, bbCarry.buf, 1, &cc);
        }
    }
    if (hiddenIn) {
        // Later pipeline stage: the previous stage's residual rows replace the
        // embedding as this stage's bbXin input.
        VkBufferCopy ch{hidInOff, 0, (VkDeviceSize)n * nEmbd * 4};
        vkCmdCopyBuffer(c.cb, stage.buf, bbXin.buf, 1, &ch);
        barrier();
    } else {
        barrier();
        disp(pEmb, sbEmb, 1, &pcE, 12);
        barrier();
        // QK_DUMP_X=<file>: read back the embedded residual rows (port-bisect
        // tooling; compare against a reference dequant of token_embd rows).
        if (const char* xf = getenv("QK_DUMP_X")) {
            VkMemoryBarrier tb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            tb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            tb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &tb, 0, nullptr, 0,
                                 nullptr);
            VkBufferCopy cx{0, (VkDeviceSize)hidOutOff, (VkDeviceSize)n * nEmbd * 4};
            vkCmdCopyBuffer(c.cb, bbXin.buf, stage.buf, 1, &cx);
            flushCB();
            if (FILE* f = fopen(xf, "wb")) {
                fwrite((uint8_t*)stageMap + hidOutOff, 4, (size_t)n * nEmbd, f);
                fclose(f);
            }
        }
    }
    disp(pRms, blayers[lFirst].sRms, 1, &pcRms, 8);
    barrier();
    // QK_DUMP_TAPS=<dir>: per-op readbacks for the FIRST layer of this stage
    // (port-bisect tooling; diff against llama.cpp eval-callback dumps).
    const char* tapsDir = getenv("QK_DUMP_TAPS");
    auto tap = [&](uint32_t il, const char* name, VkBuffer buf, size_t bytes) {
        if (!tapsDir || il != lFirst) return;
        VkMemoryBarrier tb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        tb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        tb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &tb, 0, nullptr, 0, nullptr);
        const size_t off = 64u << 20;
        VkBufferCopy cx{0, (VkDeviceSize)off, (VkDeviceSize)bytes};
        vkCmdCopyBuffer(c.cb, buf, stage.buf, 1, &cx);
        flushCB();
        char pth[512];
        snprintf(pth, sizeof pth, "%s/%s.bin", tapsDir, name);
        if (FILE* f = fopen(pth, "wb")) {
            fwrite((uint8_t*)stageMap + off, 1, bytes, f);
            fclose(f);
        }
    };
    for (uint32_t il = lFirst; il < lEnd; il++) {
        Layer& L = layers[il];
        BLayer& BL = blayers[il];
        zdim = n;
        if (L.rec) {
            tap(il, "xn", bbXn.buf, (size_t)n * nEmbd * 4);
            proj(BL.sP1, chQkv, nEmbd, false, L.iq4P1);
            proj(BL.sP2, dIn, nEmbd, false, L.iq4P2);
            disp(pAbB, BL.sAb, 2 * hV, &pcAbB, 12);
            barrier();
            tap(il, "qkv", bbBig.buf, (size_t)n * chQkv * 4);
            tap(il, "z", bbMid.buf, (size_t)n * dIn * 4);
            tap(il, "gb", bbGb.buf, (size_t)n * 2 * hV * 4);
            disp(pConvB, BL.sConv, chQkv / dS, &pcConvB, 20);
            barrier();
            tap(il, "conv", bbConvOut.buf, (size_t)n * chQkv * 4);
            if (n == 1 && dnStepGate && !tapsDir) {
                disp(pStepGate, BL.sStepGate, hV, &pcStepGate, 20);
                barrier();
            } else {
                zdim = 1;
                disp(pStepB, BL.sStep, hV, &pcStepB, 20);
                zdim = n;
                barrier();
                tap(il, "step", bbO.buf, (size_t)n * hV * dS * 4);
                disp(pGateB, BL.sGate, hV, &pcGateB, 16);
                barrier();
            }
            tap(il, "gate", bbAtt.buf, (size_t)n * dIn * 4);
            proj(BL.sWo, nEmbd, dIn, true, L.iq4Wo);
            barrier();
            tap(il, "wo", bbAttnOut.buf, (size_t)n * nEmbd * 4);
        } else {
            proj(BL.sP1, chQkv, nEmbd, false, L.iq4P1);
            proj(BL.sP2, hKV * dh, nEmbd, false, L.iq4P2);
            proj(BL.sP3, hKV * dh, nEmbd, false, L.iq4P3);
            barrier();
            disp(pPrepB, BL.sPrep, hQ + 2 * hKV, &pcFaB, 40);
            barrier();
            // Attention over the full [0, base+n) key range, but tiled along the
            // independent query axis so no single dispatch exceeds attnBudget.
            // Each tile flushes (submit + waitIdle) to bound its ring occupancy;
            // queries carry no cross-tile state so this is exact. The kernel is
            // query-BLOCKED (QB queries per workgroup share each K/V read), so
            // tile offsets stay QB-aligned and z counts blocks, not queries.
            if (useSplitK && n == 1) {
                // Decode-shaped call (the split head's per-token path): the
                // query-blocked kernel degenerates to hQ WGs serially walking
                // [0, base] — the same pathology split-K fixed on the serve
                // path. Reuse those exact shaders: rq (WG z) is 0, and bbPos
                // stands in for slotPos with the single query's position.
                *bbPosMap = base;
                struct { uint32_t pos, tmax, dh, nRot, hQ, hKV_; float eps, fb;
                         uint32_t nSplit, chunk; }
                pcSk{0, nCtx, dh, nRot, hQ, hKV, eps, kFreqBase,
                         attnSplitMax, attnChunk};
                zdim = 1;
                uint32_t group = attnGqaAuto && base + 1 >= attnGqaThreshold
                    ? 4 : attnGqaGroup;
                Pipe& ps = group == 8 ? pAttnSG8 : group == 4 ? pAttnSG4 :
                           group == 2 ? pAttnSG2 : pAttnS;
                vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, ps.p);
                vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, ps.pl,
                                        0, 1, &BL.sAttnS, 0, nullptr);
                vkCmdPushConstants(c.cb, ps.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 40, &pcSk);
                uint32_t liveSplits = attnLiveDispatch
                    ? std::min(attnSplitMax, (base + 1 + attnChunk - 1) / attnChunk)
                    : attnSplitMax;
                vkCmdDispatch(c.cb, liveSplits, hQ / group, 1);
                barrier();
                disp(pAttnR, BL.sAttnR, hQ, &pcSk, 40);
                zdim = n;
            } else {
                constexpr uint32_t kQB = 16;  // must match QB in fa_attn_batch.comp
                uint32_t qt = (uint32_t)std::max<uint64_t>(1, attnBudget / (uint64_t)(base + n));
                qt = std::min(qt, n);
                qt = std::max(kQB, qt - qt % kQB);
                for (uint32_t qo = 0; qo < n; qo += qt) {
                    uint32_t tile = std::min(qt, n - qo);
                    pcFaB.qbase = qo;
                    zdim = (tile + kQB - 1) / kQB;
                    disp(pAttnB, BL.sAttn, hQ, &pcFaB, 40);
                    if (qo + tile < n) { barrier(); flushCB(); }
                }
                pcFaB.qbase = 0;
                zdim = n;
            }
            barrier();
            proj(BL.sWo, nEmbd, dIn, true, L.iq4Wo);
        }
        barrier();
        disp(fuseRoute ? pAddNRoute : pAddN,
             fuseRoute ? BL.sAddNRoute : BL.sAddN, 1, &pcRms, 8);
        barrier();
        disp(fuseRoute ? (moeSelectHier ? pMoeRSHier : pMoeRS) : pMoeL,
             fuseRoute ? BL.sMoeRS : BL.sMoeL, nExp, &pcv, 16);
        disp(moeSharedGu64 ? pMoeGUs64 : pMoeGUs, BL.sMoeGUs, ffE, &pcv, 16);
        barrier();
        if (!fuseRoute) {
            bool fastSelect = moeSelect256 || moeRouteFused;
            disp(fastSelect ? pMoeS256 : pMoeS,
                 fastSelect ? BL.sMoeS256 : BL.sMoeS, 1, &pcv, 16);
            barrier();
        }
        // The grouped primitive is intentionally prefill-only and currently
        // targets the 35B IQ3 routed gate/up format.  Decode (n==1) and the
        // 80B IQ4 gate/up shape retain their measured card-best paths.
        bool groupMoe = moeGroupPrefill && n * nUsed >= nExp && !guIq4 && !tapsDir;
        if (groupMoe) {
            struct { uint32_t nExpert, nUsed, nTokens; } pcPairs{nExp, nUsed, n};
            zdim = 1;
            disp(pMoeGroupPairs, sbMoeGroupPairs, 1, &pcPairs, 12);
            barrier();

            vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pMoeGUGroup3.p);
            vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pMoeGUGroup3.pl, 0, 1, &BL.sMoeGUGroup, 0, nullptr);
            vkCmdPushConstants(c.cb, pMoeGUGroup3.pl, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, 16, &pcv);
            vkCmdDispatch(c.cb, ffE, nExp, 1);
        } else {
            zdim = n;
            disp(guIq4 ? pMoeGU4 : pMoeGU, BL.sMoeGU, nUsed * ffE, &pcv, 16);
        }
        barrier();
        if (groupMoe) {
            Pipe& pgd = L.downQ6 ? pMoeDnGroup6 : pMoeDnGroup4;
            vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pgd.p);
            vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pgd.pl, 0, 1, &BL.sMoeDnGroup, 0, nullptr);
            vkCmdPushConstants(c.cb, pgd.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pcv);
            vkCmdDispatch(c.cb, nEmbd, nExp, 1);
            zdim = n;
            disp(moeSharedDown32 ? pMoeDnsB32 : pMoeDnsB,
                 BL.sMoeDns, nEmbd, &pcv, 16);
            barrier();
            struct { uint32_t n; float e; uint32_t nUsed; }
                pcGroupTail{nEmbd, eps, nUsed};
            disp(pAdd3Group, BL.sAdd3Group, 1, &pcGroupTail, 12);
        } else {
            zdim = n;
            disp(L.downQ6 ? pMoeDn6 : (moeDown128 ? pMoeDn4_128 : pMoeDn4),
                 BL.sMoeDn, nEmbd, &pcv, 16);
            disp(moeSharedDown32 ? pMoeDnsB32 : pMoeDnsB,
                 BL.sMoeDns, nEmbd, &pcv, 16);
            barrier();
            disp(pAdd3, BL.sAdd3, 1, &pcRms, 8);
        }
        barrier();
        if ((il + 1 - lFirst) % flushEvery == 0 && il + 1 < lEnd) flushCB();
    }
    if (hiddenOut) {
        // Non-last pipeline stage: export the residual rows after this stage's
        // final layer (what the full model would feed the next layer's norm).
        VkMemoryBarrier mh{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mh.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mh.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mh, 0, nullptr, 0, nullptr);
        VkBufferCopy ch{0, hidOutOff, (VkDeviceSize)n * nEmbd * 4};
        vkCmdCopyBuffer(c.cb, bbXin.buf, stage.buf, 1, &ch);
    }
    // The head (last-token logits) is only needed by validation callers; when the
    // slot is being prefilled for decode, the serial step over the final prompt token
    // produces the first generated token, so skip the head entirely (a large save).
    if (wantLogits || argmaxOut) {
        zdim = n;
        lastRunRows = n;
        disp(pHead, sbHead, (vocab + 1) / 2, &pcHead, 8);
        if (argmaxOut) {
            barrier();
            const uint32_t amWgs = (vocab + 4095) / 4096;
            struct { uint32_t n, span; } pcAm{vocab, 4096};
            struct { uint32_t m; } pcAm2{amWgs};
            disp(pAm1, svAm1, amWgs, &pcAm, 8);
            barrier();
            disp(pAm2, svAm2, 1, &pcAm2, 4);
        }
        VkMemoryBarrier m2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        m2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &m2, 0, nullptr, 0, nullptr);
        if (wantLogits) {
            VkBufferCopy cp{(VkDeviceSize)(n - 1) * vocab * 4, 0, (VkDeviceSize)vocab * 4};
            vkCmdCopyBuffer(c.cb, bbLogits.buf, stage.buf, 1, &cp);
        }
        if (argmaxOut) {
            VkBufferCopy cv{0, 0, (VkDeviceSize)n * 4};
            vkCmdCopyBuffer(c.cb, bvTok.buf, bvSamp.buf, 1, &cv);
        }
    }

    VK_CHECK(vkEndCommandBuffer(c.cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cb;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
    if (wantLogits) memcpy(logits.data(), stageMap, (size_t)vocab * 4);
    if (argmaxOut) memcpy(argmaxOut, bvSampMap, (size_t)n * 4);
    if (hiddenOut) memcpy(hiddenOut, (uint8_t*)stageMap + hidOutOff, (size_t)n * nEmbd * 4);
}

int qk_engine::stageRun(uint32_t slot, const uint32_t* toks, const float* hiddenIn, uint32_t n,
                        uint32_t base, float* hiddenOut, uint32_t* idsOut) {
    if (slot >= nSlots || n < 1 || (size_t)base + n > nCtx) return -1;
    if (firstStage() ? (!toks || hiddenIn != nullptr) : !hiddenIn) return -2;
    if (lastStage() ? !idsOut : !hiddenOut) return -3;
    std::vector<float> dummy;
    for (uint32_t off = 0; off < n; off += maxB) {
        uint32_t cn = std::min(maxB, n - off);
        prefillBatchLast(toks ? toks + off : nullptr, cn, slot, dummy, /*wantLogits=*/false,
                         base + off, lastStage() ? idsOut + off : nullptr,
                         /*scratchState=*/false,
                         hiddenIn ? hiddenIn + (size_t)off * nEmbd : nullptr,
                         hiddenOut ? hiddenOut + (size_t)off * nEmbd : nullptr);
    }
    return 0;
}

int qk_engine::stageTopK(uint32_t k, uint32_t* idsOut, float* valsOut) {
    // Sampling hook: after a last-stage stageRun, hand back the top-k
    // (id, logit) of the FINAL position's row, descending, so the driver can
    // sample and feed its pick as the next position. Separate tiny submit —
    // the greedy path records nothing extra and stays bit-identical.
    if (!lastStage() || !idsOut || !valsOut || k < 1 || k > 256 || k > vocab) return -1;
    if (!lastRunRows) return -2;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    // The stage buffer below 1 MiB is the wantLogits landing zone (hidden
    // I/O regions start at hidInOff = 1 MiB); vocab*4 fits under it.
    VkBufferCopy cp{(VkDeviceSize)(lastRunRows - 1) * vocab * 4, 0, (VkDeviceSize)vocab * 4};
    vkCmdCopyBuffer(c.cb, bbLogits.buf, stage.buf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(c.cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cb;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
    const float* row = (const float*)stageMap;
    std::vector<uint32_t> idx(vocab);
    for (uint32_t i = 0; i < vocab; i++) idx[i] = i;
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [row](uint32_t a, uint32_t b) { return row[a] > row[b]; });
    std::sort(idx.begin(), idx.begin() + k,
              [row](uint32_t a, uint32_t b) { return row[a] > row[b]; });
    for (uint32_t i = 0; i < k; i++) {
        idsOut[i] = idx[i];
        valsOut[i] = row[idx[i]];
    }
    return 0;
}

uint32_t qk_engine::serialPrefillLogits(const uint32_t* toks, uint32_t n, uint32_t slot,
                                        std::vector<float>& logits) {
    logits.resize(vocab);
    // This single-slot debug/harness helper steps via stepCBs[slot], which
    // dispatches slots 0..slot *together*; the lower slots get stepped with the
    // zeroed input below (token 0 at pos 0), which writes their K/V + recurrent
    // state and corrupts any active session there. The recorded step CBs can't
    // dispatch one arbitrary slot, so refuse rather than silently trample a live
    // lower slot. (Callers use slot with all lower slots idle, so this never trips
    // in practice — it only forecloses the landmine.)
    for (uint32_t s = 0; s < slot; s++) {
        if (slots[s].active) {
            fprintf(stderr,
                    "serialPrefillLogits(slot=%u): lower slot %u is active; refusing "
                    "(would trample its state)\n",
                    slot, s);
            std::fill(logits.begin(), logits.end(), 0.0f);
            return 0;
        }
    }
    resetSlot(slot);
    for (uint32_t s = 0; s < nSlots; s++) { slotInMap[s] = 0; slotPosMap[s] = 0; }
    // Feed each prompt token at its own position through the recorded serial step CB
    // (stepCBs[slot] dispatches z=slot+1; only `slot` carries real input).
    for (uint32_t i = 0; i < n; i++) {
        slotInMap[slot] = toks[i];
        slotPosMap[slot] = i;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &stepCBs[slot];
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
    }
    // bLogits[slot] now holds the next-token logits after the last prompt token.
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cb, &bi));
    VkBufferCopy cp{(VkDeviceSize)slot * vocab * 4, 0, (VkDeviceSize)vocab * 4};
    vkCmdCopyBuffer(c.cb, bLogits.buf, stage.buf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(c.cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &c.cb;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
    memcpy(logits.data(), stageMap, (size_t)vocab * 4);
    uint32_t best = 0;
    for (uint32_t i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
    slots[slot].active = false;
    return best;
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
__attribute__((visibility("default"))) uint32_t qk_layer_first(const qk_engine* e) { return e->lFirst; }
__attribute__((visibility("default"))) uint32_t qk_layer_end(const qk_engine* e) { return e->lEnd; }
__attribute__((visibility("default"))) uint32_t qk_n_layer(const qk_engine* e) { return e->nLayer; }
__attribute__((visibility("default"))) uint32_t qk_n_embd(const qk_engine* e) { return qk_engine::nEmbd; }

__attribute__((visibility("default")))
int qk_stage_run(qk_engine* e, uint32_t slot, const uint32_t* toks, const float* hidden_in,
                 uint32_t n, uint32_t base, float* hidden_out, uint32_t* ids_out) {
    if (!e) return -1;
    return e->stageRun(slot, toks, hidden_in, n, base, hidden_out, ids_out);
}

__attribute__((visibility("default")))
int qk_stage_topk(qk_engine* e, uint32_t k, uint32_t* ids, float* vals) {
    if (!e) return -1;
    return e->stageTopK(k, ids, vals);
}

__attribute__((visibility("default")))
uint32_t qk_state_n(const qk_engine* e) { return (uint32_t)e->pcache.size(); }

__attribute__((visibility("default")))
int qk_state_save(qk_engine* e, uint32_t slot, uint32_t idx, uint32_t n_tok) {
    if (!e || slot >= e->nSlots || idx >= e->pcache.size()) return -1;
    e->copyStripes(slot, e->pcache[idx].snap.buf, /*save=*/true, n_tok);
    return 0;
}

__attribute__((visibility("default")))
int qk_state_load(qk_engine* e, uint32_t slot, uint32_t idx, uint32_t n_tok) {
    if (!e || slot >= e->nSlots || idx >= e->pcache.size()) return -1;
    e->copyStripes(slot, e->pcache[idx].snap.buf, /*save=*/false, n_tok);
    return 0;
}

__attribute__((visibility("default")))
int qk_slot_start(qk_engine* e, uint32_t slot, const uint32_t* prompt, uint32_t n_prompt,
                  uint32_t max_gen, uint32_t snap_prefix) {
    if (!e || slot >= e->nSlots) return -1;
    if (e->splitStage()) return -5;  // split stages are driven via qk_stage_run
    if (e->slots[slot].active) return -2;
    if (!prompt || n_prompt < 1 || n_prompt + max_gen > e->nCtx) return -3;
    for (uint32_t i = 0; i < n_prompt; i++) if (prompt[i] >= e->vocab) return -4;
    qk_engine::Slot& s = e->slots[slot];
    int cidx = e->matchPrefix(prompt, n_prompt);
    uint32_t start;                       // position already resident in the slot's state
    if (cidx >= 0) {
        e->restoreInto(slot, cidx);       // reuse a cached prefix (restore its state)
        start = (uint32_t)e->pcache[cidx].tokens.size();
    } else {
        e->resetSlot(slot);               // fresh: clear any prior occupant's recurrent state
        start = 0;
    }
    // Batch-prefill [start, n_prompt-1) in chunks of maxB: base=0 for a fresh prompt's
    // first chunk, base=done to CONTINUE (cache-hit suffix, or the next chunk of a >maxB
    // prompt). The final token is left for the serial step, which yields the first
    // generated token reading the batched-filled K/V + delta-rule S + conv window exactly
    // as the all-serial path would. A remaining <16-token tail prefills per-token serially.
    uint32_t target = n_prompt >= 1 ? n_prompt - 1 : 0;
    uint32_t done = start;
    auto batch_to = [&](uint32_t limit) {
        while (limit > done && limit - done >= 16) {
            uint32_t chunk = std::min(e->maxB, limit - done);
            std::vector<float> unused;
            e->prefillBatchLast(prompt + done, chunk, slot, unused, /*wantLogits=*/false, /*base=*/done);
            done += chunk;
        }
    };
    // Cross-turn reuse: snapshot the conversation-HISTORY prefix (before the
    // generation scaffold), not the whole prompt. snap_prefix is the caller's
    // history-boundary token count; keying the snapshot there makes it a genuine
    // prefix of the NEXT turn's prompt (which appends the reply + new turn after
    // the same history), so that turn restores it and prefills only its delta.
    // Keying by the full prompt — which ends in "...assistant\n<think></think>" —
    // never matches the next turn (that scaffold is replaced by the real reply).
    uint32_t snapAt = (snap_prefix > start && snap_prefix < target) ? snap_prefix : 0;
    uint32_t snapPos = 0;
    if (!getenv("QK_NO_BATCH")) {
        if (snapAt) {
            batch_to(snapAt);
            if (e->shareFork && done > start) {
                s.pos = done; s.prompt.assign(prompt, prompt + n_prompt);
                e->snapshotSlot(slot);    // keyed by prompt[0:done] = history prefix
                snapPos = done;
            }
        }
        batch_to(target);
    }
    // Prefix-cache hit-rate + prefill-cost instrumentation (QK_PCACHE_LOG).
    // reuse = tokens restored from a cached prefix; prefill = tokens (re)computed
    // this request; snap = history-boundary position cached for the next turn.
    if (getenv("QK_PCACHE_LOG")) {
        qkStatsLine("[pcache] slot=%u prompt=%u reuse=%u prefill=%u hit=%d snap=%u\n",
                    slot, n_prompt, start, done - start, cidx >= 0 ? 1 : 0, snapPos);
    }
    s.cursor = done; s.pos = done;
    s.active = true; s.prompt.assign(prompt, prompt + n_prompt);
    s.genTokens.clear();
    s.gen = 0; s.maxGen = max_gen; s.last = 0;
    s.resetSpec();  // fresh queue + n-gram index (history changed; rebuilt lazily)
    // If we did not take a history snapshot (snap_prefix disabled / prompt too
    // short), fall back to caching the full prefill so identical-prompt requests
    // still fork off it.
    if (e->shareFork && snapPos == 0 && done > start) e->snapshotSlot(slot);
    return 0;
}

__attribute__((visibility("default")))
void qk_slot_cancel(qk_engine* e, uint32_t slot) {
    if (e && slot < e->nSlots) {
        e->slots[slot].active = false;
        e->slots[slot].prompt.clear();
        e->slots[slot].resetSpec();
    }
}

__attribute__((visibility("default")))
int qk_step_chunk(qk_engine* e, uint32_t* out_tokens, uint32_t* out_counts, uint32_t* out_finished) {
    if (!e || !out_tokens || !out_counts || !out_finished) return -1;
    if (e->splitStage()) return -5;  // split stages are driven via qk_stage_run
    return e->stepChunk(out_tokens, out_counts, out_finished);
}

}  // extern "C"

// Batched-prefill GEMM validation: Y[N,M] = X[N,K]·W[M,K]^T vs a CPU reference.
static bool caseBGemm(VkCtx& c, uint32_t M, uint32_t K, uint32_t N, uint32_t iters) {
    printf("\n== batched GEMM  Y[%u,%u] = X[%u,%u] . W[%u,%u]^T  (Q8_0) ==\n", N, M, N, K, M, K);
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
    uint32_t gx = (M + 127) / 128, gz = (N + 63) / 64;  // BM=128, BN=64 (see gemm_q8_0.comp GRID)
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    auto bindDisp = [&]() {
        vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
        vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(c.cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pc);
    };
    begin(); bindDisp(); vkCmdDispatch(c.cb, gx, 1, gz); submitWait();  // warm-up + produce Y
    begin();
    bindDisp();
    for (uint32_t it = 0; it < iters; it++) {
        vkCmdDispatch(c.cb, gx, 1, gz);
        vkCmdPipelineBarrier(c.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    auto t0 = std::chrono::steady_clock::now();
    submitWait();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;

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
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    double peak = 52.0;                                             // RX 7900 XT ~52 TFLOP/s FP32
    double serialMs = (double)N * ((double)M * K / 32.0 * 34.0) / 938e6;  // N GEMVs at ~938 GB/s
    printf("  N=%-5u %7.3f ms | %5.1f TFLOP/s (%2.0f%% peak) | vs serial %6.1f ms = %.2fx | err/rms %.1g -> %s\n",
           N, ms, tflops, tflops / peak * 100, serialMs, serialMs / ms, maxErr / rms, ok ? "PASS" : "FAIL");
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

static bool readIdsFile(const char* path, std::vector<uint32_t>& out) {
    FILE* f = fopen(path, "r");
    if (!f) { perror(path); return false; }
    int v;
    while (fscanf(f, "%d%*[, \n]", &v) == 1) out.push_back((uint32_t)v);
    fclose(f);
    return !out.empty();
}

static uint64_t fnv1a64(const void* data, size_t n,
                        uint64_t h = 14695981039346656037ull) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

struct GpuMetricSample {
    double ms = 0.0;
    uint16_t hotspot = 0xffff, memTemp = 0xffff;
    uint16_t gfxActivity = 0xffff, memActivity = 0xffff, socketPower = 0xffff;
    uint16_t avgGfx = 0xffff, avgUclk = 0xffff;
    uint16_t gfx = 0xffff, uclk = 0xffff;
    uint32_t throttle = 0;
    uint64_t indepThrottle = 0;
};

static bool readGpuMetricsV13(const char* path, GpuMetricSample& s) {
    uint8_t b[128] = {};
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    size_t n = fread(b, 1, sizeof b, f);
    fclose(f);
    auto u16 = [&](size_t off) { uint16_t v; memcpy(&v, b + off, 2); return v; };
    auto u32 = [&](size_t off) { uint32_t v; memcpy(&v, b + off, 4); return v; };
    auto u64 = [&](size_t off) { uint64_t v; memcpy(&v, b + off, 8); return v; };
    if (n < 120 || u16(0) < 120 || b[2] != 1 || b[3] != 3) return false;
    s.hotspot = u16(6); s.memTemp = u16(8);
    s.gfxActivity = u16(16); s.memActivity = u16(18); s.socketPower = u16(22);
    s.avgGfx = u16(40); s.avgUclk = u16(44);
    s.gfx = u16(54); s.uclk = u16(58);
    s.throttle = u32(68); s.indepThrottle = u64(112);
    return true;
}

#ifndef QK_LIBRARY
int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "suite";

    if (mode == "list") {
        listTensors(argc > 2 ? argv[2] : "");
        return 0;
    }

    if (mode == "serve-bench") {
        // Serving-shaped timing without the completion snapshot: slot_start
        // batch-prefills prompt[0:n-1], then every measured step is the normal
        // n=1 decode path. max_gen stays one beyond the measurement window and
        // cancellation prevents the finishing copyStripes() from polluting it.
        if (argc < 4) {
            fprintf(stderr, "usage: qk serve-bench <ids-file> <nDecode> [tmax]\n");
            return 1;
        }
        std::vector<uint32_t> prompt;
        if (!readIdsFile(argv[2], prompt)) return 1;
        uint32_t nDecode = (uint32_t)atoi(argv[3]);
        uint32_t tmax = argc > 4 ? (uint32_t)atoi(argv[4]) : 8192;
        uint32_t chunk = 8;
        if (const char* v = getenv("QK_CHUNK")) chunk = (uint32_t)atoi(v);
        uint32_t prefix = 0, idleMs = 0;
        if (const char* v = getenv("QK_BENCH_PREFIX")) prefix = (uint32_t)atoi(v);
        if (const char* v = getenv("QK_BENCH_IDLE_MS")) idleMs = (uint32_t)atoi(v);
        bool trace = getenv("QK_BENCH_TRACE") != nullptr;
        if (!nDecode || !chunk || nDecode % chunk || prefix % chunk ||
            prompt.size() + (size_t)prefix + nDecode + 1 > tmax) {
            fprintf(stderr, "serve-bench: need nDecode>0 and prefix/decode divisible by "
                            "QK_CHUNK=%u, with prompt+prefix+nDecode+1 <= tmax\n", chunk);
            return 1;
        }
        qk_config cfg{1, tmax, chunk};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
        auto p0 = std::chrono::steady_clock::now();
        int rc = qk_slot_start(e, 0, prompt.data(), (uint32_t)prompt.size(),
                               prefix + nDecode + 1, 0);
        auto p1 = std::chrono::steady_clock::now();
        if (rc) { fprintf(stderr, "serve-bench: slot_start rc=%d\n", rc); qk_close(e); return 1; }

        std::vector<uint32_t> outTok(chunk), outCnt(1), prefixGen, gen;
        prefixGen.reserve(prefix);
        gen.reserve(nDecode);
        uint32_t finMask = 0;
        auto drive = [&](uint32_t want, std::vector<uint32_t>& dst,
                         const char* phase, bool traceChunks) -> bool {
            while (dst.size() < want) {
                auto c0 = std::chrono::steady_clock::now();
                int active = qk_step_chunk(e, outTok.data(), outCnt.data(), &finMask);
                auto c1 = std::chrono::steady_clock::now();
                if (active <= 0 || finMask || outCnt[0] == 0) {
                    fprintf(stderr,
                            "serve-bench: %s early stop active=%d fin=%u produced=%zu\n",
                            phase, active, finMask, dst.size());
                    return false;
                }
                dst.insert(dst.end(), outTok.begin(), outTok.begin() + outCnt[0]);
                if (traceChunks) {
                    double cms = std::chrono::duration<double, std::milli>(c1 - c0).count();
                    fprintf(stderr,
                            "[bench-trace] phase=%s end=%zu tokens=%u ms=%.3f tok/s=%.2f\n",
                            phase, dst.size(), outCnt[0], cms, outCnt[0] * 1000.0 / cms);
                }
            }
            return true;
        };

        auto w0 = std::chrono::steady_clock::now();
        if (prefix && !drive(prefix, prefixGen, "prefix", false)) {
            qk_close(e);
            return 1;
        }
        auto w1 = std::chrono::steady_clock::now();
        if (prefix) {
            fprintf(stderr, "[bench] prefix complete: %u tokens; idle=%u ms\n", prefix, idleMs);
            fflush(stderr);
        }
        if (idleMs) std::this_thread::sleep_for(std::chrono::milliseconds(idleMs));
        if (getenv("QK_STEP_PROF_DEFER")) e->profArmed = true;
        fprintf(stderr, "[bench] measure begin: %u tokens\n", nDecode);
        fflush(stderr);

        const char* metricPath = getenv("QK_BENCH_GPU_METRICS");
        uint32_t metricMs = 50;
        if (const char* v = getenv("QK_BENCH_METRICS_MS"))
            metricMs = std::max(10u, (uint32_t)atoi(v));
        std::atomic<bool> metricStop{false};
        std::vector<GpuMetricSample> metrics;
        auto metricOrigin = std::chrono::steady_clock::now();
        std::thread metricThread;
        if (metricPath) {
            metricThread = std::thread([&]() {
                while (!metricStop.load(std::memory_order_relaxed)) {
                    auto m0 = std::chrono::steady_clock::now();
                    GpuMetricSample s;
                    if (readGpuMetricsV13(metricPath, s)) {
                        auto m1 = std::chrono::steady_clock::now();
                        s.ms = std::chrono::duration<double, std::milli>(
                                   m0 + (m1 - m0) / 2 - metricOrigin).count();
                        metrics.push_back(s);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(metricMs));
                }
            });
        }
        auto d0 = std::chrono::steady_clock::now();
        if (!drive(nDecode, gen, "measure", trace)) {
            metricStop.store(true, std::memory_order_relaxed);
            if (metricThread.joinable()) metricThread.join();
            qk_close(e);
            return 1;
        }
        auto d1 = std::chrono::steady_clock::now();
        metricStop.store(true, std::memory_order_relaxed);
        if (metricThread.joinable()) metricThread.join();
        qk_slot_cancel(e, 0);
        double pms = std::chrono::duration<double, std::milli>(p1 - p0).count();
        double wms = std::chrono::duration<double, std::milli>(w1 - w0).count();
        double dms = std::chrono::duration<double, std::milli>(d1 - d0).count();
        size_t pn = prompt.size() - 1;
        printf("serve-bench: prompt %zu | batch-prefill %zu tokens in %.1f ms = %.2f tok/s "
               "| prefix %u in %.1f ms | idle %u ms | decode %u tokens in %.1f ms "
               "= %.2f tok/s (%.3f ms/tok)\n",
               prompt.size(), pn, pms, pn * 1000.0 / pms, prefix, wms, idleMs,
               nDecode, dms,
               nDecode * 1000.0 / dms, dms / nDecode);
        if (!metrics.empty()) {
            double gfxSum = 0.0, gfxInv = 0.0, uclkSum = 0.0;
            double activity = 0.0, memActivity = 0.0, power = 0.0;
            uint32_t gfxN = 0, gfxNonzero = 0, gfxHigh = 0, gfxZero = 0;
            uint32_t uclkN = 0, uclkHigh = 0, statN = 0;
            uint16_t maxHotspot = 0, maxMemTemp = 0;
            uint32_t throttle = 0; uint64_t indepThrottle = 0;
            for (const auto& s : metrics) {
                if (s.gfx != 0xffff) {
                    gfxN++; gfxSum += s.gfx;
                    if (s.gfx == 0) gfxZero++;
                    else { gfxNonzero++; gfxInv += 1.0 / s.gfx; }
                    if (s.gfx >= 2134) gfxHigh++;
                }
                if (s.uclk != 0xffff) {
                    uclkN++; uclkSum += s.uclk;
                    if (s.uclk >= 1200) uclkHigh++;
                }
                if (s.gfxActivity != 0xffff && s.memActivity != 0xffff &&
                    s.socketPower != 0xffff) {
                    statN++; activity += s.gfxActivity; memActivity += s.memActivity;
                    power += s.socketPower;
                }
                if (s.hotspot != 0xffff) maxHotspot = std::max(maxHotspot, s.hotspot);
                if (s.memTemp != 0xffff) maxMemTemp = std::max(maxMemTemp, s.memTemp);
                throttle |= s.throttle; indepThrottle |= s.indepThrottle;
                if (getenv("QK_BENCH_METRICS_DETAIL"))
                    fprintf(stderr,
                            "[dpm-sample] ms=%7.2f gfx=%u uclk=%u avg_gfx=%u avg_uclk=%u "
                            "gfx_busy=%u mem_busy=%u power=%u hotspot=%u memtemp=%u\n",
                            s.ms, s.gfx, s.uclk, s.avgGfx, s.avgUclk,
                            s.gfxActivity, s.memActivity, s.socketPower,
                            s.hotspot, s.memTemp);
            }
            fprintf(stderr,
                    "[dpm] samples=%zu interval=%ums gfx_mean=%.1f gfx_hmean=%.1f "
                    "gfx_high=%.1f%% gfx_zero=%.1f%% uclk_mean=%.1f uclk_high=%.1f%% "
                    "gfx_busy=%.1f%% mem_busy=%.1f%% power=%.1fW hotspot_max=%uC "
                    "memtemp_max=%uC throttle=0x%x indep=0x%llx\n",
                    metrics.size(), metricMs, gfxN ? gfxSum / gfxN : 0.0,
                    gfxNonzero ? gfxNonzero / gfxInv : 0.0,
                    gfxN ? 100.0 * gfxHigh / gfxN : 0.0,
                    gfxN ? 100.0 * gfxZero / gfxN : 0.0,
                    uclkN ? uclkSum / uclkN : 0.0,
                    uclkN ? 100.0 * uclkHigh / uclkN : 0.0,
                    statN ? activity / statN : 0.0,
                    statN ? memActivity / statN : 0.0,
                    statN ? power / statN : 0.0, maxHotspot, maxMemTemp,
                    throttle, (unsigned long long)indepThrottle);
        }
        if (getenv("QK_BENCH_NO_TOKENS")) {
            printf("serve-bench hashes: prefix=%016llx decode=%016llx\n",
                   (unsigned long long)fnv1a64(prefixGen.data(), prefixGen.size() * 4),
                   (unsigned long long)fnv1a64(gen.data(), gen.size() * 4));
        } else {
            if (prefix) {
                printf("PREFIX:");
                for (uint32_t t : prefixGen) printf(" %u", t);
                printf("\n");
            }
            printf("GEN:");
            for (uint32_t t : gen) printf(" %u", t);
            printf("\n");
        }
        qk_close(e);
        return 0;
    }

    if (mode == "stage-bench") {
        // Standalone first-stage benchmark (notably the 80B head [0,12)).
        // Prefill uses the production batch path; decode advances the same live
        // state with deterministic valid token inputs and copies each hidden row
        // to the host, exactly as a split head must do for its next stage.
        if (argc < 4) {
            fprintf(stderr,
                    "usage: qk stage-bench <ids-file> <nDecode> [layers=0:12] [tmax=8192]\n");
            return 1;
        }
        std::vector<uint32_t> prompt;
        if (!readIdsFile(argv[2], prompt)) return 1;
        uint32_t nDecode = (uint32_t)atoi(argv[3]);
        const char* layers = argc > 4 ? argv[4] : "0:12";
        uint32_t tmax = argc > 5 ? (uint32_t)atoi(argv[5]) : 8192;
        if (!nDecode || prompt.size() + (size_t)nDecode > tmax) {
            fprintf(stderr, "stage-bench: need nDecode>0 and prompt+nDecode <= tmax\n");
            return 1;
        }
        setenv("QK_LAYERS", layers, 1);
        qk_config cfg{1, tmax, 8};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
        for (uint32_t tok : prompt) {
            if (tok >= e->vocab) {
                fprintf(stderr, "stage-bench: token %u is outside vocab %u\n", tok, e->vocab);
                qk_close(e);
                return 1;
            }
        }
        if (!e->firstStage() || e->lastStage()) {
            fprintf(stderr, "stage-bench: layers must select a non-final first stage, got [%u,%u)\n",
                    e->lFirst, e->lEnd);
            qk_close(e);
            return 1;
        }
        const size_t width = qk_engine::nEmbd;
        std::vector<float> prefillOut(prompt.size() * width);
        auto p0 = std::chrono::steady_clock::now();
        int rc = e->stageRun(0, prompt.data(), nullptr, (uint32_t)prompt.size(), 0,
                             prefillOut.data(), nullptr);
        auto p1 = std::chrono::steady_clock::now();
        if (rc) { fprintf(stderr, "stage-bench: prefill rc=%d\n", rc); qk_close(e); return 1; }

        std::vector<float> decodeOut((size_t)nDecode * width);
        auto d0 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < nDecode; i++) {
            uint32_t tok = prompt[(prompt.size() - 1 + i) % prompt.size()];
            rc = e->stageRun(0, &tok, nullptr, 1, (uint32_t)prompt.size() + i,
                             decodeOut.data() + (size_t)i * width, nullptr);
            if (rc) break;
        }
        auto d1 = std::chrono::steady_clock::now();
        if (rc) { fprintf(stderr, "stage-bench: decode rc=%d\n", rc); qk_close(e); return 1; }
        double pms = std::chrono::duration<double, std::milli>(p1 - p0).count();
        double dms = std::chrono::duration<double, std::milli>(d1 - d0).count();
        uint64_t ph = fnv1a64(prefillOut.data(), prefillOut.size() * sizeof(float));
        uint64_t dh = fnv1a64(decodeOut.data(), decodeOut.size() * sizeof(float));
        printf("stage-bench: layers [%u,%u) prompt %zu in %.1f ms = %.2f tok/s | "
               "decode %u in %.1f ms = %.2f tok/s (%.3f ms/tok)\n",
               e->lFirst, e->lEnd, prompt.size(), pms, prompt.size() * 1000.0 / pms,
               nDecode, dms, nDecode * 1000.0 / dms, dms / nDecode);
        printf("stage-bench hashes: prefill=%016llx decode=%016llx\n",
               (unsigned long long)ph, (unsigned long long)dh);
        qk_close(e);
        return 0;
    }

    if (mode == "serve-test2") {
        // Two slots, DIFFERENT prompts, admitted at a configurable step offset —
        // the serving shape that wedges the GPU with >=2 active slots. Prompt 2
        // is admitted after `delay` step_chunk calls (0 = simultaneous admission,
        // like two requests arriving together).
        if (argc < 5) {
            fprintf(stderr, "usage: qk serve-test2 <ids1> <ids2> <nGen> [tmax] [delay]\n");
            return 1;
        }
        auto readIds = [](const char* path, std::vector<uint32_t>& out) {
            FILE* f = fopen(path, "r");
            if (!f) { perror(path); return false; }
            int v;
            while (fscanf(f, "%d%*[, \n]", &v) == 1) out.push_back((uint32_t)v);
            fclose(f);
            return true;
        };
        std::vector<uint32_t> p1, p2;
        if (!readIds(argv[2], p1) || !readIds(argv[3], p2)) return 1;
        uint32_t nGen = (uint32_t)atoi(argv[4]);
        uint32_t tmax = argc > 5 ? (uint32_t)atoi(argv[5]) : 4096;
        uint32_t delay = argc > 6 ? (uint32_t)atoi(argv[6]) : 0;
        qk_config cfg{2, tmax, 8};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
        uint32_t ch = qk_chunk(e);
        std::vector<uint32_t> outTok((size_t)2 * ch), outCnt(2);
        std::vector<std::vector<uint32_t>> gen(2);
        uint32_t finMask = 0;
        auto t0 = std::chrono::steady_clock::now();
        fprintf(stderr, "[t2] slot_start 0 (%zu toks)\n", p1.size());
        if (qk_slot_start(e, 0, p1.data(), (uint32_t)p1.size(), nGen, 0)) { fprintf(stderr, "start0 failed\n"); return 1; }
        uint32_t steps = 0;
        bool started2 = false;
        while (true) {
            if (!started2 && steps >= delay) {
                fprintf(stderr, "[t2] slot_start 1 (%zu toks) at step %u\n", p2.size(), steps);
                if (qk_slot_start(e, 1, p2.data(), (uint32_t)p2.size(), nGen, 0)) { fprintf(stderr, "start1 failed\n"); return 1; }
                started2 = true;
            }
            int active = qk_step_chunk(e, outTok.data(), outCnt.data(), &finMask);
            if (active < 0) { fprintf(stderr, "step error\n"); return 1; }
            if (active == 0 && started2) break;
            steps++;
            for (uint32_t s = 0; s < 2; s++)
                for (uint32_t i = 0; i < outCnt[s]; i++) gen[s].push_back(outTok[s * ch + i]);
            if (steps % 64 == 0)
                fprintf(stderr, "[t2] step %u gen0=%zu gen1=%zu\n", steps, gen[0].size(), gen[1].size());
        }
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        printf("serve-test2: OK gen0=%zu gen1=%zu tokens in %.1f ms (%u steps)\n",
               gen[0].size(), gen[1].size(), ms, steps);
        qk_close(e);
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
        uint32_t chunk = 8; // QK_CHUNK: sweep GPU-steps-per-host-sync (1..32)
        if (const char* e = getenv("QK_CHUNK")) chunk = (uint32_t)atoi(e);
        qk_config cfg{nSlots, tmax, chunk};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
        uint32_t ch = qk_chunk(e);
        std::vector<std::vector<uint32_t>> gen(nSlots);
        std::vector<uint32_t> outTok((size_t)nSlots * ch), outCnt(nSlots);
        uint32_t finMask = 0;
        auto t0 = std::chrono::steady_clock::now();  // time the FULL request incl. prefill (slot_start)
        for (uint32_t s = 0; s < nSlots; s++)
            qk_slot_start(e, s, prompt.data(), (uint32_t)prompt.size(), nGen, 0);
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
            qk_slot_start(e, 0, ids.data(), (uint32_t)ids.size(), g, 0);
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

    if (mode == "speccmp") {
        // Spec-decode P1 harness: the full accept / reject / rollback state
        // machine vs the serial stream. Drafts are taken from the serial
        // reference (oracle) with every C-th drafted position corrupted, forcing
        // partial accepts at controlled spots: C=0 pure oracle (always full
        // accept + promote), C=1 every draft wrong (worst case: k=1 rounds, a
        // commit pass each round). The reconstructed output must be
        // token-identical to serial in EVERY mode — that is the rollback proof.
        if (argc < 4) {
            fprintf(stderr, "usage: qk speccmp <ids-file> <nGen> [K] [C,C,...]\n");
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
        uint32_t K = argc > 4 ? (uint32_t)atoi(argv[4]) : 8;
        std::vector<uint32_t> Cs{0, 4, 2, 1};
        if (argc > 5) {
            Cs.clear();
            for (const char* p = argv[5]; *p;) {
                Cs.push_back((uint32_t)strtoul(p, (char**)&p, 10));
                if (*p == ',') p++;
            }
        }
        qk_config cfg{1, 4096, 8};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
        if (K < 2 || K > e->maxB) { fprintf(stderr, "K must be in [2, maxB]\n"); return 1; }

        // serial reference
        std::vector<uint32_t> S = prompt;
        double serialMsTok = 0;
        {
            qk_slot_start(e, 0, prompt.data(), (uint32_t)prompt.size(), nGen, 0);
            uint32_t ch = qk_chunk(e), fin = 0;
            std::vector<uint32_t> ot(ch), oc(1);
            double decodeMs = 0;
            uint32_t decodeTok = 0;
            bool prefilled = false;
            while (true) {
                auto t0 = std::chrono::steady_clock::now();
                int act = qk_step_chunk(e, ot.data(), oc.data(), &fin);
                double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count();
                if (act <= 0) break;
                for (uint32_t i = 0; i < oc[0]; i++) S.push_back(ot[i]);
                if (prefilled) { decodeMs += ms; decodeTok += oc[0]; }
                if (oc[0] > 0) prefilled = true;
                if (fin & 1u) break;
            }
            serialMsTok = decodeTok ? decodeMs / decodeTok : 0;
        }
        uint32_t np = (uint32_t)prompt.size();
        uint32_t sGen = (uint32_t)S.size() - np;
        if (sGen < 4) { fprintf(stderr, "stream too short (early EOS?)\n"); return 1; }
        printf("speccmp: prompt %u, serial %u gen tokens at %.2f ms/tok, K=%u\n",
               np, sGen, serialMsTok, K);
        printf("  %-8s %7s %7s %7s %9s %8s %10s   %s\n",
               "corrupt", "rounds", "avg_k", "full%", "commits", "ms/tok", "vs_serial", "exact");

        bool allOk = true;
        std::vector<float> dummy;
        std::vector<uint32_t> am(e->maxB), toks(e->maxB);
        for (uint32_t C : Cs) {
            // fresh live state: batch-prefill the prompt from empty; final chunk's
            // argmax gives the first generated token
            uint32_t next = 0;
            for (uint32_t off = 0; off < np;) {
                uint32_t n = std::min(e->maxB, np - off);
                bool last = off + n >= np;
                e->prefillBatchLast(prompt.data() + off, n, 0, dummy, false, off,
                                    last ? am.data() : nullptr);
                if (last) next = am[n - 1];
                off += n;
            }
            std::vector<uint32_t> out{next};
            uint32_t pos = np, rounds = 0, fullAcc = 0, commits = 0;
            uint64_t sumK = 0;
            double specMs = 0;
            bool hitEos = false;
            while (out.size() < sGen && !hitEos) {
                uint32_t Kr = std::min(K, (uint32_t)S.size() - pos);
                if (Kr < 1) break;
                toks[0] = next;
                for (uint32_t i = 1; i < Kr; i++) {
                    toks[i] = S[pos + i];
                    if (C > 0 && i % C == 0) toks[i] = toks[i] == 5 ? 6 : 5;  // guaranteed wrong
                }
                auto t0 = std::chrono::steady_clock::now();
                e->verifyRound(toks.data(), Kr, 0, pos, am.data());
                uint32_t k = 1;
                while (k < Kr && toks[k] == am[k - 1]) k++;
                if (k == Kr) {
                    e->promoteScratch(0);
                    fullAcc++;
                } else {
                    // rollback: live state is still at `pos`; re-run the k accepted
                    // tokens in live mode to commit recurrent state + K/V
                    e->prefillBatchLast(toks.data(), k, 0, dummy, false, pos);
                    commits++;
                }
                specMs += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
                rounds++;
                sumK += k;
                // emit: the accepted drafts (toks[1..k) == am[0..k-1)) plus the
                // round's new token am[k-1]. Only am[k-1] can be EOS (accepted
                // drafts come from S, which contains none).
                for (uint32_t i = 1; i < k; i++) out.push_back(toks[i]);
                if (am[k - 1] == qk_eos_token(e)) hitEos = true;
                else out.push_back(am[k - 1]);
                next = am[k - 1];
                pos += k;
            }
            if (out.size() > sGen) out.resize(sGen);
            bool ok = out.size() == sGen && std::equal(out.begin(), out.end(), S.begin() + np);
            allOk = allOk && ok;
            printf("  C=%-6u %7u %7.2f %6.0f%% %9u %8.2f %9.2fx   %s\n",
                   C, rounds, rounds ? (double)sumK / rounds : 0,
                   rounds ? 100.0 * fullAcc / rounds : 0, commits,
                   out.empty() ? 0 : specMs / out.size(),
                   specMs > 0 ? serialMsTok * out.size() / specMs : 0,
                   ok ? "EXACT" : "**MISMATCH**");
            if (!ok) {
                size_t d = 0;
                while (d < out.size() && d < sGen && out[d] == S[np + d]) d++;
                printf("      first divergence at gen token %zu/%u\n", d, sGen);
            }
        }
        printf("speccmp: %s\n", allOk ? "ROLLBACK TOKEN-EXACT (all corruption modes)"
                                      : "DIVERGENCE (see above)");
        qk_close(e);
        return allOk ? 0 : 1;
    }

    if (mode == "pipe" || mode == "pipe-worker") {
        // Pipeline-split (QK_LAYERS) harness — task: run the model as N stages
        // that each own a layer range, handing the ~8 KB/token residual row
        // across the boundary. Greedy output must be token-exact vs the
        // unsplit engine (`qk serve-test <ids> <nGen> 1 <tmax>`).
        //
        //   qk pipe <ids-file> <nGen> [split=20] [tmax=4096] [host:port]
        //     Stage 1 = layers [0,split) in-process. Stage 2 = layers
        //     [split,40): in-process too (default), or a remote
        //     `qk pipe-worker` when host:port is given.
        //   qk pipe-worker <port> [a:b=20:40] [tmax=4096]
        //     Serve a stage over TCP. Frame: {op,slot,n,base} u32 header;
        //     op1 payload = n ids (first stage) or n*2048 f32 rows; reply =
        //     n ids (last stage) or n*2048 f32 rows. op2 ends the connection.
        signal(SIGPIPE, SIG_IGN);
        auto readAll = [](int fd, void* p, size_t n) -> bool {
            uint8_t* b = (uint8_t*)p;
            while (n) { ssize_t r = read(fd, b, n); if (r <= 0) return false; b += r; n -= (size_t)r; }
            return true;
        };
        auto writeAll = [](int fd, const void* p, size_t n) -> bool {
            const uint8_t* b = (const uint8_t*)p;
            while (n) { ssize_t r = write(fd, b, n); if (r <= 0) return false; b += r; n -= (size_t)r; }
            return true;
        };
        struct PipeHdr { uint32_t op, slot, n, base, topk; };
        const uint32_t nEmbd = qk_engine::nEmbd;
        // Layer count comes from the GGUF header (35B: 40, 80B: 48) — needed
        // before any engine exists to parse/validate the stage boundaries.
        uint32_t nLay = 40;
        {
            Gguf gh;
            if (gh.open(ggufPath()))
                nLay = (uint32_t)gh.kvInt(gh.kvStr("general.architecture", "") + ".block_count", 40);
        }
        // Connection hello: client sends the magic; worker replies
        // {magic, lFirst, lEnd, nLayer, nEmbd, nSlots, nCtx} so mismatched
        // builds/splits fail loudly instead of streaming garbage. Bump the
        // magic on any wire change. qkp2 = qkp1 + state ops (op3 save /
        // op4 load, idx in the n field, 4-byte status reply). qkp3 = qkp2 +
        // a 5th header word `topk`: on an op1 frame to a LAST stage, the
        // reply appends topk (u32 id, f32 logit) pairs for the final
        // position, descending — the head-side sampler's candidates.
        const uint32_t kPipeMagic = 0x716b7033;  // "qkp3"
        char err[256] = {0};

        if (mode == "pipe-worker") {
            if (argc < 3) {
                fprintf(stderr,
                        "usage: qk pipe-worker <port> [a:b=20:40] [tmax=4096] [slots=1]\n"
                        "       (slots/tmax must cover the head server's --slots/--ctx)\n");
                return 1;
            }
            uint16_t port = (uint16_t)atoi(argv[2]);
            setenv("QK_LAYERS", argc > 3 ? argv[3] : "20:40", 1);
            uint32_t tmax = argc > 4 ? (uint32_t)atoi(argv[4]) : 4096;
            uint32_t wslots = argc > 5 ? (uint32_t)atoi(argv[5]) : 1;
            qk_config cfg{wslots, tmax, 8};
            qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
            if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
            int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
            setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
            sockaddr_in sa{};
            sa.sin_family = AF_INET; sa.sin_addr.s_addr = htonl(INADDR_ANY); sa.sin_port = htons(port);
            if (bind(ls, (sockaddr*)&sa, sizeof sa) || listen(ls, 1)) { perror("bind/listen"); return 1; }
            fprintf(stderr, "[pipe-worker] layers [%u,%u) nCtx %u listening on :%u\n",
                    e->lFirst, e->lEnd, tmax, port);
            std::vector<uint32_t> toks, ids;
            std::vector<float> hin, hout;
            while (true) {
                int fd = accept(ls, nullptr, nullptr);
                if (fd < 0) continue;
                setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                uint32_t cmagic = 0;
                if (!readAll(fd, &cmagic, 4) || cmagic != kPipeMagic) {
                    fprintf(stderr, "[pipe-worker] bad hello magic %08x — old/foreign client?\n",
                            cmagic);
                    close(fd);
                    continue;
                }
                uint32_t hello[7] = {kPipeMagic, e->lFirst, e->lEnd, nLay, nEmbd,
                                     e->nSlots,  e->nCtx};
                if (!writeAll(fd, hello, sizeof hello)) { close(fd); continue; }
                fprintf(stderr, "[pipe-worker] client connected\n");
                PipeHdr h;
                while (readAll(fd, &h, sizeof h) && h.op != 2) {
                    if (h.op == 3 || h.op == 4) {
                        // state save/load: idx rides in the n field, live token
                        // count in the topk word (0 = full stripe); 4-byte status
                        uint32_t rc = (uint32_t)(h.op == 3 ? qk_state_save(e, h.slot, h.n, h.topk)
                                                           : qk_state_load(e, h.slot, h.n, h.topk));
                        if (!writeAll(fd, &rc, 4)) break;
                        continue;
                    }
                    if (h.op != 1 || h.n < 1 || h.slot >= e->nSlots ||
                        (size_t)h.base + h.n > e->nCtx ||
                        h.topk > 256 || (h.topk && !e->lastStage()))
                        break;
                    int rc;
                    if (e->firstStage()) {
                        toks.resize(h.n);
                        if (!readAll(fd, toks.data(), (size_t)h.n * 4)) break;
                    } else {
                        hin.resize((size_t)h.n * nEmbd);
                        if (!readAll(fd, hin.data(), hin.size() * 4)) break;
                    }
                    if (e->lastStage()) ids.resize(h.n);
                    else hout.resize((size_t)h.n * nEmbd);
                    rc = e->stageRun(h.slot, e->firstStage() ? toks.data() : nullptr,
                                     e->firstStage() ? nullptr : hin.data(), h.n, h.base,
                                     e->lastStage() ? nullptr : hout.data(),
                                     e->lastStage() ? ids.data() : nullptr);
                    if (rc) { fprintf(stderr, "[pipe-worker] stage_run rc=%d\n", rc); break; }
                    bool ok2 = e->lastStage() ? writeAll(fd, ids.data(), (size_t)h.n * 4)
                                              : writeAll(fd, hout.data(), hout.size() * 4);
                    if (ok2 && h.topk) {
                        // Sampling candidates: exactly h.topk (id, logit)
                        // pairs for the final position — the head samples.
                        std::vector<uint32_t> tid(h.topk);
                        std::vector<float> tval(h.topk);
                        if (e->stageTopK(h.topk, tid.data(), tval.data())) {
                            fprintf(stderr, "[pipe-worker] stage_topk failed\n");
                            break;
                        }
                        std::vector<uint8_t> pk((size_t)h.topk * 8);
                        for (uint32_t i = 0; i < h.topk; i++) {
                            memcpy(pk.data() + (size_t)i * 8, &tid[i], 4);
                            memcpy(pk.data() + (size_t)i * 8 + 4, &tval[i], 4);
                        }
                        ok2 = writeAll(fd, pk.data(), pk.size());
                    }
                    if (!ok2) break;
                }
                close(fd);
                fprintf(stderr, "[pipe-worker] client gone\n");
            }
        }

        // driver: qk pipe <ids> <nGen> [split] [tmax] [host:port]
        // `split` is one boundary ("20") or a comma list ("13,27" -> THREE
        // in-process stages) — the multi-boundary form exercises middle
        // stages (hidden in AND hidden out). Remote mode needs exactly one
        // boundary (the worker is the tail).
        if (argc < 4) {
            fprintf(stderr,
                    "usage: qk pipe <ids-file> <nGen> [split=20|a,b,..] [tmax=4096] [host:port]\n");
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
        std::vector<uint32_t> bounds;
        for (const char* p = argc > 4 ? argv[4] : "20"; *p;) {
            bounds.push_back((uint32_t)strtoul(p, (char**)&p, 10));
            if (*p == ',') p++;
        }
        uint32_t tmax = argc > 5 ? (uint32_t)atoi(argv[5]) : 4096;
        const char* net = argc > 6 ? argv[6] : nullptr;
        uint32_t np = (uint32_t)prompt.size();
        bool boundsOk = !bounds.empty();
        for (size_t i = 0; i < bounds.size(); i++)
            boundsOk &= bounds[i] >= 1 && bounds[i] < nLay && (i == 0 || bounds[i] > bounds[i - 1]);
        if (!boundsOk || np < 1 || np + nGen > tmax || (net && bounds.size() != 1)) {
            fprintf(stderr, "pipe: bad args (boundaries ascending in 1..%u, one boundary with "
                            "host:port, prompt+nGen <= tmax)\n", nLay - 1);
            return 1;
        }
        uint32_t split = bounds[0];
        char lay[24];
        qk_config cfg{1, tmax, 8};
        // In-process stages: [0,b0), [b0,b1), ..., and — unless remote — [bLast,40).
        std::vector<qk_engine*> eng;
        uint32_t lo = 0;
        for (size_t s = 0; s < bounds.size() + (net ? 0 : 1); s++) {
            uint32_t hi = s < bounds.size() ? bounds[s] : nLay;
            snprintf(lay, sizeof lay, "%u:%u", lo, hi);
            setenv("QK_LAYERS", lay, 1);
            eng.push_back(qk_open(ggufPath(), &cfg, err, sizeof err));
            if (!eng.back()) { fprintf(stderr, "qk_open stage [%s] failed: %s\n", lay, err); return 1; }
            lo = hi;
        }
        int fd = -1;
        if (net) {
            std::string hp = net;
            size_t colon = hp.rfind(':');
            if (colon == std::string::npos) { fprintf(stderr, "pipe: bad host:port\n"); return 1; }
            std::string host = hp.substr(0, colon), port = hp.substr(colon + 1);
            addrinfo hints{}, *res = nullptr;
            hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
            if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) || !res) {
                fprintf(stderr, "pipe: cannot resolve %s\n", net); return 1;
            }
            fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
            if (fd < 0 || connect(fd, res->ai_addr, res->ai_addrlen)) { perror("connect"); return 1; }
            freeaddrinfo(res);
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            uint32_t magic = kPipeMagic, hello[7];
            if (!writeAll(fd, &magic, 4) || !readAll(fd, hello, sizeof hello) ||
                hello[0] != kPipeMagic) {
                fprintf(stderr, "pipe: worker hello failed (old/foreign build?)\n");
                return 1;
            }
            if (hello[1] != split || hello[2] != nLay || hello[3] != nLay || hello[4] != nEmbd ||
                hello[5] < 1 || hello[6] < tmax) {
                fprintf(stderr,
                        "pipe: worker mismatch: layers [%u,%u) of %u, n_embd %u, slots %u, ctx %u "
                        "(need [%u,%u) of %u, n_embd %u, ctx >= %u)\n",
                        hello[1], hello[2], hello[3], hello[4], hello[5], hello[6], split, nLay,
                        nLay, nEmbd, tmax);
                return 1;
            }
        }
        unsetenv("QK_LAYERS");
        // Run n positions from `base` through the whole chain; the final stage
        // (in-process last engine, or the remote worker) yields per-position ids.
        // Per-stage wall time accumulates into stMs.
        size_t nStages = eng.size() + (net ? 1 : 0);
        std::vector<double> stMs(nStages, 0.0);
        std::vector<float> hidA, hidB;
        auto runChain = [&](const uint32_t* toks, uint32_t n, uint32_t base,
                            uint32_t* ids) -> bool {
            hidA.resize((size_t)n * nEmbd);
            hidB.resize((size_t)n * nEmbd);
            const float* hin = nullptr;
            for (size_t s = 0; s < eng.size(); s++) {
                bool last = !net && s + 1 == eng.size();
                float* hout = last ? nullptr : (s % 2 ? hidB.data() : hidA.data());
                auto ta = std::chrono::steady_clock::now();
                if (eng[s]->stageRun(0, s == 0 ? toks : nullptr, hin, n, base, hout,
                                     last ? ids : nullptr)) {
                    fprintf(stderr, "pipe: stage %zu failed\n", s);
                    return false;
                }
                stMs[s] += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - ta).count();
                hin = hout;
            }
            if (net) {
                // QK_PIPE_DUMP=<dir>: raw dump of every hidden frame shipped to
                // the remote stage (port-bisect tooling; compare vs llama.cpp
                // eval-callback l_out-<lastLocalLayer> dumps).
                if (const char* df = getenv("QK_PIPE_DUMP")) {
                    static int fi = 0;
                    char pth[512];
                    snprintf(pth, sizeof pth, "%s/frame%d_n%u_base%u.bin", df, fi++, n, base);
                    if (FILE* dfp = fopen(pth, "wb")) {
                        fwrite(hin, 4, (size_t)n * nEmbd, dfp);
                        fclose(dfp);
                    }
                }
                auto ta = std::chrono::steady_clock::now();
                PipeHdr hd{1, 0, n, base};
                if (!(writeAll(fd, &hd, sizeof hd) && writeAll(fd, hin, (size_t)n * nEmbd * 4) &&
                      readAll(fd, ids, (size_t)n * 4))) {
                    fprintf(stderr, "pipe: remote stage failed\n");
                    return false;
                }
                stMs.back() += std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - ta).count();
            }
            return true;
        };
        std::vector<uint32_t> ids(np);
        auto t0 = std::chrono::steady_clock::now();
        // QK_PIPE_SERIAL=1: feed the prompt one token per frame (n=1, the
        // decode-shaped kernels) instead of one batched frame — port-bisect
        // tool to separate batch-only kernel bugs from per-token ones.
        if (getenv("QK_PIPE_SERIAL")) {
            for (uint32_t i = 0; i < np; i++)
                if (!runChain(&prompt[i], 1, i, ids.data() + i)) return 1;
        } else if (!runChain(prompt.data(), np, 0, ids.data())) {
            return 1;
        }
        double prefillMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0).count();
        std::fill(stMs.begin(), stMs.end(), 0.0);  // report decode-only per-stage times
        uint32_t next = ids[np - 1], pos = np;
        std::vector<uint32_t> gen;
        while (next != eng[0]->eosTok && (uint32_t)gen.size() < nGen) {
            gen.push_back(next);
            if ((uint32_t)gen.size() == nGen) break;
            uint32_t am;
            if (!runChain(&next, 1, pos, &am)) return 1;
            next = am;
            pos++;
        }
        uint32_t steps = pos - np;
        double totMs = 0;
        for (double m : stMs) totMs += m;
        printf("pipe: layers");
        lo = 0;
        for (size_t s = 0; s < nStages; s++) {
            uint32_t hi = s < bounds.size() ? bounds[s] : nLay;
            printf("%s[%u,%u)", s ? "+" : " ", lo, hi);
            lo = hi;
        }
        printf("%s | prompt %u prefill %.1f ms | %zu tokens, %.2f ms/tok (",
               net ? " over TCP" : " in-process", np, prefillMs, gen.size(),
               steps ? totMs / steps : 0.0);
        for (size_t s = 0; s < nStages; s++)
            printf("%ss%zu%s %.2f", s ? ", " : "", s + 1,
                   net && s + 1 == nStages ? "+net" : "", steps ? stMs[s] / steps : 0.0);
        printf(")\n");
        printf("GEN:");
        for (uint32_t t : gen) printf(" %u", t);
        printf("\n");
        if (fd >= 0) {
            PipeHdr bye{2, 0, 0, 0};
            writeAll(fd, &bye, sizeof bye);
            close(fd);
        }
        for (qk_engine* e : eng) qk_close(e);
        return 0;
    }

    if (mode == "verify") {
        // Spec-decode P0 harness: oracle-draft verify rounds vs the serial stream.
        // 1) Generate nGen tokens serially -> reference stream S (and decode ms/tok).
        // 2) Re-run from empty: batch-prefill the prompt, then advance in verify
        //    rounds that feed K tokens straight from S (an oracle draft = 100%
        //    acceptance). Every per-position argmax must reproduce S exactly —
        //    this is the full-accept spec-decode path end to end, and it measures
        //    the true c(K) verify-round cost incl. argmax + readback.
        if (argc < 4) {
            fprintf(stderr, "usage: qk verify <ids-file> <nGen> [K,K,...] [tmax]\n");
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
        std::vector<uint32_t> Ks;
        if (argc > 4) {
            for (const char* p = argv[4]; *p;) {
                Ks.push_back((uint32_t)strtoul(p, (char**)&p, 10));
                if (*p == ',') p++;
            }
        } else {
            Ks = {8, 16, 32};
        }
        uint32_t tmax = argc > 5 ? (uint32_t)atoi(argv[5]) : 4096;
        qk_config cfg{1, tmax, 8};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }

        // serial reference stream + decode-only ms/token (chunks after prefill ends)
        std::vector<uint32_t> S = prompt;
        double serialMsTok = 0;
        {
            qk_slot_start(e, 0, prompt.data(), (uint32_t)prompt.size(), nGen, 0);
            uint32_t ch = qk_chunk(e), fin = 0;
            std::vector<uint32_t> ot(ch), oc(1);
            double decodeMs = 0;
            uint32_t decodeTok = 0;
            bool prefilled = false;
            while (true) {
                auto t0 = std::chrono::steady_clock::now();
                int act = qk_step_chunk(e, ot.data(), oc.data(), &fin);
                double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count();
                if (act <= 0) break;
                for (uint32_t i = 0; i < oc[0]; i++) S.push_back(ot[i]);
                if (prefilled) { decodeMs += ms; decodeTok += oc[0]; }
                if (oc[0] > 0) prefilled = true;
                if (fin & 1u) break;
            }
            serialMsTok = decodeTok ? decodeMs / decodeTok : 0;
            printf("verify: prompt %zu, serial stream %zu gen tokens, serial decode %.2f ms/tok\n",
                   prompt.size(), S.size() - prompt.size(), serialMsTok);
        }
        uint32_t np = (uint32_t)prompt.size();
        if (S.size() < (size_t)np + 4) { fprintf(stderr, "stream too short (early EOS?)\n"); return 1; }

        bool allOk = true;
        printf("  %-4s %8s %10s %8s %10s   %s\n", "K", "rounds", "round_ms", "ms/tok", "vs_serial", "exact");
        for (uint32_t K : Ks) {
            if (K < 1 || K > e->maxB) { printf("  K=%u skipped (maxB=%u)\n", K, e->maxB); continue; }
            // fresh state: batch-prefill the prompt from empty in <=maxB chunks
            std::vector<float> dummy;
            for (uint32_t off = 0; off < np;) {
                uint32_t n = std::min(e->maxB, np - off);
                e->prefillBatchLast(prompt.data() + off, n, 0, dummy, false, off);
                off += n;
            }
            // oracle rounds: feed S[pos..pos+k) (entry 0 = last committed token);
            // argmax i must equal S[pos+i+1]
            uint32_t pos = np, mism = 0, rounds = 0, committed = 0;
            double roundMs = 0;
            std::vector<uint32_t> am(e->maxB);
            while (pos + 1 < (uint32_t)S.size()) {
                uint32_t k = std::min(K, (uint32_t)S.size() - pos - 1);
                auto t0 = std::chrono::steady_clock::now();
                e->prefillBatchLast(&S[pos], k, 0, dummy, false, pos, am.data());
                roundMs += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0).count();
                rounds++;
                for (uint32_t i = 0; i < k; i++)
                    if (am[i] != S[pos + i + 1]) mism++;
                committed += k;
                pos += k;
            }
            double msTok = committed ? roundMs / committed : 0;
            bool ok = mism == 0;
            allOk = allOk && ok;
            printf("  %-4u %8u %10.1f %8.2f %9.2fx   %s\n",
                   K, rounds, rounds ? roundMs / rounds : 0, msTok,
                   msTok > 0 ? serialMsTok / msTok : 0, ok ? "EXACT" : "**MISMATCH**");
            if (!ok) printf("      %u/%u positions diverged\n", mism, committed);
        }
        printf("verify: %s\n", allOk ? "ORACLE SPEC ROUNDS TOKEN-EXACT" : "DIVERGENCE (see above)");
        qk_close(e);
        return allOk ? 0 : 1;
    }

    if (mode == "prefillcmp") {
        uint32_t N = argc > 2 ? (uint32_t)atoi(argv[2]) : 32;
        uint32_t ctx = argc > 3 ? (uint32_t)atoi(argv[3]) : 2048;
        if (N < 1 || N > 1024 || N + 1 > ctx) {
            fprintf(stderr, "usage: qk prefillcmp [N<=maxB] [ctx]  (requires N+1 <= ctx)\n");
            return 1;
        }
        qk_config cfg{2, ctx, 8};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }

        // Sweep a matrix of chunk sizes x seeds in one process (model load dominates).
        // For each: serial reference logits (robust to EOS) vs batched-prefill logits;
        // require argmax(last-token) to match AND report the logit-vector max abs diff.
        uint32_t cap = std::min<uint32_t>(e->maxB, ctx - 1);
        std::vector<uint32_t> sizes;
        for (uint32_t s : {1u, 2u, 8u, 15u, 16u, 17u, 32u, 48u, 64u, 96u, 127u, 128u, 192u, 256u})
            if (s <= cap) sizes.push_back(s);
        if (N <= cap && std::find(sizes.begin(), sizes.end(), N) == sizes.end()) sizes.push_back(N);
        std::vector<uint32_t> seeds{1234u, 42u, 2026u};

        uint32_t nCase = 0, nPass = 0;
        double worstRel = 0;
        printf("prefillcmp sweep: sizes x seeds, ctx=%u  (tokS=serial argmax, tokB=batched argmax)\n", ctx);
        for (uint32_t sz : sizes) {
            for (uint32_t seed : seeds) {
                std::mt19937 rng(seed);
                std::vector<uint32_t> toks(sz);
                for (uint32_t i = 0; i < sz; i++) toks[i] = rng() % (e->vocab - 16) + 4;

                std::vector<float> logitsS, logitsB;
                uint32_t tokS = e->serialPrefillLogits(toks.data(), sz, 1, logitsS);
                e->prefillBatchLast(toks.data(), sz, 0, logitsB);
                uint32_t tokB = (uint32_t)(std::max_element(logitsB.begin(), logitsB.end()) - logitsB.begin());

                double maxAbs = 0, refMax = 1e-9;
                for (uint32_t i = 0; i < e->vocab; i++) {
                    maxAbs = std::max(maxAbs, (double)std::fabs(logitsB[i] - logitsS[i]));
                    refMax = std::max(refMax, (double)std::fabs(logitsS[i]));
                }
                double rel = maxAbs / refMax;
                worstRel = std::max(worstRel, rel);
                bool ok = (tokS == tokB);
                nCase++; if (ok) nPass++;
                printf("  N=%-4u seed=%-5u tokS=%-7u tokB=%-7u %s  max|dlogit|=%.4g rel=%.2g\n",
                       sz, seed, tokS, tokB, ok ? "MATCH" : "**MISMATCH**", maxAbs, rel);
            }
        }
        bool allOk = (nPass == nCase);
        printf("\nprefillcmp: %u/%u argmax matches, worst rel logit diff %.2g -> %s\n",
               nPass, nCase, worstRel, allOk ? "PREFILL TOKEN-EXACT" : "PREFILL DIVERGENCE");
        qk_close(e);
        return allOk ? 0 : 1;
    }

    if (mode == "prefillbench") {
        uint32_t ctx = argc > 2 ? (uint32_t)atoi(argv[2]) : 2048;
        qk_config cfg{2, ctx, 8};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
        uint32_t cap = std::min<uint32_t>(e->maxB, ctx - 1);
        std::vector<float> lS, lB;
        printf("prefillbench: serial (N per-token forwards) vs batched (1 forward), ctx=%u\n", ctx);
        printf("  %-6s %10s %10s %8s %10s\n", "N", "serial_ms", "batch_ms", "speedup", "tok/s_bat");
        for (uint32_t N : {8u, 16u, 32u, 64u, 96u, 128u, 192u, 256u}) {
            if (N > cap) continue;
            std::mt19937 rng(1234);
            std::vector<uint32_t> toks(N);
            for (uint32_t i = 0; i < N; i++) toks[i] = rng() % (e->vocab - 16) + 4;
            e->prefillBatchLast(toks.data(), N, 0, lB);   // warm
            e->serialPrefillLogits(toks.data(), N, 1, lS);
            auto t0 = std::chrono::steady_clock::now();
            e->serialPrefillLogits(toks.data(), N, 1, lS);
            auto t1 = std::chrono::steady_clock::now();
            e->prefillBatchLast(toks.data(), N, 0, lB);
            auto t2 = std::chrono::steady_clock::now();
            double sMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double bMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
            printf("  %-6u %10.2f %10.2f %7.2fx %10.0f\n", N, sMs, bMs, sMs / bMs, N / (bMs / 1000.0));
        }
        qk_close(e);
        return 0;
    }

    if (mode == "prefilldecode") {
        // End-to-end decode-handoff test: batched-prefill a prompt on slot 0, continue
        // GREEDY decode from the handed-off state, and require the full generated
        // sequence to match an all-serial run of the same prompt token-for-token.
        uint32_t N = argc > 2 ? (uint32_t)atoi(argv[2]) : 32;
        uint32_t M = argc > 3 ? (uint32_t)atoi(argv[3]) : 24;
        uint32_t ctx = argc > 4 ? (uint32_t)atoi(argv[4]) : 2048;
        if (N < 1 || N > 1024 || N + M > ctx) { fprintf(stderr, "usage: qk prefilldecode [N<=maxB] [M] [ctx]\n"); return 1; }
        qk_config cfg{2, ctx, 8};
        char err[256] = {0};
        qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
        if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
        if (N > e->maxB) { fprintf(stderr, "N=%u > maxB=%u (raise QK_MAXB)\n", N, e->maxB); return 1; }
        uint32_t ch = qk_chunk(e);
        uint32_t nSl = qk_n_slots(e);
        std::vector<uint32_t> outTok((size_t)nSl * ch), outCnt(nSl);
        uint32_t fin = 0;

        bool allOk = true;
        for (uint32_t seed : {1234u, 7u, 99u}) {
            std::mt19937 rng(seed);
            std::vector<uint32_t> toks(N);
            for (uint32_t i = 0; i < N; i++) toks[i] = rng() % (e->vocab - 16) + 4;

            // Reference: all-serial prefill+decode on slot 1.
            std::vector<uint32_t> refSeq;
            qk_slot_start(e, 1, toks.data(), N, M, 0);
            while (e->stepChunk(outTok.data(), outCnt.data(), &fin) > 0)
                for (uint32_t i = 0; i < outCnt[1]; i++) refSeq.push_back(outTok[(size_t)1 * ch + i]);

            // Batched prefill on slot 0, then continue serial decode from the handoff.
            std::vector<float> lB;
            e->prefillBatchLast(toks.data(), N, 0, lB);
            uint32_t tok0 = (uint32_t)(std::max_element(lB.begin(), lB.end()) - lB.begin());
            // Follow serial's convention: a generated EOS is a terminator, not emitted.
            std::vector<uint32_t> batSeq;
            qk_engine::Slot& s0 = e->slots[0];
            s0.prompt.assign(toks.begin(), toks.end());
            if (tok0 != e->eosTok) {
                batSeq.push_back(tok0);
                s0.genTokens.assign(1, tok0);
                s0.cursor = N; s0.pos = N; s0.gen = 1; s0.maxGen = M; s0.last = tok0; s0.active = true;
                while (e->stepChunk(outTok.data(), outCnt.data(), &fin) > 0)
                    for (uint32_t i = 0; i < outCnt[0]; i++) batSeq.push_back(outTok[(size_t)0 * ch + i]);
            } else {
                s0.active = false;  // immediate EOS: serial swallows it too -> both emit nothing
            }

            size_t cmp = std::min(refSeq.size(), batSeq.size());
            size_t match = 0;
            while (match < cmp && refSeq[match] == batSeq[match]) match++;
            bool ok = (refSeq.size() == batSeq.size()) && (match == refSeq.size());
            allOk &= ok;
            printf("  seed=%-5u N=%u M=%u serial=%zu tok, batched=%zu tok, matched %zu -> %s\n",
                   seed, N, M, refSeq.size(), batSeq.size(), match, ok ? "OK" : "**DIVERGE**");
            if (!ok) {
                printf("    serial :"); for (uint32_t t : refSeq) printf(" %u", t); printf("\n");
                printf("    batched:"); for (uint32_t t : batSeq) printf(" %u", t); printf("\n");
            }
        }
        printf("prefilldecode: %s\n", allOk ? "HANDOFF EXACT (batched prefill -> serial decode == all-serial)"
                                            : "HANDOFF DIVERGENCE");
        qk_close(e);
        return allOk ? 0 : 1;
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
        ok = caseBGemm(c, argU(2, 8192), argU(3, 2048), argU(4, 256), argU(5, 40));  // batched-prefill GEMM
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
