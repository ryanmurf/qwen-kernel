// qk — Qwen kernel harness, Metal host (Apple Silicon port of main.cpp).
// M1: fp16 GEMV. M2: quantized GEMV (Q8_0, Q6_K, IQ4_XS, IQ3_XXS) on raw
// ggml blocks, validated against CPU dequant reference and real tensors
// from the GGUF. (M3: fused blocks, M4: token loop, M5: qk.h engine.)
//
// Usage:
//   qk                        synthetic suite: f16, q8_0, q6_k, iq4_xs, iq3_xxs
//   qk f16|q8_0|q6_k|iq4_xs|iq3_xxs [M] [K] [iters]
//   qk gguf <tensor> [iters]  real weights (QK_GGUF overrides model path)
//   qk list [filter]          list tensors in the GGUF
//
// Env: QK_DEVICE=<n> device index; QK_SHADER_DIR (dir with *.metal);
//      QK_GGUF=<path>.
//
// MSL is compiled at runtime from shaders/metal/*.metal (no offline metal
// toolchain needed — mirrors the Vulkan loadSpv flow, QK_SHADER_DIR wins).
// Buffers are storageModeShared: on UMA upload/readback are plain memcpy;
// there is no staging path to emulate.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "gguf.h"
#include "quants.h"

static const char* kDefaultGguf =
    "models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf";  // set QK_GGUF; this is a last resort

// ---------- Metal context ----------

struct MtlCtx {
    id<MTLDevice>       dev = nil;
    id<MTLCommandQueue> queue = nil;
    std::map<std::string, id<MTLLibrary>> libs;                  // file -> lib
    std::map<std::string, id<MTLComputePipelineState>> pipes;    // file:fn:tpr -> pso
    const char* argv0 = nullptr;
};

static void initMtl(MtlCtx& c, const char* argv0) {
    c.argv0 = argv0;
    if (const char* e = getenv("QK_DEVICE")) {
        NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
        int i = atoi(e);
        if (i >= 0 && i < (int)all.count) c.dev = all[i];
    }
    if (!c.dev) c.dev = MTLCreateSystemDefaultDevice();
    if (!c.dev) {
        fprintf(stderr, "no Metal device\n");
        exit(1);
    }
    c.queue = [c.dev newCommandQueue];
    printf("device: %s (unified=%d, maxWorkingSet=%.1f GiB)\n",
           c.dev.name.UTF8String, (int)c.dev.hasUnifiedMemory,
           (double)c.dev.recommendedMaxWorkingSetSize / (1ull << 30));
}

// Mirrors loadSpv: QK_SHADER_DIR, then <exe-dir>/shaders/metal, then CWD.
static std::string loadMetalSource(const char* argv0, const char* name) {
    std::string path = std::string("shaders/metal/") + name;
    if (const char* d = getenv("QK_SHADER_DIR")) {
        path = std::string(d) + "/" + name;
    } else {
        std::string exe(argv0);
        auto slash = exe.rfind('/');
        if (slash != std::string::npos) {
            std::string cand = exe.substr(0, slash) + "/shaders/metal/" + name;
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
    std::string src((size_t)n, '\0');
    if (fread(&src[0], 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "short read on %s\n", path.c_str());
        exit(1);
    }
    fclose(f);

    // resolve #include "file" against the shader search path (mirrors the
    // GLSL flow where glslc handles iq_tables.glsl; <...> includes are MSL
    // stdlib and stay for the runtime compiler)
    size_t pos = 0;
    while ((pos = src.find("#include \"", pos)) != std::string::npos) {
        size_t q1 = pos + 10, q2 = src.find('"', q1);
        if (q2 == std::string::npos) break;
        std::string inc = src.substr(q1, q2 - q1);
        src = src.substr(0, pos) + loadMetalSource(argv0, inc.c_str()) +
              src.substr(q2 + 1);
    }
    return src;
}

// Pipeline cache: JIT-compile <file>.metal once, specialize per (fn, TPR).
static id<MTLComputePipelineState> getPipe(MtlCtx& c, const char* file,
                                           const char* fn, uint32_t tpr) {
    std::string key = std::string(file) + ":" + fn + ":" + std::to_string(tpr);
    auto it = c.pipes.find(key);
    if (it != c.pipes.end()) return it->second;

    id<MTLLibrary> lib = nil;
    auto lit = c.libs.find(file);
    if (lit != c.libs.end()) {
        lib = lit->second;
    } else {
        std::string src = loadMetalSource(c.argv0, (std::string(file) + ".metal").c_str());
        NSError* err = nil;
        lib = [c.dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                  options:nil
                                    error:&err];
        if (!lib) {
            fprintf(stderr, "MSL compile %s: %s\n", file,
                    err.localizedDescription.UTF8String);
            exit(1);
        }
        c.libs[file] = lib;
    }

    // tpr == 0: kernel has no function constants (mirrors makePipe's if (tpr))
    NSError* err = nil;
    id<MTLFunction> f = nil;
    if (tpr) {
        MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
        [fc setConstantValue:&tpr type:MTLDataTypeUInt atIndex:0];
        f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]
                      constantValues:fc
                               error:&err];
    } else {
        f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
    }
    if (!f) {
        fprintf(stderr, "function %s: %s\n", fn,
                err ? err.localizedDescription.UTF8String : "not found");
        exit(1);
    }
    id<MTLComputePipelineState> pso = [c.dev newComputePipelineStateWithFunction:f error:&err];
    if (!pso) {
        fprintf(stderr, "pipeline %s: %s\n", fn, err.localizedDescription.UTF8String);
        exit(1);
    }
    c.pipes[key] = pso;
    return pso;
}

// UMA buffer: shared storage, host pointer == GPU memory. No staging.
// untracked=true opts out of automatic hazard tracking — the caller owns
// ordering via explicit memoryBarrier (the Vulkan model; cuts driver
// bookkeeping in multi-dispatch chains).
static id<MTLBuffer> createBuf(MtlCtx& c, size_t size, const void* init = nullptr,
                               bool untracked = false) {
    MTLResourceOptions opt = MTLResourceStorageModeShared;
    if (untracked) opt |= MTLResourceHazardTrackingModeUntracked;
    id<MTLBuffer> b = init
        ? [c.dev newBufferWithBytes:init length:size options:opt]
        : [c.dev newBufferWithLength:size options:opt];
    if (!b) {
        fprintf(stderr, "buffer alloc failed (%zu MiB)\n", size >> 20);
        exit(1);
    }
    return b;
}

// ---------- generic GEMV run: upload, verify, benchmark ----------

// fixedNr0 != 0: llama.cpp-style kernel owning its work shape — NSG
// simdgroups per threadgroup (function constant 0, QK_TPR overrides for
// sweeps), fixedNr0 consecutive rows per simdgroup. tgMemBytes: threadgroup
// memory for staged tables (index 0).
static bool runGemv(MtlCtx& c, const char* kernelName, const void* wBytes,
                    size_t wSize, const std::vector<float>& x, uint32_t M,
                    uint32_t K, const std::vector<float>& yref, uint32_t iters,
                    uint32_t unitsPerRow, double tol = 1e-2,
                    uint32_t fixedNr0 = 0, uint32_t tgMemBytes = 0) {
    size_t sizeX = (size_t)K * 4, sizeY = (size_t)M * 4;

    // threads-per-row function constant: shrink for skinny rows so a
    // workgroup covers 256/TPR rows and stays fully occupied. Beyond the
    // Vulkan rule, Apple GPUs also want ≥4 units of work per thread (a lone
    // 16 B load per thread leaves ~40% of bandwidth on the table — measured
    // 315 vs 523 GB/s on the 248320×2048 head shape), as long as enough
    // threadgroups remain to occupy the 40 cores.
    uint32_t tpr = 256;
    while (tpr > 4 && tpr / 2 >= unitsPerRow) tpr /= 2;
    while (tpr > 8 && unitsPerRow / tpr < 4 &&
           (uint64_t)M * (tpr / 2) / 256 >= 1024) tpr /= 2;
    if (const char* e = getenv("QK_TPR")) {  // crossover experiments (M6)
        uint32_t v = (uint32_t)atoi(e);
        if (v >= 4 && v <= 256 && (v & (v - 1)) == 0) tpr = v;
    }
    uint32_t nsg = 2;  // simdgroups per tg for fixed-shape kernels (llama.cpp ships 2)
    if (fixedNr0) {
        if (const char* e = getenv("QK_TPR")) {
            uint32_t v = (uint32_t)atoi(e);
            if (v >= 1 && v <= 8) nsg = v;
        }
        tpr = nsg;  // function-constant slot carries NSG for these kernels
    }
    uint32_t rowsPerWg = fixedNr0 ? nsg * fixedNr0 : 256 / tpr;
    uint32_t tgThreads = fixedNr0 ? nsg * 32 : 256;

    id<MTLBuffer> bW = createBuf(c, wSize, wBytes);
    id<MTLBuffer> bX = createBuf(c, sizeX, x.data());
    id<MTLBuffer> bY = createBuf(c, sizeY);

    id<MTLComputePipelineState> pso = getPipe(c, kernelName, kernelName, tpr);

    uint32_t wgs = (M + rowsPerWg - 1) / rowsPerWg;
    struct { uint32_t M, K; } pc{M, K};

    auto encodeDispatch = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:bY offset:0 atIndex:2];
        [enc setBytes:&pc length:8 atIndex:3];
        if (tgMemBytes) [enc setThreadgroupMemoryLength:tgMemBytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgThreads, 1, 1)];
    };

    // correctness pass
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encodeDispatch(enc);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    std::vector<float> ygpu(M);
    memcpy(ygpu.data(), bY.contents, sizeY);

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

    if (pass && iters > 0) {
        // one command buffer, `iters` dispatches in one encoder; the Y
        // write-after-write hazard serializes them (the Vulkan barrier
        // equivalent). GPU timestamps from the command buffer bracket.
        auto runBench = [&]() -> double {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (uint32_t i = 0; i < iters; i++) encodeDispatch(enc);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / iters;  // ns/iter
            }
        };
        runBench();  // warm-up
        double ns = runBench();
        double bytes = (double)wSize + sizeX + sizeY;
        double flops = 2.0 * M * K;
        char geo[32];
        if (fixedNr0) snprintf(geo, sizeof geo, "nsg %u x %u rows", nsg, fixedNr0);
        else          snprintf(geo, sizeof geo, "tpr %u", tpr);
        printf("gpu: %8.1f µs/iter | %7.1f GB/s | %8.1f GFLOP/s | %.1f MiB/iter (UMA, %s)\n",
               ns / 1e3, bytes / ns, flops / ns, bytes / (1 << 20), geo);
    }
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

static bool caseF16(MtlCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
    printf("\n== f16 GEMV  M=%u K=%u (W %.1f MiB) ==\n", M, K, (double)M * K * 2 / (1 << 20));
    if (K % 8) {
        fprintf(stderr, "K must be a multiple of 8\n");
        return false;
    }
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
    return runGemv(c, "gemv_f16", Wh.data(), Wh.size() * 2, x, M, K, yref, iters, K / 8);
}

static bool caseQ80(MtlCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
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
    return runGemv(c, "gemv_q8_0", blocks.data(), nb * sizeof(block_q8_0),
                   x, M, K, yref, iters, K / 32);
}

static bool caseQ6K(MtlCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
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
    return runGemv(c, "gemv_q6_k", blocks.data(), nb * sizeof(block_q6_K),
                   x, M, K, yref, iters, K / 16, 1e-2, 2, 0);
}

static bool caseIQ4XS(MtlCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
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
    return runGemv(c, "gemv_iq4_xs", blocks.data(), nb * sizeof(block_iq4_xs),
                   x, M, K, yref, iters, K / 32, 1e-2, 2, 128);
}

static bool caseIQ3XXS(MtlCtx& c, uint32_t M, uint32_t K, uint32_t iters) {
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
    return runGemv(c, "gemv_iq3_xxs", blocks.data(), nb * sizeof(block_iq3_xxs),
                   x, M, K, yref, iters, K / 32, 1e-2, 4, 1152);
}

// ---------- real weights from the GGUF ----------

static const char* ggufPath() {
    const char* p = getenv("QK_GGUF");
    return p ? p : kDefaultGguf;
}

static bool caseGguf(MtlCtx& c, const std::string& tensorName, uint32_t iters) {
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
        fprintf(stderr, "type %s not supported for GEMV\n", ggmlTypeName(t->type));
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

    const char* kern = nullptr;
    uint32_t units = K, fixedNr0v = 0, tgMem = 0;
    switch (t->type) {
        case GGML_Q8_0:    kern = "gemv_q8_0";    units = K / 32; break;
        case GGML_Q6_K:    kern = "gemv_q6_k";    units = K / 16; fixedNr0v = 2; break;
        case GGML_IQ4_XS:  kern = "gemv_iq4_xs";  units = K / 32; fixedNr0v = 2; tgMem = 128; break;
        case GGML_IQ3_XXS: kern = "gemv_iq3_xxs"; units = K / 32; fixedNr0v = 4; tgMem = 1152; break;
        case GGML_F16:     kern = "gemv_f16";     units = K / 8;  break;
        default:
            fprintf(stderr, "no Metal kernel for %s\n", ggmlTypeName(t->type));
            return false;
    }
    return runGemv(c, kern, t->data, (size_t)M * rowBytes, x, M, K, yref, iters,
                   units, 1e-2, fixedNr0v, tgMem);
}

// ---------- fused MoE decode step (M3) ----------
// Six dispatches, ONE command buffer per iteration; Metal hazard tracking
// reproduces the Vulkan barrier ordering ({logits ∥ shared-gateup} ->
// select -> routed-gateup -> routed-down -> shared-down).

static bool caseMoe(MtlCtx& c, uint32_t layer, uint32_t iters) {
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
    // one simdgroup per output row everywhere; NSG simdgroups per threadgroup.
    // Four dispatches / three barriers: each barrier drains the GPU for
    // ~8 µs, so shared-expert work rides inside the routed dispatches.
    const uint32_t nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;
    id<MTLComputePipelineState> pLogits = getPipe(c, "moe_logits", "moe_logits", nsg);
    id<MTLComputePipelineState> pSelect = getPipe(c, "moe_select", "moe_select", 0);
    id<MTLComputePipelineState> pGu     = getPipe(c, "moe_gateup_all", "moe_gateup_all", nsg);
    id<MTLComputePipelineState> pDn     = downQ6 ? getPipe(c, "moe_down_all", "moe_down_all_q6k", nsg)
                                                 : getPipe(c, "moe_down_all", "moe_down_all_iq4", nsg);

    const size_t szGI = (size_t)nExp * nEmbd * 4, szGIS = (size_t)nEmbd * 4;
    const size_t szGE = (size_t)nExp * nFf * rbGE, szDE = (size_t)nExp * nEmbd * rbDE;
    const size_t szGS = (size_t)nFf * rbGS, szDS = (size_t)nEmbd * rbDS;
    const size_t szX = (size_t)nEmbd * 4, szY = (size_t)nEmbd * 4;
    const size_t szH = (size_t)(nUsed + 1) * nFf * 4, szSel = 128, szL = (size_t)(nExp + 1) * 4;

    id<MTLBuffer> bGI = createBuf(c, szGI, tGI->data, true), bGIS = createBuf(c, szGIS, tGIS->data, true);
    id<MTLBuffer> bGE = createBuf(c, szGE, tGE->data, true), bUE = createBuf(c, szGE, tUE->data, true);
    id<MTLBuffer> bDE = createBuf(c, szDE, tDE->data, true);
    id<MTLBuffer> bGS = createBuf(c, szGS, tGS->data, true), bUS = createBuf(c, szGS, tUS->data, true);
    id<MTLBuffer> bDS = createBuf(c, szDS, tDS->data, true);
    id<MTLBuffer> bX = createBuf(c, szX, x.data(), true);
    id<MTLBuffer> bL = createBuf(c, szL, nullptr, true), bH = createBuf(c, szH, nullptr, true);
    id<MTLBuffer> bSel = createBuf(c, szSel, nullptr, true), bY = createBuf(c, szY, nullptr, true);

    struct { uint32_t nEmbd, nFf, nExp, nUsed; } pcv{nEmbd, nFf, nExp, nUsed};
    auto dispatchP = [&](id<MTLComputeCommandEncoder> enc,
                         id<MTLComputePipelineState> pso,
                         std::initializer_list<id<MTLBuffer>> bufs,
                         uint32_t nOut, uint32_t outPerTg) {
        [enc setComputePipelineState:pso];
        uint32_t i = 0;
        for (id<MTLBuffer> b : bufs) [enc setBuffer:b offset:0 atIndex:i++];
        [enc setBytes:&pcv length:16 atIndex:i];
        uint32_t wgs = (nOut + outPerTg - 1) / outPerTg;
        uint32_t thr = outPerTg == 1 ? 32 : nsg * 32;
        [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
    };
    // concurrent dispatch type: dispatches overlap unless explicitly
    // barriered — the exact Vulkan semantics ({logits ∥ shared-gateup} ->
    // barrier -> select -> ...). The default serial encoder drains the GPU
    // between every dispatch, which costs ~17 µs per stage on this chain.
    auto barrier = [&](id<MTLComputeCommandEncoder> enc) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    const int only = getenv("QK_MOE_ONLY") ? atoi(getenv("QK_MOE_ONLY")) : -1;
    auto sequence = [&](id<MTLComputeCommandEncoder> enc) {
        if (only >= 0) {  // stage isolation for profiling
            switch (only) {
                case 0: dispatchP(enc, pLogits, {bGI, bGIS, bX, bL}, nExp + 1, nsg); break;
                case 1: dispatchP(enc, pSelect, {bL, bSel}, 1, 1); break;
                case 2: dispatchP(enc, pGu, {bGE, bUE, bGS, bUS, bX, bSel, bH},
                                  (nUsed + 1) * nFf, nsg); break;
                case 3: dispatchP(enc, pDn, {bDE, bDS, bH, bSel, bY}, nEmbd, nsg); break;
            }
            return;
        }
        dispatchP(enc, pLogits, {bGI, bGIS, bX, bL}, nExp + 1, nsg);
        barrier(enc);
        dispatchP(enc, pSelect, {bL, bSel}, 1, 1);
        barrier(enc);
        dispatchP(enc, pGu, {bGE, bUE, bGS, bUS, bX, bSel, bH}, (nUsed + 1) * nFf, nsg);
        barrier(enc);
        dispatchP(enc, pDn, {bDE, bDS, bH, bSel, bY}, nEmbd, nsg);
    };

    // ---- correctness ----
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        sequence(enc);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    std::vector<float> ygpu(nEmbd);
    memcpy(ygpu.data(), bY.contents, szY);
    struct SelOut { uint32_t ids[8]; float w[8]; float wShared; } selGpu;
    memcpy(&selGpu, bSel.contents, sizeof(selGpu));

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
    if ((pass || only >= 0) && iters > 0) {
        auto runBench = [&]() -> double {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc =
                    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                for (uint32_t i = 0; i < iters; i++) {
                    sequence(enc);
                    barrier(enc);
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / iters;
            }
        };
        runBench();
        double ns = runBench();
        // weights actually touched per token (selected experts only)
        double bytes = (double)szGI + szGIS +
                       (double)nUsed * (2.0 * nFf * rbGE + (double)nEmbd * rbDE) +
                       2.0 * (double)nFf * rbGS + (double)nEmbd * rbDS;
        printf("gpu: %8.1f µs/layer-moe | %6.1f GB/s (active weights %.1f MiB) | 4 dispatches, 1 submit\n",
               ns / 1e3, bytes / ns, bytes / (1 << 20));
        printf("     40 layers -> %.2f ms/token MoE-FFN share\n", ns * 40 / 1e6);
    }
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

// ---------- gated-DeltaNet decode block (M3) ----------
// attn_norm -> qkv/z/alpha/beta -> conv+silu -> l2norm -> delta rule ->
// gated norm -> ssm_out -> residual/post-norm -> MoE-FFN -> residual, as
// ONE command buffer of 15 dispatches. Conv and delta states persist on-GPU
// across tokens, mirrored on CPU for validation.

static bool caseBlock(MtlCtx& c, uint32_t layer, uint32_t nTok, uint32_t iters) {
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
    const uint32_t nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;
    id<MTLComputePipelineState> pRms   = getPipe(c, "rmsnorm", "rmsnorm", 0);
    id<MTLComputePipelineState> pGemvA = getPipe(c, "gemv_q8_0", "gemv_q8_0", 64);   // K = n_embd
    id<MTLComputePipelineState> pAb    = getPipe(c, "dn_ab", "dn_ab", nsg);
    id<MTLComputePipelineState> pConv  = getPipe(c, "dn_conv", "dn_conv", 0);
    id<MTLComputePipelineState> pStep  = getPipe(c, "dn_step", "dn_step", 0);
    id<MTLComputePipelineState> pGate  = getPipe(c, "dn_gate", "dn_gate", nsg);
    id<MTLComputePipelineState> pGemvO = getPipe(c, "gemv_q8_0", "gemv_q8_0", 128);  // K = d_inner
    id<MTLComputePipelineState> pAddN  = getPipe(c, "add_rmsnorm", "add_rmsnorm", 0);
    id<MTLComputePipelineState> pAdd   = getPipe(c, "vec_add", "vec_add", 0);
    id<MTLComputePipelineState> pMoeL  = getPipe(c, "moe_logits", "moe_logits", nsg);
    id<MTLComputePipelineState> pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    id<MTLComputePipelineState> pMoeGu = getPipe(c, "moe_gateup_all", "moe_gateup_all", nsg);
    id<MTLComputePipelineState> pMoeDn = moe.downQ6
        ? getPipe(c, "moe_down_all", "moe_down_all_q6k", nsg)
        : getPipe(c, "moe_down_all", "moe_down_all_iq4", nsg);

    // ---- buffers (untracked; explicit barriers order the chain) ----
    const size_t szMGE = (size_t)moe.nExp * moe.nFf * moe.rbGE;
    const size_t szMDE = (size_t)moe.nExp * moe.nEmbd * moe.rbDE;
    id<MTLBuffer> bANorm = createBuf(c, nEmbd * 4, tANorm->data, true);
    id<MTLBuffer> bQkvW = createBuf(c, (size_t)chQkv * rbQ8e, tQkvW->data, true);
    id<MTLBuffer> bZW = createBuf(c, (size_t)dIn * rbQ8e, tZW->data, true);
    id<MTLBuffer> bAlW = createBuf(c, (size_t)hV * nEmbd * 4, tAl->data, true);
    id<MTLBuffer> bBeW = createBuf(c, (size_t)hV * nEmbd * 4, tBe->data, true);
    id<MTLBuffer> bDt = createBuf(c, hV * 4, tDt->data, true);
    id<MTLBuffer> bAv = createBuf(c, hV * 4, tAv->data, true);
    id<MTLBuffer> bKer = createBuf(c, (size_t)chQkv * 4 * 4, tKer->data, true);
    id<MTLBuffer> bSN = createBuf(c, dS * 4, tSN->data, true);
    id<MTLBuffer> bOutW = createBuf(c, (size_t)nEmbd * rbQ8i, tOutW->data, true);
    id<MTLBuffer> bPN = createBuf(c, nEmbd * 4, tPN->data, true);
    id<MTLBuffer> bMGI = createBuf(c, (size_t)moe.nExp * nEmbd * 4, moe.gi->data, true);
    id<MTLBuffer> bMGIS = createBuf(c, nEmbd * 4, moe.gis->data, true);
    id<MTLBuffer> bMGE = createBuf(c, szMGE, moe.ge->data, true);
    id<MTLBuffer> bMUE = createBuf(c, szMGE, moe.ue->data, true);
    id<MTLBuffer> bMDE = createBuf(c, szMDE, moe.de->data, true);
    id<MTLBuffer> bMGS = createBuf(c, (size_t)moe.nFf * moe.rbGS, moe.gs->data, true);
    id<MTLBuffer> bMUS = createBuf(c, (size_t)moe.nFf * moe.rbGS, moe.us->data, true);
    id<MTLBuffer> bMDS = createBuf(c, (size_t)moe.nEmbd * moe.rbDS, moe.ds->data, true);

    id<MTLBuffer> bXin = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bXn = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bQkv = createBuf(c, chQkv * 4, nullptr, true);
    id<MTLBuffer> bZ = createBuf(c, dIn * 4, nullptr, true);
    id<MTLBuffer> bGb = createBuf(c, 2 * hV * 4, nullptr, true);
    id<MTLBuffer> bConvSt = createBuf(c, (size_t)chQkv * 3 * 4, nullptr, true);
    id<MTLBuffer> bConvOut = createBuf(c, chQkv * 4, nullptr, true);
    id<MTLBuffer> bS = createBuf(c, (size_t)hV * dS * dS * 4, nullptr, true);
    id<MTLBuffer> bO = createBuf(c, dIn * 4, nullptr, true);
    id<MTLBuffer> bAtt = createBuf(c, dIn * 4, nullptr, true);
    id<MTLBuffer> bAttnOut = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bY = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bXn2 = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bML = createBuf(c, (moe.nExp + 1) * 4, nullptr, true);
    id<MTLBuffer> bMH = createBuf(c, (size_t)(moe.nUsed + 1) * moe.nFf * 4, nullptr, true);
    id<MTLBuffer> bMSel = createBuf(c, 128, nullptr, true);
    id<MTLBuffer> bMY = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bOut = createBuf(c, nEmbd * 4, nullptr, true);
    memset(bConvSt.contents, 0, (size_t)chQkv * 3 * 4);
    memset(bS.contents, 0, (size_t)hV * dS * dS * 4);

    struct { uint32_t n; float e; } pcRms{nEmbd, eps};
    struct { uint32_t m, k; } pcQkv{chQkv, nEmbd}, pcZ{dIn, nEmbd}, pcOut{nEmbd, dIn};
    struct { uint32_t n, h; } pcAb{nEmbd, hV};
    struct { uint32_t ch; } pcConv{chQkv};
    struct { uint32_t d, hk, hv; float e; } pcStep{dS, hK, hV, eps};
    struct { uint32_t d, hv; float e; } pcGate{dS, hV, eps};
    struct { uint32_t n; } pcAdd{nEmbd};
    struct { uint32_t a, b, cc, d; } pcv{moe.nEmbd, moe.nFf, moe.nExp, moe.nUsed};

    auto dsp = [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                   std::initializer_list<id<MTLBuffer>> bufs,
                   const void* pc, uint32_t pcSize, uint32_t wgs, uint32_t thr) {
        [enc setComputePipelineState:pso];
        uint32_t i = 0;
        for (id<MTLBuffer> b : bufs) [enc setBuffer:b offset:0 atIndex:i++];
        [enc setBytes:pc length:pcSize atIndex:i];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
    };
    auto bar = [&](id<MTLComputeCommandEncoder> enc) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    const uint32_t thrN = nsg * 32;

    auto sequence = [&](id<MTLComputeCommandEncoder> enc) {
        dsp(enc, pRms, {bXin, bANorm, bXn}, &pcRms, 8, 1, 256);
        bar(enc);
        dsp(enc, pGemvA, {bQkvW, bXn, bQkv}, &pcQkv, 8, chQkv / 4, 256);
        dsp(enc, pGemvA, {bZW, bXn, bZ}, &pcZ, 8, dIn / 4, 256);
        dsp(enc, pAb, {bXn, bAlW, bBeW, bDt, bAv, bGb}, &pcAb, 8, (2 * hV + nsg - 1) / nsg, thrN);
        bar(enc);
        dsp(enc, pConv, {bConvSt, bQkv, bKer, bConvOut}, &pcConv, 4, (chQkv + 255) / 256, 256);
        bar(enc);
        dsp(enc, pStep, {bConvOut, bGb, bS, bO}, &pcStep, 16, hV, 128);
        bar(enc);
        dsp(enc, pGate, {bO, bSN, bZ, bAtt}, &pcGate, 12, (hV + nsg - 1) / nsg, thrN);
        bar(enc);
        dsp(enc, pGemvO, {bOutW, bAtt, bAttnOut}, &pcOut, 8, nEmbd / 2, 256);
        bar(enc);
        dsp(enc, pAddN, {bXin, bAttnOut, bPN, bY, bXn2}, &pcRms, 8, 1, 256);
        bar(enc);
        dsp(enc, pMoeL, {bMGI, bMGIS, bXn2, bML}, &pcv, 16, (moe.nExp + 1 + nsg - 1) / nsg, thrN);
        bar(enc);
        dsp(enc, pMoeS, {bML, bMSel}, &pcv, 16, 1, 32);
        bar(enc);
        dsp(enc, pMoeGu, {bMGE, bMUE, bMGS, bMUS, bXn2, bMSel, bMH}, &pcv, 16,
            ((moe.nUsed + 1) * moe.nFf + nsg - 1) / nsg, thrN);
        bar(enc);
        dsp(enc, pMoeDn, {bMDE, bMDS, bMH, bMSel, bMY}, &pcv, 16, (nEmbd + nsg - 1) / nsg, thrN);
        bar(enc);
        dsp(enc, pAdd, {bY, bMY, bOut}, &pcAdd, 4, 1, 256);
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
        memcpy(bXin.contents, x.data(), nEmbd * 4);
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc =
                [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            sequence(enc);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        memcpy(gpuOut.data(), bOut.contents, nEmbd * 4);

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
        const float* sg = (const float*)bS.contents;
        double maxAbs = 0, rmsS = 0;
        for (size_t i = 0; i < S.size(); i++) {
            maxAbs = std::max(maxAbs, (double)std::fabs(sg[i] - S[i]));
            rmsS += (double)S[i] * S[i];
        }
        rmsS = std::sqrt(rmsS / S.size());
        printf("delta state after %u tokens: max_abs_diff = %.3g (state rms %.3g)\n", nTok, maxAbs, rmsS);
    }

    if (pass && iters > 0) {
        auto runBench = [&]() -> double {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc =
                    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                for (uint32_t i = 0; i < iters; i++) {
                    sequence(enc);
                    bar(enc);
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / iters;
            }
        };
        runBench();
        double ns = runBench();
        printf("gpu: %8.1f µs/block | 14 dispatches, 1 submit\n", ns / 1e3);
        printf("     30 deltanet blocks -> %.2f ms/token\n", ns * 30 / 1e6);
    }
    return pass;
}


// ---------- full-attention decode block (M3) ----------
// attn_norm -> q(+gate)/k/v projections -> per-head RMS + partial NeoX rope
// -> KV-cache attention + sigmoid gate -> wo -> residual/post-norm -> MoE ->
// residual. KV cache persists on GPU.
static const float kFreqBase = 1e7f;  // qwen35moe.rope.freq_base

static bool caseABlock(MtlCtx& c, uint32_t layer, uint32_t nTok, uint32_t iters) {
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

    const uint32_t nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;
    id<MTLComputePipelineState> pRms   = getPipe(c, "rmsnorm", "rmsnorm", 0);
    id<MTLComputePipelineState> pGemvA = getPipe(c, "gemv_q8_0", "gemv_q8_0", 64);
    id<MTLComputePipelineState> pPrep  = getPipe(c, "fa_prep", "fa_prep", 0);
    id<MTLComputePipelineState> pAttn  = getPipe(c, "fa_attn", "fa_attn", 0);
    id<MTLComputePipelineState> pGemvO = getPipe(c, "gemv_q8_0", "gemv_q8_0", 128);
    id<MTLComputePipelineState> pAddN  = getPipe(c, "add_rmsnorm", "add_rmsnorm", 0);
    id<MTLComputePipelineState> pAdd   = getPipe(c, "vec_add", "vec_add", 0);
    id<MTLComputePipelineState> pMoeL  = getPipe(c, "moe_logits", "moe_logits", nsg);
    id<MTLComputePipelineState> pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    id<MTLComputePipelineState> pMoeGu = getPipe(c, "moe_gateup_all", "moe_gateup_all", nsg);
    id<MTLComputePipelineState> pMoeDn = moe.downQ6
        ? getPipe(c, "moe_down_all", "moe_down_all_q6k", nsg)
        : getPipe(c, "moe_down_all", "moe_down_all_iq4", nsg);

    const size_t szMGE = (size_t)moe.nExp * moe.nFf * moe.rbGE;
    const size_t szMDE = (size_t)moe.nExp * moe.nEmbd * moe.rbDE;
    id<MTLBuffer> bANorm = createBuf(c, nEmbd * 4, tANorm->data, true);
    id<MTLBuffer> bWq = createBuf(c, (size_t)qfN * rbQ8e, tWq->data, true);
    id<MTLBuffer> bWk = createBuf(c, (size_t)kvN * rbQ8e, tWk->data, true);
    id<MTLBuffer> bWv = createBuf(c, (size_t)kvN * rbQ8e, tWv->data, true);
    id<MTLBuffer> bQN = createBuf(c, dh * 4, tQN->data, true);
    id<MTLBuffer> bKN = createBuf(c, dh * 4, tKN->data, true);
    id<MTLBuffer> bWo = createBuf(c, (size_t)nEmbd * rbQ8a, tWo->data, true);
    id<MTLBuffer> bPN = createBuf(c, nEmbd * 4, tPN->data, true);
    id<MTLBuffer> bMGI = createBuf(c, (size_t)moe.nExp * nEmbd * 4, moe.gi->data, true);
    id<MTLBuffer> bMGIS = createBuf(c, nEmbd * 4, moe.gis->data, true);
    id<MTLBuffer> bMGE = createBuf(c, szMGE, moe.ge->data, true);
    id<MTLBuffer> bMUE = createBuf(c, szMGE, moe.ue->data, true);
    id<MTLBuffer> bMDE = createBuf(c, szMDE, moe.de->data, true);
    id<MTLBuffer> bMGS = createBuf(c, (size_t)moe.nFf * moe.rbGS, moe.gs->data, true);
    id<MTLBuffer> bMUS = createBuf(c, (size_t)moe.nFf * moe.rbGS, moe.us->data, true);
    id<MTLBuffer> bMDS = createBuf(c, (size_t)moe.nEmbd * moe.rbDS, moe.ds->data, true);

    id<MTLBuffer> bXin = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bXn = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bQfull = createBuf(c, qfN * 4, nullptr, true);
    id<MTLBuffer> bKin = createBuf(c, kvN * 4, nullptr, true);
    id<MTLBuffer> bVin = createBuf(c, kvN * 4, nullptr, true);
    id<MTLBuffer> bQhat = createBuf(c, atN * 4, nullptr, true);
    id<MTLBuffer> bKC = createBuf(c, (size_t)hKV * tmax * dh * 4, nullptr, true);
    id<MTLBuffer> bVC = createBuf(c, (size_t)hKV * tmax * dh * 4, nullptr, true);
    id<MTLBuffer> bRope = createBuf(c, (size_t)tmax * (nRot / 2) * 2 * 4, nullptr, true);
    id<MTLBuffer> bAtt = createBuf(c, atN * 4, nullptr, true);
    id<MTLBuffer> bAttnOut = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bY = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bXn2 = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bML = createBuf(c, (moe.nExp + 1) * 4, nullptr, true);
    id<MTLBuffer> bMH = createBuf(c, (size_t)(moe.nUsed + 1) * moe.nFf * 4, nullptr, true);
    id<MTLBuffer> bMSel = createBuf(c, 128, nullptr, true);
    id<MTLBuffer> bMY = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bOut = createBuf(c, nEmbd * 4, nullptr, true);
    {   // precomputed RoPE cos/sin table (see fa_prep binding 8)
        const uint32_t half = nRot / 2;
        float* rope = (float*)bRope.contents;
        for (uint32_t p = 0; p < tmax; p++)
            for (uint32_t j = 0; j < half; j++) {
                float th = (float)p * std::pow(kFreqBase, -2.f * (float)j / (float)nRot);
                rope[2 * ((size_t)p * half + j)]     = std::cos(th);
                rope[2 * ((size_t)p * half + j) + 1] = std::sin(th);
            }
    }

    struct { uint32_t n; float e; } pcRms{nEmbd, eps};
    struct { uint32_t m, k; } pcQf{qfN, nEmbd}, pcK{kvN, nEmbd}, pcWo{nEmbd, atN};
    struct FaPc { uint32_t pos, tmax, dh, nRot, hQ, hKV; float eps, fb; }
        pcFa{0, tmax, dh, nRot, hQ, hKV, eps, kFreqBase};
    struct { uint32_t n; } pcAdd{nEmbd};
    struct { uint32_t a, b, cc, d; } pcv{moe.nEmbd, moe.nFf, moe.nExp, moe.nUsed};

    auto dsp = [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                   std::initializer_list<id<MTLBuffer>> bufs,
                   const void* pc, uint32_t pcSize, uint32_t wgs, uint32_t thr) {
        [enc setComputePipelineState:pso];
        uint32_t i = 0;
        for (id<MTLBuffer> b : bufs) [enc setBuffer:b offset:0 atIndex:i++];
        [enc setBytes:pc length:pcSize atIndex:i];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
    };
    auto bar = [&](id<MTLComputeCommandEncoder> enc) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    const uint32_t thrN = nsg * 32;

    auto sequence = [&](id<MTLComputeCommandEncoder> enc, uint32_t pos) {
        pcFa.pos = pos;
        dsp(enc, pRms, {bXin, bANorm, bXn}, &pcRms, 8, 1, 256);
        bar(enc);
        dsp(enc, pGemvA, {bWq, bXn, bQfull}, &pcQf, 8, qfN / 4, 256);
        dsp(enc, pGemvA, {bWk, bXn, bKin}, &pcK, 8, kvN / 4, 256);
        dsp(enc, pGemvA, {bWv, bXn, bVin}, &pcK, 8, kvN / 4, 256);
        bar(enc);
        dsp(enc, pPrep, {bQfull, bKin, bVin, bQN, bKN, bQhat, bKC, bVC, bRope},
            &pcFa, 32, hQ + 2 * hKV, 256);
        bar(enc);
        dsp(enc, pAttn, {bQhat, bKC, bVC, bQfull, bAtt}, &pcFa, 32, hQ, 256);
        bar(enc);
        dsp(enc, pGemvO, {bWo, bAtt, bAttnOut}, &pcWo, 8, nEmbd / 2, 256);
        bar(enc);
        dsp(enc, pAddN, {bXin, bAttnOut, bPN, bY, bXn2}, &pcRms, 8, 1, 256);
        bar(enc);
        dsp(enc, pMoeL, {bMGI, bMGIS, bXn2, bML}, &pcv, 16, (moe.nExp + 1 + nsg - 1) / nsg, thrN);
        bar(enc);
        dsp(enc, pMoeS, {bML, bMSel}, &pcv, 16, 1, 32);
        bar(enc);
        dsp(enc, pMoeGu, {bMGE, bMUE, bMGS, bMUS, bXn2, bMSel, bMH}, &pcv, 16,
            ((moe.nUsed + 1) * moe.nFf + nsg - 1) / nsg, thrN);
        bar(enc);
        dsp(enc, pMoeDn, {bMDE, bMDS, bMH, bMSel, bMY}, &pcv, 16, (nEmbd + nsg - 1) / nsg, thrN);
        bar(enc);
        dsp(enc, pAdd, {bY, bMY, bOut}, &pcAdd, 4, 1, 256);
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
        memcpy(bXin.contents, x.data(), nEmbd * 4);
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc =
                [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            sequence(enc, t);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        memcpy(gpuOut.data(), bOut.contents, nEmbd * 4);

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

    if (pass && iters > 0) {
        auto runBench = [&]() -> double {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc =
                    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                for (uint32_t i = 0; i < iters; i++) {
                    sequence(enc, nTok - 1);
                    bar(enc);
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / iters;
            }
        };
        runBench();
        double ns = runBench();
        printf("gpu: %8.1f µs/block (pos=%u) | 13 dispatches, 1 submit\n", ns / 1e3, nTok - 1);
    }
    return pass;
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

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string mode = argc > 1 ? argv[1] : "suite";

        if (mode == "list") {
            listTensors(argc > 2 ? argv[2] : "");
            return 0;
        }

        MtlCtx c;
        initMtl(c, argv[0]);

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
        } else {
            fprintf(stderr, "mode '%s' not ported to Metal yet "
                            "(M2: q8_0/q6_k/iq4_xs/iq3_xxs; M3: moe/block/ablock; "
                            "M4: token; M5: serve)\n", mode.c_str());
            return 1;
        }
        return ok ? 0 : 1;
    }
}
