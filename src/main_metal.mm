// qk — Qwen kernel harness, Metal host (Apple Silicon port of main.cpp).
// M1: fp16 GEMV. M2: quantized GEMV (Q8_0, Q6_K, IQ4_XS, IQ3_XXS) on raw
// ggml blocks, validated against CPU dequant reference and real tensors
// from the GGUF. (M3: fused blocks, M4: token loop, M5: qk.h engine.)
//
// Usage:
//   qk                        synthetic suite: f16, q8_0, q6_k, iq4_xs, iq3_xxs
//   qk f16|q8_0|q6_k|iq4_xs|iq3_xxs [M] [K] [iters]
//   qk slotgemv [M] [K] [B] [TPR] [iters]  Q8_0 slot-batch comparison
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

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"
#include "quants.h"
#include "../include/qk.h"

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

// Compare a slot-batched Q8_0 kernel against the existing grid-z dispatch on
// distinct RHS vectors. This catches accidental cross-slot reuse that the
// identical-slot serving determinism test cannot see.
static bool caseQ80Batch(MtlCtx& c, uint32_t M, uint32_t K, uint32_t B,
                         uint32_t tpr, uint32_t iters) {
    if (K % 32 || (B != 2 && B != 4 && B != 8) ||
        (tpr != 32 && tpr != 64 && tpr != 128)) {
        fprintf(stderr, "slotgemv requires K%%32=0, B={2,4,8}, TPR={32,64,128}\n");
        return false;
    }
    const size_t nb = (size_t)M * K / 32;
    std::vector<block_q8_0> blocks(nb);
    std::mt19937 rng(42);
    for (auto& b : blocks) {
        b.d = qk_f32_to_f16(0.005f + 0.02f * (rng() & 0xFFFF) / 65536.0f);
        for (auto& q : b.qs) q = (int8_t)((int)(rng() % 255) - 127);
    }
    std::vector<float> x((size_t)B * K);
    for (uint32_t rq = 0; rq < B; ++rq) {
        auto xr = randomX(K, 100u + rq * 977u);
        memcpy(x.data() + (size_t)rq * K, xr.data(), K * 4);
    }

    const size_t wBytes = blocks.size() * sizeof(block_q8_0);
    const size_t xBytes = x.size() * 4;
    const size_t yBytes = (size_t)B * M * 4;
    id<MTLBuffer> bW = createBuf(c, wBytes, blocks.data());
    id<MTLBuffer> bX = createBuf(c, xBytes, x.data());
    id<MTLBuffer> bRef = createBuf(c, yBytes);
    id<MTLBuffer> bBat = createBuf(c, yBytes);
    id<MTLComputePipelineState> pRef = getPipe(c, "gemv_q8_0", "gemv_q8_0", tpr);
    char fn[32];
    snprintf(fn, sizeof fn, "gemv_q8_0_b%u", B);
    id<MTLComputePipelineState> pBat = getPipe(c, "gemv_q8_0_batch", fn, tpr);
    struct { uint32_t m, k; } pc{M, K};
    const uint32_t rowsPerTg = 256u / tpr;
    const uint32_t wgs = (M + rowsPerTg - 1u) / rowsPerTg;

    auto dispatch = [&](id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso, id<MTLBuffer> y,
                        uint32_t gridZ) {
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:y offset:0 atIndex:2];
        [enc setBytes:&pc length:8 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, gridZ)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto once = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y,
                    uint32_t gridZ) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            dispatch(enc, pso, y, gridZ);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
    };
    once(pRef, bRef, B);
    once(pBat, bBat, 1);

    const float* ref = (const float*)bRef.contents;
    const float* bat = (const float*)bBat.contents;
    double maxAbs = 0.0, maxRel = 0.0;
    size_t bitDiff = 0;
    for (size_t i = 0; i < (size_t)B * M; ++i) {
        if (memcmp(ref + i, bat + i, sizeof(float))) bitDiff++;
        const double ae = std::fabs((double)bat[i] - ref[i]);
        maxAbs = std::max(maxAbs, ae);
        maxRel = std::max(maxRel, ae / std::max(1e-5, (double)std::fabs(ref[i])));
    }
    const bool pass = bitDiff == 0;
    printf("\n== slot Q8_0 GEMV M=%u K=%u B=%u TPR=%u ==\n", M, K, B, tpr);
    printf("correctness: bit_diff=%zu max_abs=%.3g max_rel=%.3g -> %s\n",
           bitDiff, maxAbs, maxRel, pass ? "BIT-EXACT" : "FAIL");

    if (pass && iters) {
        auto bench = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y,
                         uint32_t gridZ) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (uint32_t i = 0; i < iters; ++i) dispatch(enc, pso, y, gridZ);
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / iters;
            }
        };
        bench(pRef, bRef, B); bench(pBat, bBat, 1);
        const double nsRef = bench(pRef, bRef, B);
        const double nsBat = bench(pBat, bBat, 1);
        const double refTraffic = B * (double)wBytes + xBytes + yBytes;
        const double batTraffic = (double)wBytes + xBytes + yBytes;
        printf("serial-z: %8.1f us %7.1f GB/s | batched: %8.1f us %7.1f GB/s | %.2fx\n",
               nsRef / 1e3, refTraffic / nsRef, nsBat / 1e3,
               batTraffic / nsBat, nsRef / nsBat);
    }
    return pass;
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

// Mirror per-request cache/spec stats to stderr and an optional durable file.
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
    const bool guIq4 = tGE->type == GGML_IQ4_XS && tUE->type == GGML_IQ4_XS;
    const bool guIq3 = tGE->type == GGML_IQ3_XXS && tUE->type == GGML_IQ3_XXS;
    if (tGI->type != GGML_F32 || tGIS->type != GGML_F32 ||
        (!guIq3 && !guIq4) ||
        (tDE->type != GGML_IQ4_XS && tDE->type != GGML_Q6_K) ||
        tGS->type != GGML_Q8_0 || tUS->type != GGML_Q8_0 || tDS->type != GGML_Q8_0) {
        fprintf(stderr, "layer %u tensor types don't match the compiled kernels\n", layer);
        return false;
    }
    const bool downQ6 = tDE->type == GGML_Q6_K;  // layers 34/38/39 in this GGUF

    const uint32_t nEmbd = (uint32_t)tGE->ne[0];
    const uint32_t nFf   = (uint32_t)tGE->ne[1];
    const uint32_t nExp  = (uint32_t)tGE->ne[2];
    const uint32_t nUsed =
        (uint32_t)g.kvInt(g.kvStr("general.architecture", "") + ".expert_used_count", 8);
    if (nExp > 512 || nUsed > 16) {
        fprintf(stderr, "n_expert %u / top-%u exceeds moe_select limits (512/16)\n", nExp, nUsed);
        return false;
    }
    printf("\n== moe blk.%u  n_embd=%u n_ff=%u experts=%u top-%u + shared ==\n",
           layer, nEmbd, nFf, nExp, nUsed);

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
    for (uint32_t s = 0; s < nUsed; s++) {
        uint32_t e = ids[s];
        auto dqGU = [&](const GgufTensor* t, uint32_t r2, float* out) {
            const uint8_t* row = t->data + ((size_t)e * nFf + r2) * rbGE;
            if (guIq4) dequant_row_iq4_xs((const block_iq4_xs*)row, out, nEmbd);
            else       dequant_row_iq3_xxs((const block_iq3_xxs*)row, out, nEmbd);
        };
        for (uint32_t r = 0; r < nFf; r++) {
            dqGU(tGE, r, tmpE.data());
            double ga = 0;
            for (uint32_t k = 0; k < nEmbd; k++) ga += (double)tmpE[k] * x[k];
            dqGU(tUE, r, tmpE.data());
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
    id<MTLComputePipelineState> pGu     = getPipe(c, "moe_gateup_all",
        guIq4 ? "moe_gateup_all_iq4" : "moe_gateup_all", nsg);
    id<MTLComputePipelineState> pDn     = downQ6 ? getPipe(c, "moe_down_all", "moe_down_all_q6k", nsg)
                                                 : getPipe(c, "moe_down_all", "moe_down_all_iq4", nsg);

    const size_t szGI = (size_t)nExp * nEmbd * 4, szGIS = (size_t)nEmbd * 4;
    const size_t szGE = (size_t)nExp * nFf * rbGE, szDE = (size_t)nExp * nEmbd * rbDE;
    const size_t szGS = (size_t)nFf * rbGS, szDS = (size_t)nEmbd * rbDS;
    const size_t szX = (size_t)nEmbd * 4, szY = (size_t)nEmbd * 4;
    const size_t szH = (size_t)(nUsed + 1) * nFf * 4, szSel = 160, szL = (size_t)(nExp + 1) * 4;

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
    struct SelOut { uint32_t ids[16]; float w[16]; float wShared; } selGpu;
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


// moegcmp: batched gate+up GPU-vs-GPU isolation — same random x/sel through
// the ungrouped kernel, grouped v1 (bit-exact control), and grouped v2
// (decode-once MMA); diffs the h tensors directly. Separates kernel bugs
// from whole-model noise when prefillcmp moves.
static bool caseMoeGrp(MtlCtx& c, uint32_t layer, uint32_t n, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    char nb[128];
    auto T = [&](const char* suffix) -> const GgufTensor* {
        snprintf(nb, sizeof nb, "blk.%u.%s", layer, suffix);
        const GgufTensor* t = g.find(nb);
        if (!t) fprintf(stderr, "missing tensor %s\n", nb);
        return t;
    };
    const GgufTensor* tGE = T("ffn_gate_exps.weight");
    const GgufTensor* tUE = T("ffn_up_exps.weight");
    const GgufTensor* tGS = T("ffn_gate_shexp.weight");
    const GgufTensor* tUS = T("ffn_up_shexp.weight");
    if (!tGE || !tUE || !tGS || !tUS) return false;
    if (tGE->type != GGML_IQ3_XXS || tUE->type != GGML_IQ3_XXS ||
        tGS->type != GGML_Q8_0 || tUS->type != GGML_Q8_0) {
        fprintf(stderr, "layer %u tensor types don't match the compiled kernels\n", layer);
        return false;
    }
    const uint32_t nEmbd = (uint32_t)tGE->ne[0];
    const uint32_t nFf   = (uint32_t)tGE->ne[1];
    const uint32_t nExp  = (uint32_t)tGE->ne[2];
    const uint32_t nUsed = 8;
    const uint32_t hs = (nUsed + 1) * nFf;
    printf("\n== moegcmp blk.%u  n=%u  n_embd=%u n_ff=%u experts=%u top-%u + shared ==\n",
           layer, n, nEmbd, nFf, nExp, nUsed);

    // synthetic inputs: gate+up reads only x and sel.ids (weights unused here)
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xs((size_t)n * nEmbd);
    for (auto& v : xs) v = nd(rng);
    struct SelH { uint32_t ids[16]; float w[16]; float wShared; float pad[7]; };
    std::vector<SelH> sel(n);
    for (uint32_t t = 0; t < n; t++) {
        std::vector<uint32_t> pool(nExp);
        for (uint32_t e = 0; e < nExp; e++) pool[e] = e;
        for (uint32_t s = 0; s < nUsed; s++) {
            uint32_t j = s + rng() % (nExp - s);
            std::swap(pool[s], pool[j]);
            sel[t].ids[s] = pool[s];
            sel[t].w[s] = 0.125f;
        }
        sel[t].wShared = 0.5f;
    }

    const size_t rbGE = ggmlRowBytes(GGML_IQ3_XXS, nEmbd);
    const size_t rbGS = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t szGE = (size_t)nExp * nFf * rbGE, szGS = (size_t)nFf * rbGS;
    id<MTLBuffer> bGE = createBuf(c, szGE, tGE->data, true), bUE = createBuf(c, szGE, tUE->data, true);
    id<MTLBuffer> bGS = createBuf(c, szGS, tGS->data, true), bUS = createBuf(c, szGS, tUS->data, true);
    id<MTLBuffer> bX = createBuf(c, xs.size() * 4, xs.data(), true);
    id<MTLBuffer> bSel = createBuf(c, (size_t)n * 160, sel.data(), true);
    id<MTLBuffer> bH0 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH1 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH2 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH3 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH4 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH5 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bStart = createBuf(c, (size_t)(nExp + 2) * 4, nullptr, true);
    id<MTLBuffer> bATok = createBuf(c, (size_t)n * (nUsed + 1) * 4, nullptr, true);
    id<MTLBuffer> bASlot = createBuf(c, (size_t)n * (nUsed + 1) * 4, nullptr, true);

    const uint32_t nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;
    const uint32_t thrN = nsg * 32;
    id<MTLComputePipelineState> pGu  = getPipe(c, "moe_gateup_all", "moe_gateup_all", nsg);
    id<MTLComputePipelineState> pGrp = getPipe(c, "moe_grouped", "moe_group", 0);
    id<MTLComputePipelineState> pG1  = getPipe(c, "moe_grouped", "moe_gu_grouped", nsg);
    id<MTLComputePipelineState> pG2  = getPipe(c, "moe_grouped", "moe_gu_grouped2", 0);
    id<MTLComputePipelineState> pG3  = getPipe(c, "moe_grouped", "moe_gu_grouped3", 0);
    id<MTLComputePipelineState> pG4  = getPipe(c, "moe_grouped", "moe_gu_grouped4", 0);
    id<MTLComputePipelineState> pG5  = getPipe(c, "moe_grouped", "moe_gu_grouped5", 0);

    struct { uint32_t a, b, cc, d; } pcv{nEmbd, nFf, nExp, nUsed};
    struct { uint32_t a, b, cc, d, n; } pcg{nEmbd, nFf, nExp, nUsed, n};
    auto dsp = [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                   std::initializer_list<id<MTLBuffer>> bufs, const void* pc, uint32_t pcSz,
                   uint32_t tgs, uint32_t thr, uint32_t z) {
        [enc setComputePipelineState:pso];
        uint32_t i = 0;
        for (id<MTLBuffer> b : bufs) [enc setBuffer:b offset:0 atIndex:i++];
        [enc setBytes:pc length:pcSz atIndex:i];
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, z)
            threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
    };
    auto runAll = [&](bool bench) -> std::array<double, 6> {
        std::array<double, 6> ms{0, 0, 0, 0, 0, 0};
        for (int which = 0; which < 6; which++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc =
                    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                for (uint32_t it = 0; it < (bench ? iters : 1); it++) {
                    if (which == 0) {
                        dsp(enc, pGu, {bGE, bUE, bGS, bUS, bX, bSel, bH0}, &pcv, 16,
                            (hs + nsg - 1) / nsg, thrN, n);
                    } else {
                        dsp(enc, pGrp, {bSel, bStart, bATok, bASlot}, &pcg, 20, 1, 256, 1);
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        if (which == 1)
                            dsp(enc, pG1, {bGE, bUE, bGS, bUS, bX, bStart, bATok, bASlot, bH1},
                                &pcv, 16, ((nExp + 1) * nFf + nsg - 1) / nsg, thrN, 1);
                        else if (which == 2)
                            dsp(enc, pG2, {bGE, bUE, bGS, bUS, bX, bStart, bATok, bASlot, bH2},
                                &pcv, 16, (nExp + 1) * (nFf / 32), 128, 1);
                        else if (which == 3)
                            dsp(enc, pG3, {bGE, bUE, bGS, bUS, bX, bStart, bATok, bASlot, bH3},
                                &pcv, 16, (nExp + 1) * (nFf / 64), 256, 1);
                        else if (which == 4)
                            dsp(enc, pG4, {bGE, bUE, bGS, bUS, bX, bStart, bATok, bASlot, bH4},
                                &pcv, 16, (nExp + 1) * (nFf / 32), 128, 1);
                        else
                            dsp(enc, pG5, {bGE, bUE, bGS, bUS, bX, bStart, bATok, bASlot, bH5},
                                &pcv, 16, (nExp + 1) * (nFf / 32), 128, (n + 31) / 32);
                    }
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                ms[which] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / (bench ? iters : 1);
            }
        }
        return ms;
    };
    runAll(false);

    const float* h0 = (const float*)bH0.contents;
    const float* h1 = (const float*)bH1.contents;
    const float* h2 = (const float*)bH2.contents;
    const float* h3 = (const float*)bH3.contents;
    const float* h4 = (const float*)bH4.contents;
    const float* h5 = (const float*)bH5.contents;
    double rms = 0;
    for (size_t i = 0; i < (size_t)n * hs; i++) rms += (double)h0[i] * h0[i];
    rms = std::sqrt(rms / ((size_t)n * hs));
    const double floorD = std::max(1e-4, 1e-3 * rms);
    uint32_t diff1 = 0;
    double maxRel2 = 0, maxRel3 = 0, maxRel4 = 0, maxRel5 = 0;
    size_t argRel2 = 0, argRel3 = 0;
    for (size_t i = 0; i < (size_t)n * hs; i++) {
        if (h1[i] != h0[i]) diff1++;
        double rel = std::fabs((double)h2[i] - h0[i]) /
                     std::max(floorD, (double)std::fabs(h0[i]));
        if (rel > maxRel2) { maxRel2 = rel; argRel2 = i; }
        rel = std::fabs((double)h3[i] - h0[i]) /
              std::max(floorD, (double)std::fabs(h0[i]));
        if (rel > maxRel3) { maxRel3 = rel; argRel3 = i; }
        rel = std::fabs((double)h4[i] - h0[i]) /
              std::max(floorD, (double)std::fabs(h0[i]));
        maxRel4 = std::max(maxRel4, rel);
        rel = std::fabs((double)h5[i] - h0[i]) /
              std::max(floorD, (double)std::fabs(h0[i]));
        maxRel5 = std::max(maxRel5, rel);
    }
    printf("v1 (bit-exact control): %u/%zu entries differ -> %s\n",
           diff1, (size_t)n * hs, diff1 ? "FAIL" : "PASS");
    printf("v2 (f32 decode-once):   max_rel = %.3g at [tok=%zu slot=%zu r=%zu] "
           "(h0=%g h2=%g)\n", maxRel2, argRel2 / hs, (argRel2 % hs) / nFf,
           argRel2 % nFf, h0[argRel2], h2[argRel2]);
    printf("v3 (f16 wide):          max_rel = %.3g at [tok=%zu slot=%zu r=%zu] "
           "(h0=%g h3=%g)\n", maxRel3, argRel3 / hs, (argRel3 % hs) / nFf,
           argRel3 % nFf, h0[argRel3], h3[argRel3]);
    printf("v4 (f32 wide):          max_rel = %.3g\n", maxRel4);
    printf("v5 (f16 occ 32-row):    max_rel = %.3g\n", maxRel5);

    if (iters > 0) {
        runAll(true);
        auto ms = runAll(true);
        printf("bench (%u iters): ungrouped %.3f ms | v1 %.3f ms | v2 %.3f ms | v3 %.3f ms | v4 %.3f ms | v5 %.3f ms\n",
               iters, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5]);
    }
    return diff1 == 0;
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
    m.nUsed = (uint32_t)g.kvInt(g.kvStr("general.architecture", "") + ".expert_used_count", 8);
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
    const uint32_t dnKDivH = 0;   // caseBlock is 35B-only (qwen35moe modulo)
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
    id<MTLComputePipelineState> pStep  = getPipe(c, "dn_step", "dn_step", 0);
    id<MTLComputePipelineState> pGemvO = getPipe(c, "gemv_q8_0", "gemv_q8_0", 128);  // K = d_inner
    id<MTLComputePipelineState> pAddN  = getPipe(c, "add_rmsnorm", "add_rmsnorm", 0);
    id<MTLComputePipelineState> pAdd   = getPipe(c, "vec_add", "vec_add", 0);
    id<MTLComputePipelineState> pMoeL  = getPipe(c, "moe_logits", "moe_logits", nsg);
    id<MTLComputePipelineState> pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    id<MTLComputePipelineState> pMoeLA = getPipe(c, "moe_logits", "moe_logits_addn", nsg);
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
    struct { uint32_t d, hk, hv; float e; uint32_t kd; } pcStep{dS, hK, hV, eps, dnKDivH};
    struct { uint32_t n; } pcAdd{nEmbd};
    struct { uint32_t a, b, cc, d; } pcv{moe.nEmbd, moe.nFf, moe.nExp, moe.nUsed};
    struct { uint32_t a, b, cc, d; float e; } pcv5{moe.nEmbd, moe.nFf, moe.nExp, moe.nUsed, eps};

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
        dsp(enc, pStep, {bQkv, bConvSt, bKer, bGb, bS, bZ, bSN, bAtt}, &pcStep, 20, hK, dS);
        bar(enc);
        dsp(enc, pGemvO, {bOutW, bAtt, bAttnOut}, &pcOut, 8, nEmbd / 2, 256);
        bar(enc);
        dsp(enc, pMoeLA, {bMGI, bMGIS, bXin, bAttnOut, bPN, bML, bY, bXn2}, &pcv5, 20,
            (moe.nExp + 1 + nsg - 1) / nsg, thrN);
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
        printf("gpu: %8.1f µs/block | 11 dispatches, 1 submit\n", ns / 1e3);
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
    id<MTLComputePipelineState> pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    id<MTLComputePipelineState> pMoeLA = getPipe(c, "moe_logits", "moe_logits_addn", nsg);
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
    struct { uint32_t a, b, cc, d; float e; } pcv5{moe.nEmbd, moe.nFf, moe.nExp, moe.nUsed, eps};

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
        dsp(enc, pMoeLA, {bMGI, bMGIS, bXin, bAttnOut, bPN, bML, bY, bXn2}, &pcv5, 20,
            (moe.nExp + 1 + nsg - 1) / nsg, thrN);
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
        printf("gpu: %8.1f µs/block (pos=%u) | 12 dispatches, 1 submit\n", ns / 1e3, nTok - 1);
    }
    return pass;
}


// ---------- end-to-end greedy decode (M4) ----------
// Embeds from GPU-resident ids, runs all 40 layers (30 deltanet + 10 full
// attention), LM head + two-pass argmax on GPU; the sampled token feeds the
// next step's embedding without host involvement. Prefill = one command
// buffer; the whole greedy generation = one more.
static bool caseToken(MtlCtx& c, const char* idsFile, uint32_t nGen, uint32_t tmax) {
    std::vector<int> promptIds;
    {
        FILE* f = fopen(idsFile, "r");
        if (!f) { perror(idsFile); return false; }
        int v;
        while (fscanf(f, "%d%*[, \n]", &v) == 1) promptIds.push_back(v);
        fclose(f);
    }
    if (promptIds.empty() || promptIds.size() + nGen > tmax || tmax > 1024) {
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
    const uint32_t dh = 256, hQ = 16, hKV = 2, nRot = 64;
    const std::string arch = g.kvStr("general.architecture", "");
    const uint32_t dnKDivH = (arch == "qwen3next") ? hV / hK : 0;
    const uint32_t nLayer = (uint32_t)g.kvInt(arch + ".block_count", 40);
    const uint32_t nUsed = (uint32_t)g.kvInt(arch + ".expert_used_count", 8);
    const GgufTensor* tge0 = g.find("blk.0.ffn_gate_exps.weight");
    const GgufTensor* tgi0 = g.find("blk.0.ffn_gate_inp.weight");
    if (!tge0 || !tgi0) { fprintf(stderr, "missing router tensors\n"); return false; }
    const uint32_t nExp = (uint32_t)tgi0->ne[1], ffE = (uint32_t)tge0->ne[1];
    const uint32_t vocab = (uint32_t)tHead->ne[1];
    const float eps = 1e-6f;
    const size_t rbQ8e = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbQ8i = ggmlRowBytes(GGML_Q8_0, dIn);
    const size_t rbE = ggmlRowBytes(GGML_Q6_K, nEmbd);
    printf("token mode: %zu prompt ids, gen %u, vocab %u, tmax %u\n",
           promptIds.size(), nGen, vocab, tmax);

    const uint32_t nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;
    id<MTLComputePipelineState> pRms   = getPipe(c, "rmsnorm", "rmsnorm", 0);
    id<MTLComputePipelineState> pGemvA = getPipe(c, "gemv_q8_0", "gemv_q8_0", 64);
    id<MTLComputePipelineState> pAb    = getPipe(c, "dn_ab", "dn_ab", nsg);
    id<MTLComputePipelineState> pStep  = getPipe(c, "dn_step", "dn_step", 0);
    id<MTLComputePipelineState> pGemvO = getPipe(c, "gemv_q8_0", "gemv_q8_0", 128);
    id<MTLComputePipelineState> pAddN  = getPipe(c, "add_rmsnorm", "add_rmsnorm", 0);
    id<MTLComputePipelineState> pPrep  = getPipe(c, "fa_prep", "fa_prep", 0);
    id<MTLComputePipelineState> pAttn  = getPipe(c, "fa_attn", "fa_attn", 0);
    id<MTLComputePipelineState> pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    id<MTLComputePipelineState> pMoeLA = getPipe(c, "moe_logits", "moe_logits_addn", nsg);
    id<MTLComputePipelineState> pMoeGu = getPipe(c, "moe_gateup_all", "moe_gateup_all", nsg);
    id<MTLComputePipelineState> pMoeD4 = getPipe(c, "moe_down_all", "moe_down_all_iq4", nsg);
    id<MTLComputePipelineState> pMoeD6 = getPipe(c, "moe_down_all", "moe_down_all_q6k", nsg);
    id<MTLComputePipelineState> pHead  = getPipe(c, "gemv_q6_k", "gemv_q6_k", 2);
    id<MTLComputePipelineState> pAm1   = getPipe(c, "argmax", "argmax1", 0);
    id<MTLComputePipelineState> pAm2   = getPipe(c, "argmax", "argmax2", 0);
    id<MTLComputePipelineState> pEmb   = getPipe(c, "embed_q6k", "embed_q6k", 0);

    // shared activation buffers
    id<MTLBuffer> bXin = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bXn = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bBig = createBuf(c, chQkv * 4, nullptr, true);       // qkv | qfull
    id<MTLBuffer> bMid = createBuf(c, dIn * 4, nullptr, true);         // z | qhat
    id<MTLBuffer> bKin = createBuf(c, hKV * dh * 4, nullptr, true);
    id<MTLBuffer> bVin = createBuf(c, hKV * dh * 4, nullptr, true);
    id<MTLBuffer> bGb = createBuf(c, 2 * hV * 4, nullptr, true);
    id<MTLBuffer> bConvOut = createBuf(c, chQkv * 4, nullptr, true);
    id<MTLBuffer> bO = createBuf(c, dIn * 4, nullptr, true);
    id<MTLBuffer> bAtt = createBuf(c, dIn * 4, nullptr, true);
    id<MTLBuffer> bAttnOut = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bY = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bXn2 = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bML = createBuf(c, (size_t)(nExp + 1) * 4, nullptr, true);
    id<MTLBuffer> bMH = createBuf(c, (size_t)(nUsed + 1) * ffE * 4, nullptr, true);
    id<MTLBuffer> bMSel = createBuf(c, 160, nullptr, true);
    id<MTLBuffer> bMY = createBuf(c, nEmbd * 4, nullptr, true);
    id<MTLBuffer> bONorm = createBuf(c, nEmbd * 4, tONorm->data, true);
    id<MTLBuffer> bHeadW = createBuf(c, (size_t)vocab * rbE, tHead->data, true);
    id<MTLBuffer> bLogits = createBuf(c, (size_t)vocab * 4, nullptr, true);
    id<MTLBuffer> bEmbdW = createBuf(c, (size_t)vocab * rbE, tEmbd->data, true);
    id<MTLBuffer> bAV = createBuf(c, 64 * 4, nullptr, true);
    id<MTLBuffer> bAI = createBuf(c, 64 * 4, nullptr, true);
    id<MTLBuffer> bTok = createBuf(c, 4, nullptr, true);
    id<MTLBuffer> bRb = createBuf(c, (size_t)tmax * 4, nullptr, true);
    id<MTLBuffer> bRope = createBuf(c, (size_t)tmax * (nRot / 2) * 2 * 4, nullptr, true);
    id<MTLBuffer> bPids = createBuf(c, promptIds.size() * 4, nullptr, true);
    {
        uint32_t* pi = (uint32_t*)bPids.contents;
        for (size_t i = 0; i < promptIds.size(); i++) pi[i] = (uint32_t)promptIds[i];
        const uint32_t half = nRot / 2;
        float* rope = (float*)bRope.contents;
        for (uint32_t p = 0; p < tmax; p++)
            for (uint32_t j = 0; j < half; j++) {
                float th = (float)p * std::pow(kFreqBase, -2.f * (float)j / (float)nRot);
                rope[2 * ((size_t)p * half + j)]     = std::cos(th);
                rope[2 * ((size_t)p * half + j) + 1] = std::sin(th);
            }
    }

    struct Layer {
        bool rec = false, downQ6 = false;
        id<MTLBuffer> aNorm, pn;
        id<MTLBuffer> qkvW, zW, alW, beW, dt, av, ker, sn, outW, convSt, S;   // rec
        id<MTLBuffer> wq, wk, wv, qn, kn, wo, kc, vc;                          // attn
        id<MTLBuffer> mgi, mgis, mge, mue, mde, mgs, mus, mds;                 // moe
    };
    std::vector<Layer> layers(nLayer);
    char nb[128];
    size_t umaMB = 0;
    auto W = [&](const GgufTensor* t, size_t n) -> id<MTLBuffer> {
        umaMB += n >> 20;
        return createBuf(c, n, t ? (const void*)t->data : nullptr, true);
    };
    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        auto T = [&](const char* suffix) -> const GgufTensor* {
            snprintf(nb, sizeof nb, "blk.%u.%s", il, suffix);
            return g.find(nb);
        };
        MoeT moe;
        if (!loadMoeT(g, il, moe)) return false;
        L.downQ6 = moe.downQ6;
        L.rec = T("ssm_a") != nullptr;
        L.aNorm = W(T("attn_norm.weight"), nEmbd * 4);
        L.pn = W(T("post_attention_norm.weight"), nEmbd * 4);
        if (L.rec) {
            L.qkvW = W(T("attn_qkv.weight"), (size_t)chQkv * rbQ8e);
            L.zW = W(T("attn_gate.weight"), (size_t)dIn * rbQ8e);
            L.alW = W(T("ssm_alpha.weight"), (size_t)hV * nEmbd * 4);
            L.beW = W(T("ssm_beta.weight"), (size_t)hV * nEmbd * 4);
            L.dt = W(T("ssm_dt.bias"), hV * 4);
            L.av = W(T("ssm_a"), hV * 4);
            L.ker = W(T("ssm_conv1d.weight") ? T("ssm_conv1d.weight") : T("ssm_conv1d"),
                      (size_t)chQkv * 4 * 4);
            L.sn = W(T("ssm_norm.weight"), dS * 4);
            L.outW = W(T("ssm_out.weight"), (size_t)nEmbd * rbQ8i);
            L.convSt = W(nullptr, (size_t)chQkv * 3 * 4);
            L.S = W(nullptr, (size_t)hV * dS * dS * 4);
            memset(L.convSt.contents, 0, (size_t)chQkv * 3 * 4);
            memset(L.S.contents, 0, (size_t)hV * dS * dS * 4);
        } else {
            L.wq = W(T("attn_q.weight"), (size_t)chQkv * rbQ8e);
            L.wk = W(T("attn_k.weight"), (size_t)hKV * dh * rbQ8e);
            L.wv = W(T("attn_v.weight"), (size_t)hKV * dh * rbQ8e);
            L.qn = W(T("attn_q_norm.weight"), dh * 4);
            L.kn = W(T("attn_k_norm.weight"), dh * 4);
            L.wo = W(T("attn_output.weight"), (size_t)nEmbd * rbQ8i);
            L.kc = W(nullptr, (size_t)hKV * tmax * dh * 4);
            L.vc = W(nullptr, (size_t)hKV * tmax * dh * 4);
        }
        L.mgi = W(moe.gi, (size_t)moe.nExp * nEmbd * 4);
        L.mgis = W(moe.gis, nEmbd * 4);
        L.mge = W(moe.ge, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        L.mue = W(moe.ue, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        L.mde = W(moe.de, (size_t)moe.nExp * moe.nEmbd * moe.rbDE);
        L.mgs = W(moe.gs, (size_t)moe.nFf * moe.rbGS);
        L.mus = W(moe.us, (size_t)moe.nFf * moe.rbGS);
        L.mds = W(moe.ds, (size_t)moe.nEmbd * moe.rbDS);
        if (il % 8 == 7) printf("  loaded %u/40 layers (%zu MB resident)\n", il + 1, umaMB);
    }
    printf("model resident: ~%zu MB\n", umaMB);

    struct { uint32_t n; float e; } pcRms{nEmbd, eps};
    struct { uint32_t m, k; } pcQkv{chQkv, nEmbd}, pcZ{dIn, nEmbd}, pcKV{hKV * dh, nEmbd},
        pcWo{nEmbd, dIn}, pcHead{vocab, nEmbd};
    struct { uint32_t n, h; } pcAb{nEmbd, hV};
    struct { uint32_t d, hk, hv; float e; uint32_t kd; } pcStep{dS, hK, hV, eps, dnKDivH};
    struct { uint32_t a, b, cc, d; } pcv{nEmbd, ffE, nExp, nUsed};
    struct { uint32_t a, b, cc, d; float e; } pcv5{nEmbd, ffE, nExp, nUsed, eps};
    struct { uint32_t pos, tmax, dh, nRot, hQ, hKV_; float eps, fb; }
        pcFa{0, tmax, dh, nRot, hQ, hKV, eps, kFreqBase};
    const uint32_t amWgs = (vocab + 4095) / 4096;
    struct { uint32_t n, span; } pcAm{vocab, 4096};
    struct { uint32_t m, pos; } pcAm2{amWgs, 0};
    struct { uint32_t kdim, idx, perReq; float e; } pcEmb{nEmbd, 0, 0, eps};
    const uint32_t thrN = nsg * 32;

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

    auto recordToken = [&](id<MTLComputeCommandEncoder> enc, uint32_t pos, bool withHead) {
        pcFa.pos = pos;   // embed_q6k already produced bXin AND bXn (layer-0 norm)
        for (uint32_t il = 0; il < nLayer; il++) {
            Layer& L = layers[il];
            if (L.rec) {
                dsp(enc, pGemvA, {L.qkvW, bXn, bBig}, &pcQkv, 8, chQkv / 4, 256);
                dsp(enc, pGemvA, {L.zW, bXn, bMid}, &pcZ, 8, dIn / 4, 256);
                dsp(enc, pAb, {bXn, L.alW, L.beW, L.dt, L.av, bGb}, &pcAb, 8,
                    (2 * hV + nsg - 1) / nsg, thrN);
                bar(enc);
                dsp(enc, pStep, {bBig, L.convSt, L.ker, bGb, L.S, bMid, L.sn, bAtt},
                    &pcStep, 20, hK, dS);
                bar(enc);
                dsp(enc, pGemvO, {L.outW, bAtt, bAttnOut}, &pcWo, 8, nEmbd / 2, 256);
            } else {
                dsp(enc, pGemvA, {L.wq, bXn, bBig}, &pcQkv, 8, chQkv / 4, 256);
                dsp(enc, pGemvA, {L.wk, bXn, bKin}, &pcKV, 8, hKV * dh / 4, 256);
                dsp(enc, pGemvA, {L.wv, bXn, bVin}, &pcKV, 8, hKV * dh / 4, 256);
                bar(enc);
                dsp(enc, pPrep, {bBig, bKin, bVin, L.qn, L.kn, bMid, L.kc, L.vc, bRope},
                    &pcFa, 32, hQ + 2 * hKV, 256);
                bar(enc);
                dsp(enc, pAttn, {bMid, L.kc, L.vc, bBig, bAtt}, &pcFa, 32, hQ, 256);
                bar(enc);
                dsp(enc, pGemvO, {L.wo, bAtt, bAttnOut}, &pcWo, 8, nEmbd / 2, 256);
            }
            bar(enc);
            dsp(enc, pMoeLA, {L.mgi, L.mgis, bXin, bAttnOut, L.pn, bML, bY, bXn2}, &pcv5, 20,
                (nExp + 1 + nsg - 1) / nsg, thrN);
            bar(enc);
            dsp(enc, pMoeS, {bML, bMSel}, &pcv, 16, 1, 32);
            bar(enc);
            dsp(enc, pMoeGu, {L.mge, L.mue, L.mgs, L.mus, bXn2, bMSel, bMH}, &pcv, 16,
                ((nUsed + 1) * ffE + nsg - 1) / nsg, thrN);
            bar(enc);
            dsp(enc, L.downQ6 ? pMoeD6 : pMoeD4, {L.mde, L.mds, bMH, bMSel, bMY}, &pcv, 16,
                (nEmbd + nsg - 1) / nsg, thrN);
            bar(enc);
            // layer tail: residual sum + NEXT layer's norm (output_norm at the end)
            id<MTLBuffer> nextNorm = il + 1 < nLayer ? layers[il + 1].aNorm : bONorm;
            dsp(enc, pAddN, {bY, bMY, nextNorm, bXin, bXn}, &pcRms, 8, 1, 256);
            bar(enc);
        }
        if (withHead) {
            // bXn holds output_norm(x) from the last layer tail
            dsp(enc, pHead, {bHeadW, bXn, bLogits}, &pcHead, 8, (vocab + 3) / 4, 64);
        }
    };

    const uint32_t nPrompt = (uint32_t)promptIds.size();

    // ---- prefill: one command buffer ----
    auto t0 = std::chrono::steady_clock::now();
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (uint32_t pos = 0; pos < nPrompt; pos++) {
            pcEmb.idx = pos;
            dsp(enc, pEmb, {bEmbdW, bPids, bXin, layers[0].aNorm, bXn}, &pcEmb, 16, 1, 256);
            bar(enc);
            recordToken(enc, pos, pos + 1 == nPrompt);
            bar(enc);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    auto t1 = std::chrono::steady_clock::now();
    double prefillMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- greedy decode: one command buffer, GPU-resident sampling ----
    double gpuS = 0;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (uint32_t i = 0; i < nGen; i++) {
            uint32_t pos = nPrompt + i;
            pcAm2.pos = i;
            dsp(enc, pAm1, {bLogits, bAV, bAI}, &pcAm, 8, amWgs, 256);
            bar(enc);
            dsp(enc, pAm2, {bAV, bAI, bTok, bRb}, &pcAm2, 8, 1, 256);
            bar(enc);
            pcEmb.idx = 0;
            dsp(enc, pEmb, {bEmbdW, bTok, bXin, layers[0].aNorm, bXn}, &pcEmb, 16, 1, 256);
            bar(enc);
            recordToken(enc, pos, true);
            bar(enc);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        gpuS = cb.GPUEndTime - cb.GPUStartTime;
    }
    auto t2 = std::chrono::steady_clock::now();
    double decodeMs = std::chrono::duration<double, std::milli>(t2 - t1).count();

    const uint32_t* rb = (const uint32_t*)bRb.contents;
    printf("GEN:");
    for (uint32_t i = 0; i < nGen; i++) printf(" %u", rb[i]);
    printf("\n");
    printf("prefill: %u tokens in %.1f ms (%.1f tok/s serial)\n",
           nPrompt, prefillMs, nPrompt * 1000.0 / prefillMs);
    printf("decode:  %u tokens in %.1f ms wall / %.1f ms gpu -> %.2f ms/tok, %.1f tok/s\n",
           nGen, decodeMs, gpuS * 1e3, gpuS * 1e3 / nGen, nGen / gpuS);
    return true;
}


// ===================== qk.h C ABI: persistent per-slot engine =====================
// Metal implementation of the server-oriented engine: N slots each hold their
// own sequence; one encoded step (grid z = slot) advances all of them, with
// per-slot positions from a slotPos buffer (fa_*_srv kernels). Uses the fused
// Metal kernel set (dn_step megakernel, moe_logits_addn, merged MoE), which is
// token-parity-validated against llama.cpp. On UMA, prefix-cache snapshots,
// state resets and carry seeding are plain memcpy/memset — no staging.

// (buffer, offset) pair for weight bindings: weights live at offsets inside
// ONE no-copy MTLBuffer wrapping the GGUF mmap (zero copies at open, no
// double residency). Implicitly constructible from a plain buffer so
// activation bindings read unchanged.
struct WB {
    id<MTLBuffer> b;
    NSUInteger o;
    WB(id<MTLBuffer> buf) : b(buf), o(0) {}
    WB(id<MTLBuffer> buf, NSUInteger off) : b(buf), o(off) {}
};

struct qk_engine {
    MtlCtx c{};
    Gguf g;
    id<MTLBuffer> gbuf;   // no-copy view of the whole GGUF mmap
    uint32_t nSlots = 0, nCtx = 0, chunkN = 0;
    bool shareFork = false;
    uint32_t vocab = 0, eosTok = 248046, bosTok = 248044;
    // Pipeline split (QK_LAYERS=a:b): this engine owns layers [lFirst, lEnd).
    uint32_t lFirst = 0, lEnd = 40;
    bool firstStage() const { return lFirst == 0; }
    bool lastStage() const { return lEnd == nLayer; }
    bool splitStage() const { return lFirst != 0 || lEnd != nLayer; }
    // Rows bbLogits held after the most recent head+argmax pass — stageTopK
    // reads the final row's logits from it (the sampling hook).
    uint32_t lastRunRows = 0;
    static constexpr uint32_t nEmbd = 2048, chQkv = 8192, dIn = 4096, hV = 32, dS = 128, hK = 16;
    static constexpr uint32_t dh = 256, hQ = 16, hKV = 2, nRot = 64;
    // Model-shape knobs that differ between Qwen3.6-35B (40/256/8/512) and
    // Qwen3-Next-80B (48/512/10/512); read from GGUF KVs / tensor shapes in
    // open(). Tensor DIMS above are identical across both models.
    uint32_t nLayer = 40, nExp = 256, nUsed = 8, ffE = 512;
    bool guIq4 = false;   // routed gate/up experts IQ4_XS (80B) vs IQ3_XXS (35B)
    bool embQ8 = false;   // token_embd Q8_0 (80B repack) vs Q6_K (35B)
    float eps = 1e-6f;
    uint32_t nsg = 4;

    id<MTLComputePipelineState> pGemvA, pGemvO, pGemvAB, pGemvOB,
        pAb, pStep, pPrep, pAttn,
        pAttnSplit, pAttnReduce, pMoeS,
        pMoeLA, pMoeGu, pMoeD4, pMoeD6, pAddN, pHead, pAm1, pAm2, pEmb,
        pPrepB, pAttnB, pAttnBM, pAbB, pConvB, pStepB, pGateB, pGemmB;
    // IQ4_XS twins for the 80B: dense-proj gemv/gemm + routed gate/up.
    id<MTLComputePipelineState> pGemv4, pGemmB4, pMoeGu4, pMoeGuG4i, pMoeGuG5i;
    // chunked delta rule (prefill DN): parallel kq/solve + chunk-serial step
    id<MTLComputePipelineState> pDnKq, pDnSolve, pDnStepC;
    id<MTLBuffer> bbDnKQ, bbDnUW, bbDnAtt, bbDnEl;
    bool dnChunk = true;   // QK_DN_CHUNK=0 falls back to dn_step_batch
    // DeltaNet GQA k-pairing: 0 = h % hK (qwen35moe), else h / kDiv
    // (qwen3next: hV/hK = 2). Set from general.architecture at open.
    uint32_t dnKDiv = 0;
    std::vector<id<MTLBuffer>> extraBufs;   // host-built weights (ssm_ba split)
    struct WeightWin { const uint8_t* base; size_t len; id<MTLBuffer> buf; };
    std::vector<WeightWin> gwin;            // no-copy windows when the file > maxBufferLength
    uint32_t gemmThreads = 256;
    uint32_t gemmBM = 128, gemmBN = 64;
    bool slotBatch = false;
    uint32_t slotTprA = 64, slotTprO = 128;
    uint32_t slotBatchN = 0;

    struct Layer {
        bool rec = false, downQ6 = false;
        // per-projection quant flags: true = IQ4_XS weight (dispatch the *4
        // pipe). p1 = qkv|q, p2 = z|k, p3 = v, wo = ssm_out|attn_output.
        bool iq4P1 = false, iq4P2 = false, iq4P3 = false, iq4Wo = false;
        WB aNorm{nil}, pn{nil};
        WB qkvW{nil}, zW{nil}, alW{nil}, beW{nil}, dt{nil}, av{nil}, ker{nil},
           sn{nil}, outW{nil};                                       // rec
        WB wq{nil}, wk{nil}, wv{nil}, qn{nil}, kn{nil}, wo{nil};     // attn
        WB mgi{nil}, mgis{nil}, mge{nil}, mue{nil}, mde{nil}, mgs{nil},
           mus{nil}, mds{nil};                                       // moe
        id<MTLBuffer> st1, st2;      // per-slot state: rec=(conv,S) attn=(kc,vc)
        size_t ps1 = 0, ps2 = 0;     // per-slot byte stride
    };
    std::vector<Layer> layers;

    id<MTLBuffer> bXin, bXn, bBig, bMid, bKin, bVin, bGb, bAtt, bAttnOut, bY, bXn2,
        bML, bMH, bMSel, bMY, bLogits, bRope, bAV, bAI,
        bTok, bRbScratch, bSlotIn, bSlotPos, bPartial;
    WB bONorm{nil}, bHeadW{nil}, bEmbdW{nil};

    uint32_t maxB = 0;
    double lastCbGpu = 0, lastCbWall = 0;
    id<MTLBuffer> bbXin, bbXn, bbBig, bbMid, bbKin, bbVin, bbGb, bbConvOut, bbO,
        bbAtt, bbAttnOut, bbY, bbXn2, bbML, bbMH, bbMSel, bbMY, bbLogits, bbIds, bbCarry,
        bbAV, bbAI, bbTok;
    id<MTLComputePipelineState> pRms, pMoeGrp, pMoeGrpLive,
        pMoeGuG, pMoeGuG2, pMoeGuG3, pMoeGuG4, pMoeGuG4Live,
        pMoeGuG5, pMoeGuG5Live,
        pMoeDG4, pMoeDG6, pMoeDG4Live, pMoeDG6Live,
        pMoeDGH4, pMoeDGH6, pMoeDGP4, pMoeDGP6,
        pMoeDGP4Live, pMoeDGP6Live, pMoeDR, pLogG;
    id<MTLBuffer> bbStart, bbATok, bbASlot, bbLive, bbMDy;
    // Grouped (decode-once) MoE gate+up for prefill chunks. Variants:
    // 1 = v1 read-once (SLOWER — kept as bit-exact control); 2 = v2 f32
    // narrow; 3 = v3 f16 64x32 (llama mul_mm_id class — fastest, opt-in);
    // 4 = v4 f32 32x32 (exact-class, default). Grouping only pays once
    // experts see enough tokens: default fires at n >= moeGroupN (192) with
    // variant 4; QK_MOE_GROUPED forces a variant at ALL n (0 disables),
    // QK_MOE_GROUP_N overrides the threshold.
    int moeGrouped = 4;
    uint32_t moeGroupN = 192;
    int moeSlotGrouped = 0;  // 35B default v4; env 0..5 selects/rolls back
    int moeSlotParts = 3;    // bit 0=gate/up, bit 1=down
    bool moeSlotLive = true;
    uint32_t moeSlotMin = 8;
    bool moeSlotSameOnly = true;

    struct Slot {
        bool active = false;
        std::vector<uint32_t> prompt;
        std::vector<uint32_t> genTokens;
        uint32_t cursor = 0, pos = 0, gen = 0, maxGen = 0, last = 0;
        // A verify round may emit more than the ABI chunk; overflow drains on
        // later calls without GPU work. finPending delays slot release until
        // that queue is empty.
        std::vector<uint32_t> outQ;
        size_t outQHead = 0;
        bool finPending = false;
        // Latest earlier occurrence of each specL-token gram in prompt+output.
        std::unordered_map<uint64_t, uint32_t> ngram;
        uint32_t ngramBuilt = 0;
        uint64_t equivId = 0;  // exact-prompt trajectory class for MoE compaction
        uint32_t specRounds = 0, specFed = 0, specEmitted = 0, serialSteps = 0;
        void resetSpec() {
            outQ.clear();
            outQHead = 0;
            finPending = false;
            ngram.clear();
            ngramBuilt = 0;
            specRounds = specFed = specEmitted = serialSteps = 0;
        }
    };
    std::vector<Slot> slots;
    uint64_t equivClock = 0;
    bool specOn = false;
    uint32_t specL = 6, specK = 8;
    std::vector<uint32_t> specToks, specAm;
    uint32_t specDraft(uint32_t slot);

    struct CacheEntry {
        std::vector<uint32_t> tokens;
        std::vector<uint8_t> snap;   // host copy of all st1/st2 stripes (UMA memcpy)
        uint64_t lru = 0;
        bool valid = false;
    };
    std::vector<CacheEntry> pcache;
    std::vector<size_t> snapOff1, snapOff2;
    size_t snapSize = 0;
    uint64_t lruClock = 0;

    bool open(const char* path, const qk_config& cfg, char* err, size_t errLen);
    int stepChunk(uint32_t* outTok, uint32_t* outCnt, uint32_t* outFin);
    void prefillBatchLast(const uint32_t* toks, uint32_t n, uint32_t slot,
                          std::vector<float>& logits, bool wantLogits = true, uint32_t base = 0,
                          uint32_t* argmaxOut = nullptr, const float* hiddenIn = nullptr,
                          float* hiddenOut = nullptr, bool scratchState = false);
    int stageRun(uint32_t slot, const uint32_t* toks, const float* hiddenIn, uint32_t n,
                 uint32_t base, float* hiddenOut, uint32_t* idsOut);
    // Top-k (ids, logits) of the final position's row after a last-stage
    // stageRun — the split driver's sampling hook (see qk_stage_topk in qk.h).
    int stageTopK(uint32_t k, uint32_t* idsOut, float* valsOut);
    uint32_t serialPrefillLogits(const uint32_t* toks, uint32_t n, uint32_t slot,
                                 std::vector<float>& logits);
    void resetSlot(uint32_t slot);
    void snapshotSlot(uint32_t slot);
    int matchPrefix(const uint32_t* prompt, uint32_t n);
    void restoreInto(uint32_t slot, int cacheIdx);
    void copyStripes(uint32_t slot, uint8_t* snap, bool save, uint32_t nTok = 0);
    // The recurrent scratch stripe is index nSlots. Verification advances
    // DeltaNet/conv there while attention KV writes remain on the live slot.
    void copyDnStripes(uint32_t fromStripe, uint32_t toStripe);
    void verifyRound(const uint32_t* toks, uint32_t n, uint32_t slot, uint32_t base,
                     uint32_t* outIds);
    void promoteScratch(uint32_t slot) { copyDnStripes(nSlots, slot); }
    void encodeStep(id<MTLComputeCommandEncoder> enc, uint32_t zdim,
                    bool equivalentSlots = false);
};

void qk_engine::resetSlot(uint32_t slot) {
    for (auto& L : layers) {   // GPU is idle at every call site (engine is synchronous)
        if (!L.st1) continue;  // layer outside this stage's [lFirst, lEnd)
        memset((uint8_t*)L.st1.contents + (size_t)slot * L.ps1, 0, L.ps1);
        memset((uint8_t*)L.st2.contents + (size_t)slot * L.ps2, 0, L.ps2);
    }
}

void qk_engine::copyStripes(uint32_t slot, uint8_t* snap, bool save, uint32_t nTok) {
    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        if (!L.st1) continue;
        uint8_t* s1 = (uint8_t*)L.st1.contents + (size_t)slot * L.ps1;
        uint8_t* s2 = (uint8_t*)L.st2.contents + (size_t)slot * L.ps2;
        // Attention KV stripes are [hKV][tmax][dh]: with a live token count,
        // copy only each kv-heads first nTok rows so snapshot cost and
        // resident pages track the conversation, not capacity. Recurrent
        // (DeltaNet/conv) stripes always copy whole.
        if (!L.rec && nTok && nTok < nCtx) {
            size_t headBytes = L.ps1 / hKV;
            size_t liveBytes = (size_t)nTok * dh * 4;
            for (uint32_t h = 0; h < hKV; h++) {
                size_t off = (size_t)h * headBytes;
                if (save) {
                    memcpy(snap + snapOff1[il] + off, s1 + off, liveBytes);
                    memcpy(snap + snapOff2[il] + off, s2 + off, liveBytes);
                } else {
                    memcpy(s1 + off, snap + snapOff1[il] + off, liveBytes);
                    memcpy(s2 + off, snap + snapOff2[il] + off, liveBytes);
                }
            }
            continue;
        }
        if (save) {
            memcpy(snap + snapOff1[il], s1, L.ps1);
            memcpy(snap + snapOff2[il], s2, L.ps2);
        } else {
            memcpy(s1, snap + snapOff1[il], L.ps1);
            memcpy(s2, snap + snapOff2[il], L.ps2);
        }
    }
}

void qk_engine::copyDnStripes(uint32_t fromStripe, uint32_t toStripe) {
    // Every caller is synchronous: the preceding command buffer has completed
    // and the next one has not been committed, so shared-memory copies are
    // coherent without another Metal submission.
    for (Layer& L : layers) {
        if (!L.rec || !L.st1) continue;
        memcpy((uint8_t*)L.st1.contents + (size_t)toStripe * L.ps1,
               (uint8_t*)L.st1.contents + (size_t)fromStripe * L.ps1, L.ps1);
        memcpy((uint8_t*)L.st2.contents + (size_t)toStripe * L.ps2,
               (uint8_t*)L.st2.contents + (size_t)fromStripe * L.ps2, L.ps2);
    }
}

void qk_engine::verifyRound(const uint32_t* toks, uint32_t n, uint32_t slot, uint32_t base,
                            uint32_t* outIds) {
    copyDnStripes(slot, nSlots);  // seed scratch from live state at `base`
    std::vector<float> dummy;
    prefillBatchLast(toks, n, slot, dummy, /*wantLogits=*/false, base, outIds,
                     /*hiddenIn=*/nullptr, /*hiddenOut=*/nullptr, /*scratchState=*/true);
}

void qk_engine::snapshotSlot(uint32_t slot) {
    if (pcache.empty()) return;
    Slot& sl = slots[slot];
    size_t fedGen = sl.pos > sl.prompt.size() ? sl.pos - sl.prompt.size() : 0;
    if (fedGen > sl.genTokens.size()) fedGen = sl.genTokens.size();
    if (sl.pos < 8 || sl.pos > nCtx) return;
    int idx = 0;
    for (uint32_t i = 0; i < pcache.size(); i++) {
        if (!pcache[i].valid) { idx = (int)i; break; }
        if (pcache[i].lru < pcache[idx].lru) idx = (int)i;
    }
    CacheEntry& e = pcache[idx];
    size_t pp = std::min((size_t)sl.pos, sl.prompt.size());
    e.tokens.assign(sl.prompt.begin(), sl.prompt.begin() + pp);
    e.tokens.insert(e.tokens.end(), sl.genTokens.begin(), sl.genTokens.begin() + fedGen);
    e.lru = ++lruClock;
    e.valid = true;
    copyStripes(slot, e.snap.data(), /*save=*/true);
}

int qk_engine::matchPrefix(const uint32_t* prompt, uint32_t n) {
    int best = -1;
    uint32_t bestLen = 8;
    for (uint32_t i = 0; i < pcache.size(); i++) {
        if (!pcache[i].valid) continue;
        const auto& tk = pcache[i].tokens;
        uint32_t L = (uint32_t)tk.size();
        if (L >= n || L <= bestLen) continue;
        bool ok = true;
        for (uint32_t j = 0; j < L; j++)
            if (tk[j] != prompt[j]) { ok = false; break; }
        if (ok) { best = (int)i; bestLen = L; }
    }
    return best;
}

void qk_engine::restoreInto(uint32_t slot, int cacheIdx) {
    pcache[cacheIdx].lru = ++lruClock;
    copyStripes(slot, pcache[cacheIdx].snap.data(), /*save=*/false);
}

bool qk_engine::open(const char* path, const qk_config& cfg, char* err, size_t errLen) {
    auto fail = [&](const char* m) { if (err && errLen) snprintf(err, errLen, "%s", m); return false; };
    nSlots = cfg.n_slots; nCtx = cfg.n_ctx; chunkN = cfg.chunk;
    shareFork = getenv("QK_FORK") != nullptr;
    if (nSlots < 1 || nSlots > 16 || nCtx < 64 || nCtx > 65536 || chunkN < 1 || chunkN > 32)
        return fail("qk_open: bad config");
    initMtl(c, "libqk");
    if (!g.open(path)) return fail("qk_open: cannot open GGUF");
    // Model-shape KVs (arch-prefixed): 35B = qwen35moe 40/8, 80B = qwen3next 48/10.
    const std::string arch = g.kvStr("general.architecture", "");
    dnKDiv = (arch == "qwen3next") ? hV / hK : 0;
    nLayer = (uint32_t)g.kvInt(arch + ".block_count", nLayer);
    if (nLayer < 1 || nLayer > 256) return fail("qk_open: bad block_count");
    nUsed = (uint32_t)g.kvInt(arch + ".expert_used_count", nUsed);
    if (nUsed < 1 || nUsed > 16) return fail("qk_open: expert_used_count > 16 unsupported");
    eosTok = (uint32_t)g.kvInt("tokenizer.ggml.eos_token_id", eosTok);
    bosTok = (uint32_t)g.kvInt("tokenizer.ggml.bos_token_id", bosTok);
    lEnd = nLayer;   // full model by default (member init used the pre-open default)
    if (const char* v = getenv("QK_LAYERS")) {   // pipeline-split stage [a,b)
        uint32_t a = 0, b = 0;
        if (sscanf(v, "%u:%u", &a, &b) == 2 && a < b && b <= nLayer) { lFirst = a; lEnd = b; }
        else {
            snprintf(err, errLen, "qk_open: bad QK_LAYERS (want a:b with 0 <= a < b <= %u)", nLayer);
            return false;
        }
    }
    {   // ONE no-copy buffer over the mmap: zero weight copies at open.
        // QK_COPY_WEIGHTS=1: copy the GGUF into a device-owned shared buffer
        // instead (RSS +16.6 GB) — probe for mmap-residency overhead.
        const size_t psz = (size_t)getpagesize();
        const size_t len = (g.size() + psz - 1) & ~(psz - 1);
        if (getenv("QK_COPY_WEIGHTS")) {
            gbuf = [c.dev newBufferWithLength:len
                                      options:MTLResourceStorageModeShared |
                                              MTLResourceHazardTrackingModeUntracked];
            if (!gbuf) return fail("qk_open: weight copy alloc failed");
            memcpy(gbuf.contents, g.base(), g.size());
        } else if (len <= c.dev.maxBufferLength &&
                   !getenv("QK_WIN_MAX")) {   // QK_WIN_MAX=<GB>: force windows (debug)
            gbuf = [c.dev newBufferWithBytesNoCopy:(void*)g.base()
                                            length:len
                                           options:MTLResourceStorageModeShared |
                                                   MTLResourceHazardTrackingModeUntracked
                                       deallocator:nil];
        } else {
            // window the mapping: sort tensors by address, cut windows at
            // tensor boundaries so no tensor straddles one
            std::vector<const GgufTensor*> ts;
            for (const auto& kv : g.tensors()) ts.push_back(&kv.second);
            std::sort(ts.begin(), ts.end(), [](const GgufTensor* a, const GgufTensor* b) {
                return a->data < b->data;
            });
            size_t maxLen = (size_t)c.dev.maxBufferLength & ~(psz - 1);
            if (const char* wm = getenv("QK_WIN_MAX"))
                maxLen = std::min(maxLen, ((size_t)atol(wm) << 30) & ~(psz - 1));
            size_t i = 0;
            while (i < ts.size()) {
                const uint8_t* w0 = (const uint8_t*)((uintptr_t)ts[i]->data & ~(uintptr_t)(psz - 1));
                size_t j = i;
                const uint8_t* end = nullptr;
                while (j < ts.size()) {
                    const GgufTensor* t = ts[j];
                    size_t bytes = ggmlRowBytes((GgmlType)t->type, (uint32_t)t->ne[0]) *
                                   t->ne[1] * t->ne[2] * t->ne[3];
                    const uint8_t* e = t->data + bytes;
                    if ((size_t)(e - w0) > maxLen) break;
                    end = e;
                    j++;
                }
                if (j == i) return fail("qk_open: tensor larger than maxBufferLength");
                size_t wlen = ((size_t)(end - w0) + psz - 1) & ~(psz - 1);
                id<MTLBuffer> wb = [c.dev newBufferWithBytesNoCopy:(void*)w0
                                                            length:wlen
                                                           options:MTLResourceStorageModeShared |
                                                                   MTLResourceHazardTrackingModeUntracked
                                                       deallocator:nil];
                if (!wb) return fail("qk_open: weight window alloc failed");
                gwin.push_back({w0, wlen, wb});
                i = j;
            }
            fprintf(stderr, "qk_open: %zu weight windows over %.1f GB\n",
                    gwin.size(), g.size() / 1e9);
        }
        if (!gbuf && gwin.empty()) return fail("qk_open: newBufferWithBytesNoCopy failed");
        // QK_MLOCK=1: wire the mapping. No-copy weights degrade 2-6x
        // after memory-pressure evictions (per-submit GPU rewiring of
        // faulted pages, sys-time bound) and do NOT self-heal; mlock
        // keeps zero-copy AND immunity. Serving configs want this.
        // A split stage wires only its OWNED tensors (the 80B file is
        // 42.8 GB — whole-file mlock next to another worker would wire
        // most of RAM); full-model engines wire the whole mapping.
        if (getenv("QK_MLOCK")) {
            if (!splitStage()) {
                if (mlock(g.base(), g.size()) != 0)
                    fprintf(stderr, "qk_open: mlock failed (%s) — continuing unwired\n",
                            strerror(errno));
            } else {
                size_t wired = 0;
                auto lockT = [&](const GgufTensor* t) {
                    if (!t) return;
                    size_t bytes = ggmlRowBytes((GgmlType)t->type, (uint32_t)t->ne[0]) *
                                   t->ne[1] * t->ne[2] * t->ne[3];
                    if (mlock((void*)t->data, bytes) == 0) wired += bytes;
                };
                for (const auto& kv : g.tensors()) {
                    const std::string& nm = kv.first;
                    bool own = false;
                    if (nm.rfind("blk.", 0) == 0) {
                        uint32_t il = (uint32_t)atoi(nm.c_str() + 4);
                        own = il >= lFirst && il < lEnd;
                    } else {
                        own = (firstStage() && nm == "token_embd.weight") ||
                              (lastStage() && (nm == "output.weight" ||
                                               nm == "output_norm.weight"));
                    }
                    if (own) lockT(&kv.second);
                }
                fprintf(stderr, "qk_open: stage mlock wired %.1f GB\n", wired / 1e9);
            }
        }
    }
    // The 80B file (42.8 GB) exceeds maxBufferLength (~36 GB on this device),
    // so one no-copy buffer over the whole mmap is impossible. Split the
    // mapping into page-aligned no-copy WINDOWS cut at tensor boundaries and
    // resolve each tensor to (window buffer, offset).
    if (!gbuf && !gwin.empty()) {}  // (windows built below when needed)
    auto WOFF = [&](const GgufTensor* t) -> WB {
        if (gbuf) return WB{gbuf, (NSUInteger)((const uint8_t*)t->data - g.base())};
        // windows are page-padded and can overlap at the seams — pick one that
        // contains the ENTIRE tensor, not just its first byte
        const size_t bytes = ggmlRowBytes((GgmlType)t->type, (uint32_t)t->ne[0]) *
                             t->ne[1] * t->ne[2] * t->ne[3];
        for (const auto& w : gwin)
            if (t->data >= w.base && t->data + bytes <= w.base + w.len)
                return WB{w.buf, (NSUInteger)(t->data - w.base)};
        fprintf(stderr, "qk_open: tensor outside every weight window\n");
        abort();
    };
    const GgufTensor* tEmbd = g.find("token_embd.weight");
    const GgufTensor* tONorm = g.find("output_norm.weight");
    const GgufTensor* tHead = g.find("output.weight");
    if (!tEmbd || !tONorm || !tHead ||
        (tEmbd->type != GGML_Q6_K && tEmbd->type != GGML_Q8_0) || tHead->type != GGML_Q6_K)
        return fail("qk_open: missing/unexpected embd/head tensors");
    embQ8 = tEmbd->type == GGML_Q8_0;
    vocab = (uint32_t)tHead->ne[1];
    {   // routed-expert counts from the first owned layer (uniform per file)
        char nb0[96];
        snprintf(nb0, sizeof nb0, "blk.%u.ffn_gate_inp.weight", lFirst);
        const GgufTensor* tgi0 = g.find(nb0);
        snprintf(nb0, sizeof nb0, "blk.%u.ffn_gate_exps.weight", lFirst);
        const GgufTensor* tge0 = g.find(nb0);
        if (!tgi0 || !tge0) return fail("qk_open: missing router tensors");
        nExp = (uint32_t)tgi0->ne[1];
        ffE = (uint32_t)tge0->ne[1];
        if (nExp < nUsed || nExp > 512 || ffE % 256 != 0)
            return fail("qk_open: expert shape unsupported (n_expert <= 512, n_ff_exp %% 256)");
        guIq4 = tge0->type == GGML_IQ4_XS;   // uniform per file (loadMoeT re-checks per layer)
    }
    const size_t rbQ8e = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbQ8i = ggmlRowBytes(GGML_Q8_0, dIn);
    const size_t rbE = ggmlRowBytes(GGML_Q6_K, nEmbd);
    const uint32_t nB = nSlots, tmax = nCtx;
    nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;

    pRms   = getPipe(c, "rmsnorm", "rmsnorm", 0);
    pGemvA = getPipe(c, "gemv_q8_0", "gemv_q8_0", 64);
    pGemvO = getPipe(c, "gemv_q8_0", "gemv_q8_0", 128);
    // Q8_0 dense projections can decode each weight block once for a group of
    // slots. The 80B repack uses IQ4_XS dense weights and intentionally stays
    // on its existing path. QK_SLOT_BATCH=0 is the benchmark/rollback switch.
    if (!guIq4) {
        slotBatchN = nSlots % 8u == 0u ? 8u : nSlots % 4u == 0u ? 4u
                                                    : nSlots % 2u == 0u ? 2u : 0u;
        slotBatch = slotBatchN != 0;
    }
    if (const char* sb = getenv("QK_SLOT_BATCH")) slotBatch = atoi(sb) != 0;
    if (slotBatch) {
        auto readTpr = [](const char* name, uint32_t dflt) {
            const char* v = getenv(name);
            uint32_t x = v ? (uint32_t)atoi(v) : dflt;
            return x == 32 || x == 64 || x == 128 ? x : dflt;
        };
        slotTprA = readTpr("QK_SLOT_TPR_A", slotTprA);
        slotTprO = readTpr("QK_SLOT_TPR_O", slotTprO);
        if (const char* v = getenv("QK_SLOT_B")) {
            uint32_t x = (uint32_t)atoi(v);
            if ((x == 2 || x == 4 || x == 8) && nSlots % x == 0) slotBatchN = x;
        }
        if (!slotBatchN) slotBatch = false;
    }
    if (slotBatch) {
        char fn[32];
        snprintf(fn, sizeof fn, "gemv_q8_0_b%u", slotBatchN);
        pGemvAB = getPipe(c, "gemv_q8_0_batch", fn, slotTprA);
        pGemvOB = getPipe(c, "gemv_q8_0_batch", fn, slotTprO);
    }
    pAb    = getPipe(c, "dn_ab", "dn_ab", nsg);
    pStep  = getPipe(c, "dn_step", "dn_step", 0);
    pPrep  = getPipe(c, "fa_srv", "fa_prep_srv", 0);
    pAttn  = getPipe(c, "fa_srv", "fa_attn_srv", 0);
    pAttnSplit  = getPipe(c, "fa_srv", "fa_attn_srv_split", 0);   // flash-decoding
    pAttnReduce = getPipe(c, "fa_srv", "fa_attn_srv_reduce", 0);
    pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    pMoeLA = getPipe(c, "moe_logits", "moe_logits_addn", nsg);
    pMoeGu = getPipe(c, "moe_gateup_all", "moe_gateup_all", nsg);
    pMoeD4 = getPipe(c, "moe_down_all", "moe_down_all_iq4", nsg);
    pMoeD6 = getPipe(c, "moe_down_all", "moe_down_all_q6k", nsg);
    pAddN  = getPipe(c, "add_rmsnorm", "add_rmsnorm", 0);
    pHead  = getPipe(c, "gemv_q6_k", "gemv_q6_k", 2);
    pAm1   = getPipe(c, "argmax", "argmax1", 0);
    pAm2   = getPipe(c, "argmax", "argmax2", 0);
    pEmb   = getPipe(c, embQ8 ? "embed_q8_0" : "embed_q6k", embQ8 ? "embed_q8_0" : "embed_q6k", 0);
    pPrepB = getPipe(c, "fa_batch", "fa_prep_batch", 0);
    pAttnB = getPipe(c, "fa_batch", "fa_attn_batch", 0);
    pAttnBM = getPipe(c, "fa_batch", "fa_attn_batch_mma", 0);
    pAbB   = getPipe(c, "dn_batch", "dn_ab_batch", nsg);
    pConvB = getPipe(c, "dn_batch", "dn_conv_batch", 0);
    pStepB = getPipe(c, "dn_batch", "dn_step_batch", 0);
    pGateB = getPipe(c, "dn_batch", "dn_gate_batch", nsg);
    if (const char* v = getenv("QK_DN_CHUNK")) dnChunk = atoi(v) != 0;
    if (dnChunk) {
        pDnKq    = getPipe(c, "dn_chunk", "dn_chunk_kq", 0);
        pDnSolve = getPipe(c, "dn_chunk", "dn_chunk_solve", 0);
        pDnStepC = getPipe(c, "dn_chunk", "dn_chunk_step", dS);   // fc(0)=dS
    }
    {   // QK_GEMM=scalar|sg|h picks the prefill GEMM (default: exact scalar).
        // Default: the f16-fragment GEMM (llama.cpp Metal's prefill precision
        // class; accepted via prefillcmp 36/36 argmax + prefilldecode HANDOFF
        // EXACT). QK_GEMM=scalar forces the bit-exact f32 path.
        const char* gv = getenv("QK_GEMM");
        const char* fn = "gemm_q8_0_hp";
        if (gv && !strcmp(gv, "scalar")) fn = "gemm_q8_0";
        if (gv && !strcmp(gv, "h")) fn = "gemm_q8_0_h";
        if (gv && !strcmp(gv, "sg")) fn = "gemm_q8_0_sg";
        if (gv && !strcmp(gv, "h2")) fn = "gemm_q8_0_h2";
        if (gv && !strcmp(gv, "hp")) fn = "gemm_q8_0_hp";
        pGemmB = getPipe(c, "gemm_q8_0", fn, 0);
        gemmThreads = !strcmp(fn, "gemm_q8_0_sg") ? 128 : 256;  // sg is 4-simd; scalar+h are 256
        if (!strcmp(fn, "gemm_q8_0_h2")) { gemmBM = 64; gemmBN = 128; }
        if (!strcmp(fn, "gemm_q8_0_hp")) { gemmBM = 64; gemmBN = 32; gemmThreads = 128; }
    }
    pMoeGrp  = getPipe(c, "moe_grouped", "moe_group", 0);
    pMoeGrpLive = getPipe(c, "moe_grouped", "moe_group_live", 0);
    pMoeGuG  = getPipe(c, "moe_grouped", "moe_gu_grouped", nsg);
    pMoeGuG2 = getPipe(c, "moe_grouped", "moe_gu_grouped2", 0);
    pMoeGuG3 = getPipe(c, "moe_grouped", "moe_gu_grouped3", 0);
    pMoeGuG4 = getPipe(c, "moe_grouped", "moe_gu_grouped4", 0);
    pMoeGuG4Live = getPipe(c, "moe_grouped", "moe_gu_grouped4_live", 0);
    pMoeGuG5 = getPipe(c, "moe_grouped", "moe_gu_grouped5", 0);
    pMoeGuG5Live = getPipe(c, "moe_grouped", "moe_gu_grouped5_live", 0);
    if (guIq4) {
        pGemv4    = getPipe(c, "gemv_iq4_xs", "gemv_iq4_xs", 2);   // NSG=2: 64 thr, 4 rows/tg
        pGemmB4   = getPipe(c, "gemm_iq4_xs", "gemm_iq4_xs_hp", 0);
        pMoeGu4   = getPipe(c, "moe_gateup_all", "moe_gateup_all_iq4", nsg);
        pMoeGuG4i = getPipe(c, "moe_grouped", "moe_gu_grouped4_iq4", 0);
        pMoeGuG5i = getPipe(c, "moe_grouped", "moe_gu_grouped5_iq4", 0);
    }
    pMoeDG4  = getPipe(c, "moe_down_grouped", "moe_down_grouped_iq4", 0);
    pMoeDG6  = getPipe(c, "moe_down_grouped", "moe_down_grouped_q6k", 0);
    pMoeDG4Live = getPipe(c, "moe_down_grouped", "moe_down_grouped_live_iq4", 0);
    pMoeDG6Live = getPipe(c, "moe_down_grouped", "moe_down_grouped_live_q6k", 0);
    pMoeDGH4 = getPipe(c, "moe_down_grouped", "moe_down_grouped_h_iq4", 0);
    pMoeDGH6 = getPipe(c, "moe_down_grouped", "moe_down_grouped_h_q6k", 0);
    pMoeDGP4 = getPipe(c, "moe_down_grouped", "moe_down_grouped_p_iq4", 0);
    pMoeDGP6 = getPipe(c, "moe_down_grouped", "moe_down_grouped_p_q6k", 0);
    pMoeDGP4Live = getPipe(c, "moe_down_grouped", "moe_down_grouped_p_live_iq4", 0);
    pMoeDGP6Live = getPipe(c, "moe_down_grouped", "moe_down_grouped_p_live_q6k", 0);
    pMoeDR   = getPipe(c, "moe_down_grouped", "moe_down_reduce", 0);
    pLogG    = getPipe(c, "moe_logits", "moe_logits_gemm", 0);
    if (const char* mg = getenv("QK_MOE_GROUPED")) { moeGrouped = atoi(mg); moeGroupN = 0; }
    if (const char* mn = getenv("QK_MOE_GROUP_N")) moeGroupN = (uint32_t)atoi(mn);
    // The compact decode gate/up kernels currently implement the 35B IQ3
    // routed layout only. Keep other 256-expert layouts on their native path.
    const bool slotGroupedFormat = nExp == 256 && !guIq4;
    moeSlotGrouped = slotGroupedFormat ? 4 : 0;
    if (const char* ms = getenv("QK_MOE_SLOT_GROUPED")) {
        int v = atoi(ms);
        if (v >= 0 && v <= 5 && (v == 0 || slotGroupedFormat)) moeSlotGrouped = v;
        else fprintf(stderr,
            "QK_MOE_SLOT_GROUPED=%d ignored (requires 0..5 and 256-expert IQ3 gate/up)\n", v);
    }
    if (const char* mp = getenv("QK_MOE_SLOT_PARTS")) moeSlotParts = atoi(mp) & 3;
    if (const char* ml = getenv("QK_MOE_SLOT_LIVE")) moeSlotLive = atoi(ml) != 0;
    if (const char* mm = getenv("QK_MOE_SLOT_MIN")) {
        uint32_t v = (uint32_t)atoi(mm);
        if (v >= 1) moeSlotMin = v;
    }
    if (const char* mf = getenv("QK_MOE_SLOT_FORCE")) moeSlotSameOnly = atoi(mf) == 0;
    static_assert(dS <= 128, "dn_step_batch srow[32] holds dState/4 float4s");

    bXin = createBuf(c, (size_t)nB * nEmbd * 4, nullptr, true);
    bXn = createBuf(c, (size_t)nB * nEmbd * 4, nullptr, true);
    bBig = createBuf(c, (size_t)nB * chQkv * 4, nullptr, true);
    bMid = createBuf(c, (size_t)nB * dIn * 4, nullptr, true);
    bKin = createBuf(c, (size_t)nB * hKV * dh * 4, nullptr, true);
    bVin = createBuf(c, (size_t)nB * hKV * dh * 4, nullptr, true);
    bGb = createBuf(c, (size_t)nB * 2 * hV * 4, nullptr, true);
    bAtt = createBuf(c, (size_t)nB * dIn * 4, nullptr, true);
    bAttnOut = createBuf(c, (size_t)nB * nEmbd * 4, nullptr, true);
    bY = createBuf(c, (size_t)nB * nEmbd * 4, nullptr, true);
    bXn2 = createBuf(c, (size_t)nB * nEmbd * 4, nullptr, true);
    bML = createBuf(c, (size_t)nB * (nExp + 1) * 4, nullptr, true);
    bMH = createBuf(c, (size_t)nB * (nUsed + 1) * ffE * 4, nullptr, true);
    bMSel = createBuf(c, (size_t)nB * 160, nullptr, true);   // sizeof(SelT)
    bMY = createBuf(c, (size_t)nB * nEmbd * 4, nullptr, true);
    if (lastStage()) {
        bONorm = WOFF(tONorm);
    } else {
        // dummy boundary norm: the stage's last add_rmsnorm tail still writes
        // an xn, but it is dead — the next stage re-norms with its own layer's
        // attn_norm. Zeroed weights keep the dead lane finite.
        id<MTLBuffer> z = createBuf(c, nEmbd * 4, nullptr, true);
        memset(z.contents, 0, nEmbd * 4);
        bONorm = WB{z, 0};
    }
    bHeadW = WOFF(tHead);
    bLogits = splitStage() ? nil : createBuf(c, (size_t)nB * vocab * 4, nullptr, true);
    bEmbdW = WOFF(tEmbd);
    bRope = createBuf(c, (size_t)tmax * (nRot / 2) * 2 * 4, nullptr, true);
    bAV = createBuf(c, (size_t)nB * 64 * 4, nullptr, true);
    bAI = createBuf(c, (size_t)nB * 64 * 4, nullptr, true);
    bTok = createBuf(c, (size_t)nB * 4, nullptr, true);
    bRbScratch = createBuf(c, 64, nullptr, true);
    bSlotIn = createBuf(c, (size_t)nB * 4, nullptr, true);
    bSlotPos = createBuf(c, (size_t)nB * 4, nullptr, true);
    // split-K decode-attention partials: [slot][head][chunk](acc[dh], m, l)
    { const uint32_t maxChunks = (nCtx + 255u) / 256u;
      bPartial = createBuf(c, (size_t)nB * hQ * maxChunks * (dh + 2) * 4, nullptr, true); }
    memset(bSlotIn.contents, 0, (size_t)nB * 4);
    memset(bSlotPos.contents, 0, (size_t)nB * 4);
    {
        const uint32_t half = nRot / 2;
        float* rope = (float*)bRope.contents;
        for (uint32_t p = 0; p < tmax; p++)
            for (uint32_t j = 0; j < half; j++) {
                float th = (float)p * std::pow(kFreqBase, -2.f * (float)j / (float)nRot);
                rope[2 * ((size_t)p * half + j)] = std::cos(th);
                rope[2 * ((size_t)p * half + j) + 1] = std::sin(th);
            }
    }

    uint32_t cap = 128;
    if (const char* v = getenv("QK_MAXB")) {
        long x = atol(v);
        if (x >= 16 && x <= 1024) cap = (uint32_t)x;
    }
    maxB = cap;
    bbXin = createBuf(c, (size_t)cap * nEmbd * 4, nullptr, true);
    bbXn = createBuf(c, (size_t)cap * nEmbd * 4, nullptr, true);
    bbBig = createBuf(c, (size_t)cap * chQkv * 4, nullptr, true);
    bbMid = createBuf(c, (size_t)cap * dIn * 4, nullptr, true);
    bbKin = createBuf(c, (size_t)cap * hKV * dh * 4, nullptr, true);
    bbVin = createBuf(c, (size_t)cap * hKV * dh * 4, nullptr, true);
    bbGb = createBuf(c, (size_t)cap * 2 * hV * 4, nullptr, true);
    bbConvOut = createBuf(c, (size_t)cap * chQkv * 4, nullptr, true);
    bbO = createBuf(c, (size_t)((cap + 63) / 64 * 64) * dIn * 4, nullptr, true);
    bbAtt = createBuf(c, (size_t)cap * dIn * 4, nullptr, true);
    bbAttnOut = createBuf(c, (size_t)cap * nEmbd * 4, nullptr, true);
    bbY = createBuf(c, (size_t)cap * nEmbd * 4, nullptr, true);
    bbXn2 = createBuf(c, (size_t)cap * nEmbd * 4, nullptr, true);
    bbML = createBuf(c, (size_t)cap * (nExp + 1) * 4, nullptr, true);
    bbMH = createBuf(c, (size_t)cap * (nUsed + 1) * ffE * 4, nullptr, true);
    bbMSel = createBuf(c, (size_t)cap * 160, nullptr, true);   // sizeof(SelT)
    bbMY = createBuf(c, (size_t)cap * nEmbd * 4, nullptr, true);
    bbLogits = lastStage() ? createBuf(c, (size_t)cap * vocab * 4, nullptr, true) : nil;
    if (lastStage()) {
        bbAV = createBuf(c, (size_t)cap * 64 * 4, nullptr, true);
        bbAI = createBuf(c, (size_t)cap * 64 * 4, nullptr, true);
        bbTok = createBuf(c, (size_t)cap * 4, nullptr, true);
    }
    bbIds = createBuf(c, (size_t)cap * 4, nullptr, true);
    bbStart = createBuf(c, (size_t)(nExp + 2) * 4, nullptr, true);
    bbATok = createBuf(c, (size_t)cap * (nUsed + 1) * 4, nullptr, true);
    bbASlot = createBuf(c, (size_t)cap * (nUsed + 1) * 4, nullptr, true);
    bbLive = createBuf(c, ((size_t)cap * nUsed + 1) * 4, nullptr, true);
    bbMDy = createBuf(c, (size_t)cap * (nUsed + 1) * nEmbd * 4, nullptr, true);
    bbCarry = createBuf(c, (size_t)nLayer * chQkv * 3 * 4, nullptr, true);
    memset(bbCarry.contents, 0, (size_t)nLayer * chQkv * 3 * 4);
    if (dnChunk) {   // chunked-DN scratch: per (chunk, head) tiles, reused per layer
        const size_t nChMax = (cap + 63) / 64;
        bbDnKQ  = createBuf(c, nChMax * hK * 2 * 64 * 64 * 4, nullptr, true);
        bbDnUW  = createBuf(c, nChMax * hV * 4 * 64 * (size_t)dS * 4, nullptr, true);
        bbDnAtt = createBuf(c, nChMax * hV * 64 * 64 * 4, nullptr, true);
        bbDnEl  = createBuf(c, nChMax * hV * 4, nullptr, true);
    }

    layers.resize(nLayer);
    char nb[128];
    for (uint32_t il = lFirst; il < lEnd; il++) {
        Layer& L = layers[il];
        auto T = [&](const char* suffix) -> const GgufTensor* {
            snprintf(nb, sizeof nb, "blk.%u.%s", il, suffix); return g.find(nb);
        };
        auto W = [&](const GgufTensor* t, size_t) -> WB { return WOFF(t); };
        // Dense projection: Q8_0 (35B) or IQ4_XS (80B; attn_v stays Q8_0
        // there). Missing tensor or other type is a hard error.
        bool denseBad = false;
        auto Wd = [&](const char* suffix, bool& iq4) -> WB {
            const GgufTensor* t = T(suffix);
            if (!t || (t->type != GGML_Q8_0 && t->type != GGML_IQ4_XS)) {
                fprintf(stderr, "blk.%u.%s: missing or unsupported dense type\n", il, suffix);
                denseBad = true;
                iq4 = false;
                return WB{gbuf, 0};
            }
            iq4 = t->type == GGML_IQ4_XS;
            return WOFF(t);
        };
        MoeT moe;
        if (!loadMoeT(g, il, moe)) return fail("qk_open: MoE tensors missing");
        L.downQ6 = moe.downQ6;
        L.rec = T("ssm_a") != nullptr;
        L.aNorm = W(T("attn_norm.weight"), nEmbd * 4);
        L.pn = W(T("post_attention_norm.weight"), nEmbd * 4);
        if (L.rec) {
            L.qkvW = Wd("attn_qkv.weight", L.iq4P1);
            L.zW = Wd("attn_gate.weight", L.iq4P2);
            if (T("ssm_alpha.weight")) {
                L.alW = W(T("ssm_alpha.weight"), (size_t)hV * nEmbd * 4);
                L.beW = W(T("ssm_beta.weight"), (size_t)hV * nEmbd * 4);
            } else {
                // qwen3next fuses the two: ssm_ba [nEmbd, 2*hV], interleaved
                // per k-head group g: rows g*4+{0,1} = beta (v-heads 2g,2g+1),
                // rows g*4+{2,3} = alpha (llama.cpp qwen3next view split).
                // Tiny tensor — dequant on the host, de-interleave into the
                // engine's split alpha/beta layout; dn_ab is unchanged.
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
                extraBufs.push_back(createBuf(c, al.size() * 4, al.data(), true));
                L.alW = WB{extraBufs.back(), 0};
                extraBufs.push_back(createBuf(c, be.size() * 4, be.data(), true));
                L.beW = WB{extraBufs.back(), 0};
            }
            L.dt = W(T("ssm_dt.bias"), hV * 4);
            L.av = W(T("ssm_a"), hV * 4);
            L.ker = W(T("ssm_conv1d.weight") ? T("ssm_conv1d.weight") : T("ssm_conv1d"),
                      (size_t)chQkv * 4 * 4);
            L.sn = W(T("ssm_norm.weight"), dS * 4);
            L.outW = Wd("ssm_out.weight", L.iq4Wo);
            L.ps1 = (size_t)chQkv * 3 * 4;
            L.ps2 = (size_t)hV * dS * dS * 4;
        } else {
            L.wq = Wd("attn_q.weight", L.iq4P1);
            L.wk = Wd("attn_k.weight", L.iq4P2);
            L.wv = Wd("attn_v.weight", L.iq4P3);
            L.qn = W(T("attn_q_norm.weight"), dh * 4);
            L.kn = W(T("attn_k_norm.weight"), dh * 4);
            L.wo = Wd("attn_output.weight", L.iq4Wo);
            L.ps1 = (size_t)hKV * tmax * dh * 4;
            L.ps2 = (size_t)hKV * tmax * dh * 4;
        }
        // +slack: fa_attn_batch_mma reads full 8-key MMA tiles, so at near-full
        // context (nk ~ tmax) the last slot's K/V load over-reads a few rows
        // past the buffer. One KBM(64)-key block of zeroed slack keeps it
        // in-bounds and finite (over-read rows are P=0, so contribute nothing).
        const size_t stSlack = (size_t)64 * dh * 4;
        // Recurrent layers reserve one shared spec-verification scratch stripe
        // at index nSlots. Attention has rollback-free positional KV and needs
        // only the live slot stripes.
        const size_t nStateStripes = nB + (L.rec ? 1u : 0u);
        L.st1 = createBuf(c, nStateStripes * L.ps1 + stSlack, nullptr, true);
        L.st2 = createBuf(c, nStateStripes * L.ps2 + stSlack, nullptr, true);
        memset(L.st1.contents, 0, nStateStripes * L.ps1 + stSlack);
        memset(L.st2.contents, 0, nStateStripes * L.ps2 + stSlack);
        L.mgi = W(moe.gi, (size_t)moe.nExp * nEmbd * 4);
        L.mgis = W(moe.gis, nEmbd * 4);
        L.mge = W(moe.ge, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        L.mue = W(moe.ue, (size_t)moe.nExp * moe.nFf * moe.rbGE);
        L.mde = W(moe.de, (size_t)moe.nExp * moe.nEmbd * moe.rbDE);
        L.mgs = W(moe.gs, (size_t)moe.nFf * moe.rbGS);
        L.mus = W(moe.us, (size_t)moe.nFf * moe.rbGS);
        L.mds = W(moe.ds, (size_t)moe.nEmbd * moe.rbDS);
        if (denseBad) return fail("qk_open: dense projection tensor missing or bad type");
    }

    snapOff1.resize(nLayer);
    snapOff2.resize(nLayer);
    size_t off = 0;
    for (uint32_t il = 0; il < nLayer; il++) {
        snapOff1[il] = off; off += layers[il].ps1;
        snapOff2[il] = off; off += layers[il].ps2;
    }
    snapSize = off;
    uint32_t PCACHE_N = 3;
    if (const char* pcn = getenv("QK_PCACHE")) {
        long v = strtol(pcn, nullptr, 10);
        if (v >= 1 && v <= 256) PCACHE_N = (uint32_t)v;
    }
    pcache.resize(PCACHE_N);
    for (auto& e : pcache) e.snap.resize(snapSize);

    specOn = getenv("QK_SPEC") != nullptr;
    if (const char* v = getenv("QK_SPEC_L")) {
        long x = atol(v);
        if (x >= 2 && x <= 64) specL = (uint32_t)x;
    }
    if (const char* v = getenv("QK_SPEC_K")) {
        long x = atol(v);
        // The default hp projection path has a demonstrated K=64 near-tie;
        // cap the exact serving policy at 32. Wider scalar experiments remain
        // available through the standalone verify harness.
        if (x >= 2 && (uint32_t)x <= std::min(maxB, 32u)) specK = (uint32_t)x;
        else fprintf(stderr, "QK_SPEC_K=%ld ignored (exact range 2..%u)\n",
                     x, std::min(maxB, 32u));
    }
    specToks.resize(maxB);
    specAm.resize(maxB);
    slots.resize(nSlots);
    return true;
}

// Encode one serial decode/prefill step for slots [0, zdim): embed(+L0 norm)
// -> 40 layers (srv attention) -> head -> argmax into bTok.
void qk_engine::encodeStep(id<MTLComputeCommandEncoder> enc, uint32_t zdim,
                           bool equivalentSlots) {
    auto dsp = [&](id<MTLComputePipelineState> pso,
                   std::initializer_list<WB> bufs,
                   const void* pc, uint32_t pcSize, uint32_t wgs, uint32_t thr,
                   uint32_t gridZ = 0) {
        [enc setComputePipelineState:pso];
        uint32_t i = 0;
        for (const WB& b : bufs) [enc setBuffer:b.b offset:b.o atIndex:i++];
        [enc setBytes:pc length:pcSize atIndex:i];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, gridZ ? gridZ : zdim)
            threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
    };
    auto bar = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    // split-K (flash-decoding) decode attention. QK_FA_SPLIT=1 enables it;
    // dspY adds the chunk (y) grid axis the plain dsp lacks.
    static const bool faSplit = getenv("QK_FA_SPLIT") != nullptr;
    auto dspY = [&](id<MTLComputePipelineState> pso, std::initializer_list<WB> bufs,
                    const void* pc, uint32_t pcSize, uint32_t wgs, uint32_t ydim,
                    uint32_t thr) {
        [enc setComputePipelineState:pso];
        uint32_t i = 0;
        for (const WB& b : bufs) [enc setBuffer:b.b offset:b.o atIndex:i++];
        [enc setBytes:pc length:pcSize atIndex:i];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, ydim, zdim)
            threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
    };
    const uint32_t thrN = nsg * 32;
    const bool slotB = slotBatch && zdim >= slotBatchN && zdim % slotBatchN == 0;
    auto bwgs = [](uint32_t rows, uint32_t tpr) {
        const uint32_t rowsPerTg = 256u / tpr;
        return (rows + rowsPerTg - 1u) / rowsPerTg;
    };

    struct { uint32_t n; float e; } pcRms{nEmbd, eps};
    struct { uint32_t m, k; } pcQkv{chQkv, nEmbd}, pcZ{dIn, nEmbd}, pcKV{hKV * dh, nEmbd},
        pcWo{nEmbd, dIn}, pcHead{vocab, nEmbd};
    struct { uint32_t n, h; } pcAb{nEmbd, hV};
    struct { uint32_t d, hk, hv; float e; uint32_t kd; } pcStep{dS, hK, hV, eps, dnKDiv};
    struct { uint32_t a, b, cc, d; } pcv{nEmbd, ffE, nExp, nUsed};
    struct { uint32_t a, b, cc, d; float e; } pcv5{nEmbd, ffE, nExp, nUsed, eps};
    struct { uint32_t pos, tmax, dh_, nRot_, hQ_, hKV_; float e, fb; }
        pcFa{0, nCtx, dh, nRot, hQ, hKV, eps, kFreqBase};
    // split-K: dispatch only the chunks the deepest slot actually needs
    // (over-length chunk-TGs early-return); partial stride is the full ceil(ctx/256).
    uint32_t maxPos = 0;
    { const uint32_t* sp = (const uint32_t*)bSlotPos.contents;
      for (uint32_t s = 0; s < zdim; ++s) maxPos = std::max(maxPos, sp[s]); }
    const uint32_t splitChunks = (maxPos + 1u + 255u) / 256u;
    struct { uint32_t pos, tmax, dh_, nRot_, hQ_, hKV_; float e, fb; uint32_t maxChunks; }
        pcFaSplit{0, nCtx, dh, nRot, hQ, hKV, eps, kFreqBase, (nCtx + 255u) / 256u};
    const uint32_t amWgs = (vocab + 4095) / 4096;
    struct { uint32_t n, span; } pcAm{vocab, 4096};
    struct { uint32_t m, pos; } pcAm2{amWgs, 0};
    struct { uint32_t k, idx, pr; float e; } pcE{nEmbd, 0, 1, eps};

    dsp(pEmb, {bEmbdW, bSlotIn, bXin, layers[0].aNorm, bXn}, &pcE, 16, 1, 256);
    bar();
    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        if (L.rec) {
            if (L.iq4P1) dsp(pGemv4, {L.qkvW, bXn, bBig}, &pcQkv, 8, chQkv / 4, 64);
            else         dsp(slotB ? pGemvAB : pGemvA, {L.qkvW, bXn, bBig}, &pcQkv,
                             8, slotB ? bwgs(chQkv, slotTprA) : chQkv / 4, 256,
                             slotB ? zdim / slotBatchN : 0);
            if (L.iq4P2) dsp(pGemv4, {L.zW, bXn, bMid}, &pcZ, 8, dIn / 4, 64);
            else         dsp(slotB ? pGemvAB : pGemvA, {L.zW, bXn, bMid}, &pcZ,
                             8, slotB ? bwgs(dIn, slotTprA) : dIn / 4, 256,
                             slotB ? zdim / slotBatchN : 0);
            dsp(pAb, {bXn, L.alW, L.beW, L.dt, L.av, bGb}, &pcAb, 8,
                (2 * hV + nsg - 1) / nsg, thrN);
            bar();
            dsp(pStep, {bBig, L.st1, L.ker, bGb, L.st2, bMid, L.sn, bAtt},
                &pcStep, 20, hK, dS);
            bar();
            if (L.iq4Wo) dsp(pGemv4, {L.outW, bAtt, bAttnOut}, &pcWo, 8, nEmbd / 4, 64);
            else         dsp(slotB ? pGemvOB : pGemvO, {L.outW, bAtt, bAttnOut}, &pcWo,
                             8, slotB ? bwgs(nEmbd, slotTprO) : nEmbd / 2, 256,
                             slotB ? zdim / slotBatchN : 0);
        } else {
            if (L.iq4P1) dsp(pGemv4, {L.wq, bXn, bBig}, &pcQkv, 8, chQkv / 4, 64);
            else         dsp(slotB ? pGemvAB : pGemvA, {L.wq, bXn, bBig}, &pcQkv,
                             8, slotB ? bwgs(chQkv, slotTprA) : chQkv / 4, 256,
                             slotB ? zdim / slotBatchN : 0);
            if (L.iq4P2) dsp(pGemv4, {L.wk, bXn, bKin}, &pcKV, 8, hKV * dh / 4, 64);
            else         dsp(slotB ? pGemvAB : pGemvA, {L.wk, bXn, bKin}, &pcKV,
                             8, slotB ? bwgs(hKV * dh, slotTprA) : hKV * dh / 4, 256,
                             slotB ? zdim / slotBatchN : 0);
            if (L.iq4P3) dsp(pGemv4, {L.wv, bXn, bVin}, &pcKV, 8, hKV * dh / 4, 64);
            else         dsp(slotB ? pGemvAB : pGemvA, {L.wv, bXn, bVin}, &pcKV,
                             8, slotB ? bwgs(hKV * dh, slotTprA) : hKV * dh / 4, 256,
                             slotB ? zdim / slotBatchN : 0);
            bar();
            dsp(pPrep, {bBig, bKin, bVin, L.qn, L.kn, bMid, L.st1, L.st2, bRope, bSlotPos},
                &pcFa, 32, hQ + 2 * hKV, 256);
            bar();
            if (faSplit) {
                dspY(pAttnSplit, {bMid, L.st1, L.st2, bPartial, bSlotPos},
                     &pcFaSplit, 36, hQ, splitChunks, 256);
                bar();
                dsp(pAttnReduce, {bPartial, bBig, bAtt, bSlotPos}, &pcFaSplit, 36, hQ, 256);
            } else {
                dsp(pAttn, {bMid, L.st1, L.st2, bBig, bAtt, bSlotPos}, &pcFa, 32, hQ, 256);
            }
            bar();
            if (L.iq4Wo) dsp(pGemv4, {L.wo, bAtt, bAttnOut}, &pcWo, 8, nEmbd / 4, 64);
            else         dsp(slotB ? pGemvOB : pGemvO, {L.wo, bAtt, bAttnOut}, &pcWo,
                             8, slotB ? bwgs(nEmbd, slotTprO) : nEmbd / 2, 256,
                             slotB ? zdim / slotBatchN : 0);
        }
        bar();
        dsp(pMoeLA, {L.mgi, L.mgis, bXin, bAttnOut, L.pn, bML, bY, bXn2}, &pcv5, 20,
            (nExp + 1 + nsg - 1) / nsg, thrN);
        bar();
        dsp(pMoeS, {bML, bMSel}, &pcv, 16, 1, 32);
        bar();
        const bool slotGrouped = moeSlotGrouped && moeSlotParts && zdim >= moeSlotMin &&
            (!moeSlotSameOnly || equivalentSlots);
        const bool compactLive = slotGrouped &&
            (moeSlotGrouped == 4 || moeSlotGrouped == 5) && moeSlotLive;
        const uint32_t maxLive = equivalentSlots ? nUsed + 1u : zdim * nUsed + 1u;
        if (slotGrouped) {
            struct { uint32_t a, b, cc, d, n; } pcg{nEmbd, ffE, nExp, nUsed, zdim};
            if (compactLive)
                dsp(pMoeGrpLive, {bMSel, bbStart, bbATok, bbASlot, bbLive},
                    &pcg, 20, 1, 256, 1);
            else
                dsp(pMoeGrp, {bMSel, bbStart, bbATok, bbASlot},
                    &pcg, 20, 1, 256, 1);
            bar();
        }
        if (slotGrouped && (moeSlotParts & 1)) {
            if (moeSlotGrouped == 5) {
                if (compactLive)
                    dsp(pMoeGuG5Live,
                        {L.mge, L.mue, L.mgs, L.mus, bXn2, bbStart, bbATok,
                         bbASlot, bMH, bbLive}, &pcv, 16,
                        maxLive * (ffE / 32), 128, (zdim + 31) / 32);
                else
                    dsp(pMoeGuG5,
                        {L.mge, L.mue, L.mgs, L.mus, bXn2, bbStart, bbATok,
                         bbASlot, bMH}, &pcv, 16,
                        (nExp + 1) * (ffE / 32), 128, (zdim + 31) / 32);
            }
            else if (moeSlotGrouped == 4) {
                if (compactLive)
                    dsp(pMoeGuG4Live,
                        {L.mge, L.mue, L.mgs, L.mus, bXn2, bbStart, bbATok,
                         bbASlot, bMH, bbLive}, &pcv, 16,
                        maxLive * (ffE / 32), 128, 1);
                else
                    dsp(pMoeGuG4,
                        {L.mge, L.mue, L.mgs, L.mus, bXn2, bbStart, bbATok,
                         bbASlot, bMH}, &pcv, 16,
                        (nExp + 1) * (ffE / 32), 128, 1);
            }
            else if (moeSlotGrouped == 3)
                dsp(pMoeGuG3, {L.mge, L.mue, L.mgs, L.mus, bXn2, bbStart, bbATok,
                                bbASlot, bMH}, &pcv, 16,
                    (nExp + 1) * (ffE / 64), 256, 1);
            else if (moeSlotGrouped == 2)
                dsp(pMoeGuG2, {L.mge, L.mue, L.mgs, L.mus, bXn2, bbStart, bbATok,
                                bbASlot, bMH}, &pcv, 16,
                    (nExp + 1) * (ffE / 32), 128, 1);
            else
                dsp(pMoeGuG, {L.mge, L.mue, L.mgs, L.mus, bXn2, bbStart, bbATok,
                               bbASlot, bMH}, &pcv, 16,
                    ((nExp + 1) * ffE + nsg - 1) / nsg, thrN, 1);
        } else {
            dsp(guIq4 ? pMoeGu4 : pMoeGu,
                {L.mge, L.mue, L.mgs, L.mus, bXn2, bMSel, bMH}, &pcv, 16,
                ((nUsed + 1) * ffE + nsg - 1) / nsg, thrN);
        }
        bar();
        if (slotGrouped && (moeSlotParts & 2)) {
            if (moeSlotGrouped == 5) {
                if (compactLive)
                    dsp(L.downQ6 ? pMoeDGP6Live : pMoeDGP4Live,
                        {L.mde, L.mds, bMH, bbStart, bbATok, bbASlot, bbMDy, bbLive},
                        &pcv, 16, maxLive * (nEmbd / 64), 128,
                        (zdim + 31) / 32);
                else
                    dsp(L.downQ6 ? pMoeDGP6 : pMoeDGP4,
                        {L.mde, L.mds, bMH, bbStart, bbATok, bbASlot, bbMDy},
                        &pcv, 16, (nExp + 1) * (nEmbd / 64), 128,
                        (zdim + 31) / 32);
            }
            else if (moeSlotGrouped == 3)
                dsp(L.downQ6 ? pMoeDGH6 : pMoeDGH4,
                    {L.mde, L.mds, bMH, bbStart, bbATok, bbASlot, bbMDy}, &pcv, 16,
                    (nExp + 1) * (nEmbd / 64), 256, 1);
            else {
                if (compactLive)
                    dsp(L.downQ6 ? pMoeDG6Live : pMoeDG4Live,
                        {L.mde, L.mds, bMH, bbStart, bbATok, bbASlot, bbMDy, bbLive},
                        &pcv, 16, maxLive * (nEmbd / 32), 128, 1);
                else
                    dsp(L.downQ6 ? pMoeDG6 : pMoeDG4,
                        {L.mde, L.mds, bMH, bbStart, bbATok, bbASlot, bbMDy},
                        &pcv, 16, (nExp + 1) * (nEmbd / 32), 128, 1);
            }
            bar();
            dsp(pMoeDR, {bbMDy, bMSel, bMY}, &pcv, 16,
                (nEmbd + 255) / 256, 256);
        } else {
            dsp(L.downQ6 ? pMoeD6 : pMoeD4,
                {L.mde, L.mds, bMH, bMSel, bMY}, &pcv, 16,
                (nEmbd + nsg - 1) / nsg, thrN);
        }
        bar();
        WB nextNorm = il + 1 < nLayer ? layers[il + 1].aNorm : bONorm;
        dsp(pAddN, {bY, bMY, nextNorm, bXin, bXn}, &pcRms, 8, 1, 256);
        bar();
    }
    dsp(pHead, {bHeadW, bXn, bLogits}, &pcHead, 8, (vocab + 3) / 4, 64);
    bar();
    dsp(pAm1, {bLogits, bAV, bAI}, &pcAm, 8, amWgs, 256);
    bar();
    dsp(pAm2, {bAV, bAI, bTok, bRbScratch}, &pcAm2, 8, 1, 256);
}

uint32_t qk_engine::specDraft(uint32_t slot) {
    Slot& sl = slots[slot];
    const uint32_t L = specL;
    const size_t histN = sl.prompt.size() + sl.genTokens.size();
    if (histN < (size_t)L + 1) return 0;
    auto tokenAt = [&](size_t i) -> uint32_t {
        return i < sl.prompt.size() ? sl.prompt[i] : sl.genTokens[i - sl.prompt.size()];
    };
    auto gramHash = [&](size_t end) {
        uint64_t h = 1469598103934665603ull;
        for (size_t i = end - L; i < end; i++) {
            h ^= tokenAt(i);
            h *= 1099511628211ull;
        }
        return h;
    };

    // Index only grams ending before the current suffix, so it cannot match
    // itself. Hash collisions can waste a round but cannot alter output because
    // acceptance compares the actual target argmax tokens.
    if (sl.ngramBuilt < L) sl.ngramBuilt = L;
    for (size_t end = sl.ngramBuilt; end < histN; end++)
        sl.ngram[gramHash(end)] = (uint32_t)end;
    sl.ngramBuilt = (uint32_t)histN;
    auto it = sl.ngram.find(gramHash(histN));
    if (it == sl.ngram.end()) return 0;

    const uint32_t matchEnd = it->second;
    const uint32_t maxN = std::min(std::min(specK, nCtx - sl.pos), sl.maxGen - sl.gen);
    if (maxN < 2) return 0;
    const uint32_t nDraft =
        std::min<uint32_t>(maxN - 1, (uint32_t)(histN - matchEnd));
    specToks[0] = tokenAt(histN - 1);  // sampled pending token, not yet fed
    for (uint32_t i = 0; i < nDraft; i++) specToks[i + 1] = tokenAt(matchEnd + i);
    return nDraft + 1;
}

int qk_engine::stepChunk(uint32_t* outTok, uint32_t* outCnt, uint32_t* outFin) {
    for (uint32_t s = 0; s < nSlots; s++) outCnt[s] = 0;
    *outFin = 0;
    int activeAtEntry = 0;
    for (uint32_t s = 0; s < nSlots; s++) if (slots[s].active) activeAtEntry++;
    if (!activeAtEntry) return 0;

    uint32_t* slotIn = (uint32_t*)bSlotIn.contents;
    uint32_t* slotPos = (uint32_t*)bSlotPos.contents;
    const uint32_t* tok = (const uint32_t*)bTok.contents;
    static const bool noEos = getenv("QK_NO_EOS") != nullptr;

    auto finish = [&](uint32_t s) {
        Slot& sl = slots[s];
        if (sl.finPending && sl.outQHead >= sl.outQ.size()) {
            if (specOn && sl.specRounds && getenv("QK_SPEC_LOG"))
                qkStatsLine("[spec] slot=%u rounds=%u fed=%u emitted=%u serial=%u "
                            "avg_accept=%.2f\n",
                            s, sl.specRounds, sl.specFed, sl.specEmitted, sl.serialSteps,
                            (double)sl.specFed / sl.specRounds);
            sl.resetSpec();
            sl.active = false;
            *outFin |= 1u << s;
        }
    };
    auto emit = [&](uint32_t s, uint32_t t) {
        if (outCnt[s] < chunkN) outTok[s * chunkN + outCnt[s]++] = t;
        else slots[s].outQ.push_back(t);
    };

    // Drain tokens already verified in a previous call. No GPU work is needed.
    for (uint32_t s = 0; s < nSlots; s++) {
        Slot& sl = slots[s];
        if (!sl.active) continue;
        while (sl.outQHead < sl.outQ.size() && outCnt[s] < chunkN)
            outTok[s * chunkN + outCnt[s]++] = sl.outQ[sl.outQHead++];
        if (sl.outQHead >= sl.outQ.size()) {
            sl.outQ.clear();
            sl.outQHead = 0;
        }
        finish(s);
    }

    // V1 speculates only when exactly one unfinished slot is active. A verify
    // round blocks for about one serial chunk and would otherwise be unfair to
    // concurrent slots; multi-slot batching is a separate optimization.
    if (specOn) {
        int only = -1, nActive = 0;
        for (uint32_t s = 0; s < nSlots; s++) {
            if (slots[s].active && !slots[s].finPending) {
                nActive++;
                only = (int)s;
            }
        }
        if (nActive == 1) {
            Slot& sl = slots[only];
            while (sl.active && !sl.finPending && sl.cursor >= sl.prompt.size() &&
                   outCnt[only] < chunkN) {
                uint32_t n = specDraft((uint32_t)only);
                if (n < 2) break;
                verifyRound(specToks.data(), n, (uint32_t)only, sl.pos, specAm.data());

                uint32_t accepted = 1;
                while (accepted < n && specToks[accepted] == specAm[accepted - 1]) accepted++;

                uint32_t emitN = accepted;
                bool eosHit = false;
                if (!noEos) {
                    for (uint32_t i = 0; i < accepted; i++) {
                        if (specAm[i] != eosTok) continue;
                        eosHit = true;
                        emitN = i;       // EOS itself is not returned to the caller
                        accepted = i + 1;  // but the state through its predictor is committed
                        break;
                    }
                }

                if (accepted == n) {
                    promoteScratch((uint32_t)only);
                } else {
                    std::vector<float> dummy;
                    prefillBatchLast(specToks.data(), accepted, (uint32_t)only, dummy,
                                     /*wantLogits=*/false, sl.pos);
                }
                sl.pos += accepted;
                for (uint32_t i = 0; i < emitN; i++) {
                    sl.genTokens.push_back(specAm[i]);
                    sl.last = specAm[i];
                    sl.gen++;
                    emit((uint32_t)only, specAm[i]);
                }
                sl.specRounds++;
                sl.specFed += accepted;
                sl.specEmitted += emitN;

                if (eosHit || sl.gen >= sl.maxGen || sl.pos >= nCtx) {
                    snapshotSlot((uint32_t)only);
                    sl.finPending = true;
                    finish((uint32_t)only);
                    break;
                }
            }
        }
    }

    // Serial fallback for slots with no high-confidence draft and for all
    // multi-slot steps. It is also responsible for prompt ingestion.
    for (uint32_t step = 0; step < chunkN; step++) {
        int nAct = 0;
        uint32_t maxZ = 0;
        bool need = false;
        for (uint32_t s = 0; s < nSlots; s++) {
            Slot& sl = slots[s];
            if (!sl.active || sl.finPending) {
                slotIn[s] = 0;
                slotPos[s] = 0;
                continue;
            }
            nAct++; maxZ = s + 1;
            if (sl.cursor < sl.prompt.size()) {
                slotIn[s] = sl.prompt[sl.cursor];
                slotPos[s] = sl.cursor;
                need = true;
            } else {
                slotIn[s] = sl.last;
                slotPos[s] = sl.pos;
                if (outCnt[s] < chunkN) need = true;
            }
        }
        if (!nAct || !need) break;

        bool equivalentSlots = moeSlotGrouped && maxZ >= 2 && nAct == (int)maxZ;
        if (equivalentSlots) {
            const Slot& r = slots[0];
            for (uint32_t s = 1; s < maxZ; ++s) {
                const Slot& q = slots[s];
                if (!q.active || q.equivId != r.equivId || q.cursor != r.cursor ||
                    q.pos != r.pos || q.genTokens.size() != r.genTokens.size() ||
                    q.last != r.last) {
                    equivalentSlots = false;
                    break;
                }
            }
        }

        @autoreleasepool {
            auto he0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc =
                [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            encodeStep(enc, maxZ, equivalentSlots);
            [enc endEncoding];
            auto he1 = std::chrono::steady_clock::now();
            [cb commit];
            [cb waitUntilCompleted];
            auto he2 = std::chrono::steady_clock::now();
            if (getenv("QK_STEP_STATS")) {
                static double gpuSum = 0, encSum = 0, waitSum = 0; static uint32_t nStep = 0;
                gpuSum += cb.GPUEndTime - cb.GPUStartTime; nStep++;
                encSum += std::chrono::duration<double>(he1 - he0).count();
                waitSum += std::chrono::duration<double>(he2 - he1).count();
                if (nStep % 64 == 0)
                    fprintf(stderr, "[step] n=%u gpu %.2f | encode %.2f | commit+wait %.2f ms avg\n",
                            nStep, gpuSum * 1e3 / nStep, encSum * 1e3 / nStep, waitSum * 1e3 / nStep);
            }
        }

        for (uint32_t s = 0; s < nSlots; s++) {
            Slot& sl = slots[s];
            if (!sl.active || sl.finPending) continue;
            uint32_t sampled = tok[s];
            bool prefilling = sl.cursor < sl.prompt.size();
            if (prefilling && sl.cursor + 1 < sl.prompt.size()) {
                sl.cursor++;
                continue;
            }
            if (prefilling) { sl.cursor = (uint32_t)sl.prompt.size(); sl.pos = (uint32_t)sl.prompt.size(); }
            sl.serialSteps++;
            if (sampled == eosTok && !noEos) {
                if (!prefilling) sl.pos++;
                snapshotSlot(s);
                sl.finPending = true;
                finish(s);
                continue;
            }
            emit(s, sampled);
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

void qk_engine::prefillBatchLast(const uint32_t* toks, uint32_t n, uint32_t slot,
                                 std::vector<float>& logits, bool wantLogits, uint32_t base,
                                 uint32_t* argmaxOut, const float* hiddenIn, float* hiddenOut,
                                 bool scratchState) {
    if ((!toks && !hiddenIn) || n < 1 || n > maxB || slot >= nSlots ||
        (size_t)base + n > nCtx ||
        (hiddenIn && firstStage()) || (!hiddenIn && !firstStage()) ||
        ((wantLogits || argmaxOut) && !lastStage())) {
        fprintf(stderr, "prefillBatchLast: bad args n=%u slot=%u base=%u stage=[%u,%u)\n",
                n, slot, base, lFirst, lEnd);
        exit(1);
    }
    if (hiddenIn) memcpy(bbXin.contents, hiddenIn, (size_t)n * nEmbd * 4);   // UMA
    else memcpy(bbIds.contents, toks, (size_t)n * 4);
    // QK_PREFILL_SKIP=proj|moe|dn|attn|gu|down: drop that stage class
    // (results are WRONG — timing isolation only)
    static const char* skip = getenv("QK_PREFILL_SKIP");
    const bool skProj = skip && strstr(skip, "proj");
    const bool skMoe  = skip && strstr(skip, "moe");
    const bool skDn   = skip && strstr(skip, "dn");
    const bool skAttn = skip && strstr(skip, "attn");
    const bool skGu   = skip && strstr(skip, "gu");
    const bool skDown = skip && strstr(skip, "down");
    // MMA prefill attention is the default (beats scalar ~2x on attn, pp512
    // win vs llama). QK_FA_MMA=0 forces the old scalar kernel (rollback/debug).
    static const bool faMma = !getenv("QK_FA_MMA") || atoi(getenv("QK_FA_MMA")) != 0;
    if (wantLogits) logits.resize(vocab);
    if (base == 0) resetSlot(slot);
    // seed each deltanet layer's conv carry (plain UMA memcpy; GPU idle here)
    for (uint32_t il = lFirst; il < lEnd; il++) {
        if (!layers[il].rec) continue;
        const uint32_t stateSlot = scratchState ? nSlots : slot;
        uint8_t* dst = (uint8_t*)bbCarry.contents + (size_t)il * chQkv * 3 * 4;
        if (base == 0) memset(dst, 0, (size_t)chQkv * 3 * 4);
        else memcpy(dst, (uint8_t*)layers[il].st1.contents +
                         (size_t)stateSlot * layers[il].ps1,
                    (size_t)chQkv * 3 * 4);
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        auto dspz = [&](id<MTLComputePipelineState> pso,
                        std::initializer_list<WB> bufs,
                        const void* pc, uint32_t pcSize, uint32_t wgs, uint32_t thr, uint32_t z) {
            [enc setComputePipelineState:pso];
            uint32_t i = 0;
            for (const WB& b : bufs) [enc setBuffer:b.b offset:b.o atIndex:i++];
            [enc setBytes:pc length:pcSize atIndex:i];
            [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, z)
                threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
        };
        auto bar = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
        const uint32_t thrN = nsg * 32;

        struct { uint32_t n; float e; } pcRms{nEmbd, eps};
        struct { uint32_t m, k; } pcP;
        struct { uint32_t n, hv, Tn; } pcAbB{nEmbd, hV, n};
        struct { uint32_t channels, dState, qkCh; float e; uint32_t Tn; }
            pcConvB{chQkv, dS, 2 * hK * dS, eps, n};
        struct { uint32_t dState, hK_, hV_, Tn, kd; } pcStepB{dS, hK, hV, n, dnKDiv};
        struct { uint32_t dState, hK_, hV_, Tn, kd; } pcDnC{dS, hK, hV, n, dnKDiv};
        const uint32_t nChDn = (n + 63) / 64;
        struct { uint32_t dState, hV_; float e; uint32_t Tn; } pcGateB{dS, hV, eps, n};
        struct { uint32_t a, b, cc, d; } pcv{nEmbd, ffE, nExp, nUsed};
        struct { uint32_t a, b, cc, d; float e; } pcv5{nEmbd, ffE, nExp, nUsed, eps};
        struct { uint32_t tmax, dh_, nRot_, hQ_, hKV_; float e, fb; uint32_t base_, Tn, qbase; }
            pcFaB{nCtx, dh, nRot, hQ, hKV, eps, kFreqBase, base, n, 0};
        struct { uint32_t k, idx, pr; float e; } pcE{nEmbd, 0, 1, eps};
        struct { uint32_t M, K, N; } pcG;

        // projection: tiled GEMM for n>=48 (weight reads amortized), else z=n GEMV
        auto proj = [&](WB W, id<MTLBuffer> X, id<MTLBuffer> Y,
                        uint32_t M, uint32_t K, bool isOut, bool iq4) {
            if (skProj) return;
            if (n >= 48) {
                pcG = {M, K, n};
                if (iq4)
                    dspz(pGemmB4, {W, X, Y}, &pcG, 12, (M + 63) / 64, 128, (n + 31) / 32);
                else
                    dspz(pGemmB, {W, X, Y}, &pcG, 12, (M + gemmBM - 1) / gemmBM, gemmThreads,
                         (n + gemmBN - 1) / gemmBN);
            } else {
                pcP = {M, K};
                if (iq4)
                    dspz(pGemv4, {W, X, Y}, &pcP, 8, (M + 3) / 4, 64, n);
                else
                    dspz(isOut ? pGemvO : pGemvA, {W, X, Y}, &pcP, 8,
                         isOut ? (M + 1) / 2 : (M + 3) / 4, 256, n);
            }
        };

        if (hiddenIn) {
            // later stage: the previous stage's residual rows are already in
            // bbXin; apply this stage's first-layer norm to produce bbXn
            dspz(pRms, {bbXin, layers[lFirst].aNorm, bbXn}, &pcRms, 8, 1, 256, n);
        } else {
            dspz(pEmb, {bEmbdW, bbIds, bbXin, layers[lFirst].aNorm, bbXn}, &pcE, 16, 1, 256, n);
        }
        bar();
        for (uint32_t il = lFirst; il < lEnd; il++) {
            Layer& L = layers[il];
            // Verify advances recurrent state in the shared scratch stripe;
            // attention KV remains positional in the live slot and needs no
            // rollback after a rejected suffix.
            const uint32_t stateSlot = scratchState && L.rec ? nSlots : slot;
            size_t so1 = (size_t)stateSlot * L.ps1, so2 = (size_t)stateSlot * L.ps2;
            if (L.rec) {
                proj(L.qkvW, bbXn, bbBig, chQkv, nEmbd, false, L.iq4P1);
                proj(L.zW, bbXn, bbMid, dIn, nEmbd, false, L.iq4P2);
                dspz(pAbB, {bbXn, L.alW, L.beW, L.dt, L.av, bbGb}, &pcAbB, 12,
                     (2 * hV + nsg - 1) / nsg, thrN, n);
                bar();
                if (!skDn) {
                dspz(pConvB, {WB{bbCarry, (NSUInteger)il * chQkv * 3 * 4}, bbBig, L.ker,
                              bbConvOut, WB{L.st1, so1}},
                     &pcConvB, 20, chQkv / dS, dS, n);
                bar();
                if (dnChunk) {
                    dspz(pDnKq, {bbConvOut, bbDnKQ}, &pcDnC, 20, hK, dS, nChDn);
                    bar();
                    dspz(pDnSolve, {bbConvOut, bbGb, bbDnKQ, bbDnUW, bbDnAtt, bbDnEl},
                         &pcDnC, 20, hV, dS, nChDn);
                    bar();
                    dspz(pDnStepC, {bbDnUW, bbDnAtt, bbDnEl, bbO, WB{L.st2, so2}},
                         &pcDnC, 20, hV, dS, 2);   // z = state column panel
                } else {
                    dspz(pStepB, {bbConvOut, bbGb, bbO, WB{L.st2, so2}},
                         &pcStepB, 20, hV, dS, 1);
                }
                bar();
                dspz(pGateB, {bbO, L.sn, bbMid, bbAtt}, &pcGateB, 16,
                     (hV + nsg - 1) / nsg, thrN, n);
                bar();
                }
                proj(L.outW, bbAtt, bbAttnOut, nEmbd, dIn, true, L.iq4Wo);
            } else {
                proj(L.wq, bbXn, bbBig, chQkv, nEmbd, false, L.iq4P1);
                proj(L.wk, bbXn, bbKin, hKV * dh, nEmbd, false, L.iq4P2);
                proj(L.wv, bbXn, bbVin, hKV * dh, nEmbd, false, L.iq4P3);
                bar();
                if (!skAttn) {
                dspz(pPrepB, {bbBig, bbKin, bbVin, L.qn, L.kn, bbMid, WB{L.st1, so1},
                              WB{L.st2, so2}, bRope}, &pcFaB, 40, hQ + 2 * hKV, 256, n);
                bar();
                if (faMma)
                    dspz(pAttnBM, {bbMid, WB{L.st1, so1}, WB{L.st2, so2}, bbBig, bbAtt},
                         &pcFaB, 40, hQ, 256, (n + 15) / 16);   // QTM=16 queries/tile
                else
                    dspz(pAttnB, {bbMid, WB{L.st1, so1}, WB{L.st2, so2}, bbBig, bbAtt},
                         &pcFaB, 40, hQ, 256, n);
                bar();
                }
                proj(L.wo, bbAtt, bbAttnOut, nEmbd, dIn, true, L.iq4Wo);
            }
            bar();
            if (moeGrouped && n >= moeGroupN) {
                // grouped regime: residual+norm via add_rmsnorm (same y/xn2 as
                // the fused kernel), router logits as one f32 GEMM
                dspz(pAddN, {bbXin, bbAttnOut, L.pn, bbY, bbXn2}, &pcRms, 8, 1, 256, n);
                bar();
                struct { uint32_t M, K, N; } pcLg{nExp + 1, nEmbd, n};
                dspz(pLogG, {L.mgi, L.mgis, bbXn2, bbML}, &pcLg, 12, (nExp + 1 + 31) / 32, 128,
                     (n + 31) / 32);
            } else {
                dspz(pMoeLA, {L.mgi, L.mgis, bbXin, bbAttnOut, L.pn, bbML, bbY, bbXn2},
                     &pcv5, 20, (nExp + 1 + nsg - 1) / nsg, thrN, n);
            }
            bar();
            if (!skMoe) {
            dspz(pMoeS, {bbML, bbMSel}, &pcv, 16, 1, 32, n);
            bar();
            if (moeGrouped && n >= moeGroupN) {
                struct { uint32_t a, b, cc, d, n; } pcg{nEmbd, ffE, nExp, nUsed, n};
                dspz(pMoeGrp, {bbMSel, bbStart, bbATok, bbASlot}, &pcg, 20, 1, 256, 1);
                bar();
                if (!skGu) {
                if (moeGrouped == 5)
                    dspz(guIq4 ? pMoeGuG5i : pMoeGuG5,
                         {L.mge, L.mue, L.mgs, L.mus, bbXn2, bbStart, bbATok,
                          bbASlot, bbMH}, &pcv, 16, (nExp + 1) * (ffE / 32), 128,
                         (n + 31) / 32);
                else if (guIq4)   // 80B: every other variant maps to the f32 iq4 twin
                    dspz(pMoeGuG4i, {L.mge, L.mue, L.mgs, L.mus, bbXn2, bbStart, bbATok,
                                     bbASlot, bbMH}, &pcv, 16, (nExp + 1) * (ffE / 32), 128, 1);
                else if (moeGrouped == 4)
                    dspz(pMoeGuG4, {L.mge, L.mue, L.mgs, L.mus, bbXn2, bbStart, bbATok,
                                    bbASlot, bbMH}, &pcv, 16, (nExp + 1) * (ffE / 32), 128, 1);
                else if (moeGrouped == 3)
                    dspz(pMoeGuG3, {L.mge, L.mue, L.mgs, L.mus, bbXn2, bbStart, bbATok,
                                    bbASlot, bbMH}, &pcv, 16, (nExp + 1) * (ffE / 64), 256, 1);
                else if (moeGrouped == 2)
                    dspz(pMoeGuG2, {L.mge, L.mue, L.mgs, L.mus, bbXn2, bbStart, bbATok,
                                    bbASlot, bbMH}, &pcv, 16, (nExp + 1) * (ffE / 32), 128, 1);
                else
                    dspz(pMoeGuG, {L.mge, L.mue, L.mgs, L.mus, bbXn2, bbStart, bbATok,
                                   bbASlot, bbMH}, &pcv, 16, ((nExp + 1) * ffE + nsg - 1) / nsg,
                         thrN, 1);
                }
                bar();
                if (!skDown) {
                if (moeGrouped == 5)
                    dspz(L.downQ6 ? pMoeDGP6 : pMoeDGP4,
                         {L.mde, L.mds, bbMH, bbStart, bbATok, bbASlot, bbMDy},
                         &pcv, 16, (nExp + 1) * (nEmbd / 64), 128, (n + 31) / 32);
                else if (moeGrouped == 3)
                    dspz(L.downQ6 ? pMoeDGH6 : pMoeDGH4,
                         {L.mde, L.mds, bbMH, bbStart, bbATok, bbASlot, bbMDy},
                         &pcv, 16, (nExp + 1) * (nEmbd / 64), 256, 1);
                else
                    dspz(L.downQ6 ? pMoeDG6 : pMoeDG4,
                         {L.mde, L.mds, bbMH, bbStart, bbATok, bbASlot, bbMDy},
                         &pcv, 16, (nExp + 1) * (nEmbd / 32), 128, 1);
                bar();
                dspz(pMoeDR, {bbMDy, bbMSel, bbMY}, &pcv, 16, (nEmbd + 255) / 256, 256, n);
                }
            } else {
                if (!skGu)
                dspz(guIq4 ? pMoeGu4 : pMoeGu,
                     {L.mge, L.mue, L.mgs, L.mus, bbXn2, bbMSel, bbMH}, &pcv, 16,
                     ((nUsed + 1) * ffE + nsg - 1) / nsg, thrN, n);
                bar();
                if (!skDown)
                dspz(L.downQ6 ? pMoeD6 : pMoeD4, {L.mde, L.mds, bbMH, bbMSel, bbMY},
                     &pcv, 16, (nEmbd + nsg - 1) / nsg, thrN, n);
            }
            bar();
            }
            WB nextNorm = il + 1 < lEnd ? layers[il + 1].aNorm : bONorm;
            dspz(pAddN, {bbY, bbMY, nextNorm, bbXin, bbXn}, &pcRms, 8, 1, 256, n);
            bar();
        }
        if (wantLogits || argmaxOut) {
            // per-position ids (argmaxOut: stageRun wire contract) need the
            // head for every row; logits-only callers need just the LAST row
            // — the z=n head is ~417 MB of Q6_K weight reads PER ROW.
            const uint32_t nh = argmaxOut ? n : 1;
            lastRunRows = nh;
            struct { uint32_t m, k; } pcHead{vocab, nEmbd};
            dspz(pHead, {bHeadW, nh == 1 ? WB{bbXn, (NSUInteger)(n - 1) * nEmbd * 4} : WB{bbXn},
                         bbLogits}, &pcHead, 8, (vocab + 3) / 4, 64, nh);
            if (argmaxOut) {
                bar();
                const uint32_t amWgs = (vocab + 4095) / 4096;
                struct { uint32_t nn, span; } pcAm{vocab, 4096};
                struct { uint32_t m, pos; } pcAm2{amWgs, 0};
                dspz(pAm1, {bbLogits, bbAV, bbAI}, &pcAm, 8, amWgs, 256, n);
                bar();
                dspz(pAm2, {bbAV, bbAI, bbTok, bRbScratch}, &pcAm2, 8, 1, 256, n);
            }
        }
        [enc endEncoding];
        auto tw0 = std::chrono::steady_clock::now();
        [cb commit];
        [cb waitUntilCompleted];
        lastCbGpu = cb.GPUEndTime - cb.GPUStartTime;
        lastCbWall = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
    }
    if (wantLogits)
        memcpy(logits.data(),
               (uint8_t*)bbLogits.contents + (argmaxOut ? (size_t)(n - 1) * vocab * 4 : 0),
               (size_t)vocab * 4);
    if (argmaxOut) memcpy(argmaxOut, bbTok.contents, (size_t)n * 4);
    if (hiddenOut) memcpy(hiddenOut, bbXin.contents, (size_t)n * nEmbd * 4);   // UMA
    if (getenv("QK_STAGE_STATS")) {
        static double gSum = 0, wSum = 0; static uint32_t nCall = 0;
        gSum += lastCbGpu; wSum += lastCbWall; nCall++;
        if (nCall % 32 == 0)
            fprintf(stderr, "[stage] n=%u calls=%u gpu %.2f | submit-wall %.2f ms avg\n",
                    n, nCall, gSum * 1e3 / nCall, wSum * 1e3 / nCall);
    }
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
                         hiddenIn ? hiddenIn + (size_t)off * nEmbd : nullptr,
                         hiddenOut ? hiddenOut + (size_t)off * nEmbd : nullptr);
    }
    return 0;
}

int qk_engine::stageTopK(uint32_t k, uint32_t* idsOut, float* valsOut) {
    // Sampling hook: top-k (id, logit) of the FINAL position's row after a
    // last-stage stageRun, descending. UMA read — bbLogits.contents is
    // coherent after prefillBatchLast's waitUntilCompleted; the greedy path
    // is untouched. NB lastRunRows is the head's nh (== n when argmaxOut).
    if (!lastStage() || !idsOut || !valsOut || k < 1 || k > 256 || k > vocab) return -1;
    if (!lastRunRows) return -2;
    const float* row = (const float*)((const uint8_t*)bbLogits.contents +
                                      (size_t)(lastRunRows - 1) * vocab * 4);
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
    for (uint32_t s = 0; s < slot; s++) {
        if (slots[s].active) {
            fprintf(stderr, "serialPrefillLogits(slot=%u): lower slot %u is active; refusing\n",
                    slot, s);
            std::fill(logits.begin(), logits.end(), 0.0f);
            return 0;
        }
    }
    resetSlot(slot);
    uint32_t* slotIn = (uint32_t*)bSlotIn.contents;
    uint32_t* slotPos = (uint32_t*)bSlotPos.contents;
    for (uint32_t s = 0; s < nSlots; s++) { slotIn[s] = 0; slotPos[s] = 0; }
    for (uint32_t i = 0; i < n; i++) {
        slotIn[slot] = toks[i];
        slotPos[slot] = i;
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc =
                [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            encodeStep(enc, slot + 1);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    memcpy(logits.data(), (uint8_t*)bLogits.contents + (size_t)slot * vocab * 4,
           (size_t)vocab * 4);
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
uint32_t qk_state_n(const qk_engine* e) { return (uint32_t)e->pcache.size(); }

int qk_state_save(qk_engine* e, uint32_t slot, uint32_t idx, uint32_t n_tok) {
    if (!e || slot >= e->nSlots || idx >= e->pcache.size()) return -1;
    e->copyStripes(slot, e->pcache[idx].snap.data(), /*save=*/true, n_tok);
    return 0;
}

int qk_state_load(qk_engine* e, uint32_t slot, uint32_t idx, uint32_t n_tok) {
    if (!e || slot >= e->nSlots || idx >= e->pcache.size()) return -1;
    e->copyStripes(slot, e->pcache[idx].snap.data(), /*save=*/false, n_tok);
    return 0;
}

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
int qk_slot_start(qk_engine* e, uint32_t slot, const uint32_t* prompt, uint32_t n_prompt,
                  uint32_t max_gen, uint32_t snap_prefix) {
    if (!e || slot >= e->nSlots) return -1;
    if (e->splitStage()) return -5;  // split stages are driven via qk_stage_run
    if (e->slots[slot].active) return -2;
    if (!prompt || n_prompt < 1 || n_prompt + max_gen > e->nCtx) return -3;
    for (uint32_t i = 0; i < n_prompt; i++) if (prompt[i] >= e->vocab) return -4;
    qk_engine::Slot& s = e->slots[slot];
    int cidx = e->matchPrefix(prompt, n_prompt);
    uint32_t start;
    if (cidx >= 0) {
        e->restoreInto(slot, cidx);
        start = (uint32_t)e->pcache[cidx].tokens.size();
    } else {
        e->resetSlot(slot);
        start = 0;
    }
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
    uint32_t snapAt = (snap_prefix > start && snap_prefix < target) ? snap_prefix : 0;
    uint32_t snapPos = 0;
    if (!getenv("QK_NO_BATCH")) {
        if (snapAt) {
            batch_to(snapAt);
            if (e->shareFork && done > start) {
                s.pos = done; s.prompt.assign(prompt, prompt + n_prompt);
                e->snapshotSlot(slot);
                snapPos = done;
            }
        }
        batch_to(target);
    }
    if (getenv("QK_PCACHE_LOG")) {
        qkStatsLine("[pcache] slot=%u prompt=%u reuse=%u prefill=%u hit=%d snap=%u\n",
                    slot, n_prompt, start, done - start, cidx >= 0 ? 1 : 0, snapPos);
    }
    s.cursor = done; s.pos = done;
    s.active = true; s.prompt.assign(prompt, prompt + n_prompt);
    s.genTokens.clear();
    s.gen = 0; s.maxGen = max_gen; s.last = 0;
    s.resetSpec();
    s.equivId = 0;
    for (uint32_t i = 0; i < e->nSlots; ++i) {
        if (i == slot || !e->slots[i].active || e->slots[i].prompt != s.prompt) continue;
        s.equivId = e->slots[i].equivId;
        break;
    }
    if (!s.equivId) s.equivId = ++e->equivClock;
    if (e->shareFork && snapPos == 0 && done > start) e->snapshotSlot(slot);
    return 0;
}

__attribute__((visibility("default")))
void qk_slot_cancel(qk_engine* e, uint32_t slot) {
    if (e && slot < e->nSlots) {
        e->slots[slot].active = false;
        e->slots[slot].prompt.clear();
        e->slots[slot].equivId = 0;
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


// Batched-prefill GEMM validation: Y[N,M] = X[N,K]·W[M,K]^T vs CPU reference.
static bool caseBGemm(MtlCtx& c, uint32_t M, uint32_t K, uint32_t N, uint32_t iters) {
    printf("\n== bgemm  Y[%u,%u] = X[%u,%u] . W(q8)[%u,%u]^T ==\n", N, M, N, K, M, K);
    if (K % 32) { fprintf(stderr, "K must be a multiple of 32\n"); return false; }
    size_t nb = (size_t)M * K / 32;
    std::vector<block_q8_0> blocks(nb);
    std::mt19937 rng(42);
    for (auto& b : blocks) {
        b.d = qk_f32_to_f16(0.005f + 0.02f * (rng() & 0xFFFF) / 65536.0f);
        for (auto& q : b.qs) q = (int8_t)((int)(rng() % 255) - 127);
    }
    std::vector<float> X((size_t)N * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : X) v = nd(rng);

    std::vector<float> yref((size_t)N * M), tmp(K);
    for (uint32_t m = 0; m < M; m++) {
        dequant_row_q8_0(&blocks[(size_t)m * (K / 32)], tmp.data(), K);
        for (uint32_t n = 0; n < N; n++) {
            double a = 0;
            for (uint32_t k = 0; k < K; k++) a += (double)tmp[k] * X[(size_t)n * K + k];
            yref[(size_t)n * M + m] = (float)a;
        }
    }

    const char* gv = getenv("QK_GEMM");
    const char* fn = "gemm_q8_0";
    if (gv && !strcmp(gv, "sg")) fn = "gemm_q8_0_sg";
    if (gv && !strcmp(gv, "h")) fn = "gemm_q8_0_h";
    if (gv && !strcmp(gv, "h32")) fn = "gemm_q8_0_h32";
    if (gv && !strcmp(gv, "h2")) fn = "gemm_q8_0_h2";
    if (gv && !strcmp(gv, "hp")) fn = "gemm_q8_0_hp";
    bool scalar = !strcmp(fn, "gemm_q8_0");
    const bool hp = !strcmp(fn, "gemm_q8_0_hp");
    const uint32_t tBM = !strcmp(fn, "gemm_q8_0_h2") ? 64 : hp ? 64 : 128;
    const uint32_t tBN = !strcmp(fn, "gemm_q8_0_h2") ? 128 : hp ? 32 : 64;
    id<MTLComputePipelineState> pso = getPipe(c, "gemm_q8_0", fn, 0);
    printf("variant: %s\n", fn);
    id<MTLBuffer> bW = createBuf(c, nb * sizeof(block_q8_0), blocks.data());
    id<MTLBuffer> bX = createBuf(c, X.size() * 4, X.data());
    id<MTLBuffer> bY = createBuf(c, yref.size() * 4);
    struct { uint32_t M, K, N; } pc{M, K, N};
    auto enc1 = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:bY offset:0 atIndex:2];
        [enc setBytes:&pc length:12 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((M + tBM - 1) / tBM, 1, (N + tBN - 1) / tBN)
            threadsPerThreadgroup:MTLSizeMake(!strcmp(fn, "gemm_q8_0_sg") || hp ? 128 : 256, 1, 1)];
    };
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        enc1(enc);
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
    const float* yg = (const float*)bY.contents;
    double rms = 0;
    for (size_t i = 0; i < yref.size(); i++) rms += (double)yref[i] * yref[i];
    rms = std::sqrt(rms / yref.size());
    double denomFloor = std::max(1e-3, 1e-3 * rms);
    double maxRel = 0; uint32_t bad = 0;
    for (size_t i = 0; i < yref.size(); i++) {
        double rel = std::fabs((double)yg[i] - yref[i]) / std::max(denomFloor, (double)std::fabs(yref[i]));
        maxRel = std::max(maxRel, rel);
        if (rel > 1e-2 && bad++ < 4) printf("  y[%zu]: gpu=%g ref=%g\n", i, yg[i], yref[i]);
    }
    bool pass = bad == 0;
    printf("correctness: max_rel_err = %.3g  ->  %s%s\n", maxRel, pass ? "PASS" : "FAIL",
           (!pass && !scalar) ? " (f16 class: synthetic tolerance unpassable; real gate is prefillcmp)" : "");
    if ((pass || !scalar) && iters) {
        auto run = [&]() -> double {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (uint32_t i = 0; i < iters; i++) enc1(enc);
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / iters;
            }
        };
        run();
        double ns = run();
        double flops = 2.0 * M * K * N;
        double wBytes = (double)nb * 34;
        printf("gpu: %8.1f µs/iter | %7.1f GFLOP/s | W %6.1f GB/s-equiv\n",
               ns / 1e3, flops / ns, wBytes / ns);
    }
    return pass;
}

// Chunked delta rule (dn_chunk_{kq,solve,step}) vs the sequential
// dn_step_batch on random inputs with a NONZERO initial state. The sequential
// kernel is the trusted reference (token-exact vs llama.cpp CPU end-to-end);
// chunking is a floating-point reorder of the same math, so we gate on
// max_rel over the o outputs and the final state.
static bool caseDnChunk(MtlCtx& c, uint32_t TnMain, uint32_t iters) {
    const uint32_t dS = 128, hK = 16, hV = 32;
    const uint32_t chQkv = (2 * hK + hV) * dS;
    const uint32_t TnCap = std::max(TnMain, 512u);
    printf("\n== dncmp  chunked delta rule vs dn_step_batch (dS=%u hK=%u hV=%u) ==\n",
           dS, hK, hV);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> conv((size_t)TnCap * chQkv), gb((size_t)TnCap * 2 * hV),
        st0((size_t)hV * dS * dS);
    for (auto& v : conv) v = nd(rng);
    for (uint32_t t = 0; t < TnCap; t++)   // L2-normalize q/k rows (dn_conv does)
        for (uint32_t hh = 0; hh < 2 * hK; hh++) {
            float* r = &conv[(size_t)t * chQkv + hh * dS];
            float ss = 0;
            for (uint32_t i = 0; i < dS; i++) ss += r[i] * r[i];
            float sc = 1.0f / std::sqrt(std::max(ss, 1e-12f));
            for (uint32_t i = 0; i < dS; i++) r[i] *= sc;
        }
    for (uint32_t t = 0; t < TnCap; t++)
        for (uint32_t h = 0; h < hV; h++) {
            gb[(size_t)t * 2 * hV + h] = -std::fabs(nd(rng)) * 1.5f;         // log decay <= 0
            gb[(size_t)t * 2 * hV + hV + h] = 1.f / (1.f + std::exp(-nd(rng)));  // beta
        }
    for (auto& v : st0) v = 0.3f * nd(rng);

    id<MTLBuffer> bConv = createBuf(c, conv.size() * 4, conv.data());
    id<MTLBuffer> bGb = createBuf(c, gb.size() * 4, gb.data());
    id<MTLBuffer> bSa = createBuf(c, st0.size() * 4);
    id<MTLBuffer> bSb = createBuf(c, st0.size() * 4);
    id<MTLBuffer> bOa = createBuf(c, (size_t)TnCap * hV * dS * 4);
    id<MTLBuffer> bOb = createBuf(c, (size_t)TnCap * hV * dS * 4);
    const size_t nChMax = (TnCap + 63) / 64;
    id<MTLBuffer> bKQ = createBuf(c, nChMax * hK * 2 * 64 * 64 * 4);
    id<MTLBuffer> bUW = createBuf(c, nChMax * hV * 4 * 64 * (size_t)dS * 4);
    id<MTLBuffer> bAtt = createBuf(c, nChMax * hV * 64 * 64 * 4);
    id<MTLBuffer> bEl = createBuf(c, nChMax * hV * 4);

    id<MTLComputePipelineState> pSeq = getPipe(c, "dn_batch", "dn_step_batch", 0);
    id<MTLComputePipelineState> pKq = getPipe(c, "dn_chunk", "dn_chunk_kq", 0);
    id<MTLComputePipelineState> pSolve = getPipe(c, "dn_chunk", "dn_chunk_solve", 0);
    id<MTLComputePipelineState> pStepC = getPipe(c, "dn_chunk", "dn_chunk_step", dS);

    struct { uint32_t dS_, hK_, hV_, Tn, kd; } pc{dS, hK, hV, 0, 0};
    auto encSeq = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pSeq];
        [enc setBuffer:bConv offset:0 atIndex:0];
        [enc setBuffer:bGb offset:0 atIndex:1];
        [enc setBuffer:bOa offset:0 atIndex:2];
        [enc setBuffer:bSa offset:0 atIndex:3];
        [enc setBytes:&pc length:20 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(hV, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
    };
    auto encChain = [&](id<MTLComputeCommandEncoder> enc) {
        const uint32_t nCh = (pc.Tn + 63) / 64;
        [enc setComputePipelineState:pKq];
        [enc setBuffer:bConv offset:0 atIndex:0];
        [enc setBuffer:bKQ offset:0 atIndex:1];
        [enc setBytes:&pc length:20 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(hK, 1, nCh)
            threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
        [enc setComputePipelineState:pSolve];
        [enc setBuffer:bConv offset:0 atIndex:0];
        [enc setBuffer:bGb offset:0 atIndex:1];
        [enc setBuffer:bKQ offset:0 atIndex:2];
        [enc setBuffer:bUW offset:0 atIndex:3];
        [enc setBuffer:bAtt offset:0 atIndex:4];
        [enc setBuffer:bEl offset:0 atIndex:5];
        [enc setBytes:&pc length:20 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(hV, 1, nCh)
            threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
        [enc setComputePipelineState:pStepC];
        [enc setBuffer:bUW offset:0 atIndex:0];
        [enc setBuffer:bAtt offset:0 atIndex:1];
        [enc setBuffer:bEl offset:0 atIndex:2];
        [enc setBuffer:bOb offset:0 atIndex:3];
        [enc setBuffer:bSb offset:0 atIndex:4];
        [enc setBytes:&pc length:20 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(hV, 1, 2)
            threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
    };
    auto runOnce = [&](bool chain) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];  // serial
            chain ? encChain(enc) : encSeq(enc);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
    };
    auto maxRel = [&](const float* a, const float* b, size_t nEl) {
        double rms = 0;
        for (size_t i = 0; i < nEl; i++) rms += (double)a[i] * a[i];
        rms = std::sqrt(rms / nEl);
        double floorD = std::max(1e-4, 1e-3 * rms), mr = 0;
        for (size_t i = 0; i < nEl; i++)
            mr = std::max(mr, std::fabs((double)a[i] - b[i]) /
                                  std::max(floorD, (double)std::fabs(a[i])));
        return mr;
    };

    bool pass = true;
    for (uint32_t kd : {0u, hV / hK}) {   // both GQA pairings (35B modulo, 80B consecutive)
    pc.kd = kd;
    for (uint32_t Tn : {1u, 5u, 64u, 65u, 200u, TnMain}) {
        if (Tn > TnCap) continue;
        pc.Tn = Tn;
        memcpy(bSa.contents, st0.data(), st0.size() * 4);
        memcpy(bSb.contents, st0.data(), st0.size() * 4);
        runOnce(false);
        runOnce(true);
        double relO = maxRel((const float*)bOa.contents, (const float*)bOb.contents,
                             (size_t)Tn * hV * dS);
        double relS = maxRel((const float*)bSa.contents, (const float*)bSb.contents,
                             st0.size());
        bool ok = relO < 5e-3 && relS < 5e-3;
        pass &= ok;
        printf("  kDiv=%u Tn=%-4u o max_rel %.3g | state max_rel %.3g -> %s\n",
               kd, Tn, relO, relS, ok ? "PASS" : "FAIL");
    }
    }
    pc.kd = 0;

    if (iters) {
        pc.Tn = TnMain;
        auto timeIt = [&](bool chain) -> double {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (uint32_t i = 0; i < iters; i++) chain ? encChain(enc) : encSeq(enc);
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / iters;
            }
        };
        timeIt(false); double nsSeq = timeIt(false);
        timeIt(true);  double nsCh = timeIt(true);
        printf("gpu @Tn=%u: sequential %8.1f µs | chunked %8.1f µs | %.2fx\n",
               TnMain, nsSeq / 1e3, nsCh / 1e3, nsSeq / nsCh);
        // per-kernel split (encode one pipeline repeatedly)
        auto timeOne = [&](uint32_t which) -> double {
            const uint32_t nCh = (pc.Tn + 63) / 64;
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (uint32_t i = 0; i < iters; i++) {
                    if (which == 0) {
                        [enc setComputePipelineState:pKq];
                        [enc setBuffer:bConv offset:0 atIndex:0];
                        [enc setBuffer:bKQ offset:0 atIndex:1];
                        [enc setBytes:&pc length:20 atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(hK, 1, nCh)
                            threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
                    } else if (which == 1) {
                        [enc setComputePipelineState:pSolve];
                        [enc setBuffer:bConv offset:0 atIndex:0];
                        [enc setBuffer:bGb offset:0 atIndex:1];
                        [enc setBuffer:bKQ offset:0 atIndex:2];
                        [enc setBuffer:bUW offset:0 atIndex:3];
                        [enc setBuffer:bAtt offset:0 atIndex:4];
                        [enc setBuffer:bEl offset:0 atIndex:5];
                        [enc setBytes:&pc length:20 atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(hV, 1, nCh)
                            threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
                    } else {
                        [enc setComputePipelineState:pStepC];
                        [enc setBuffer:bUW offset:0 atIndex:0];
                        [enc setBuffer:bAtt offset:0 atIndex:1];
                        [enc setBuffer:bEl offset:0 atIndex:2];
                        [enc setBuffer:bOb offset:0 atIndex:3];
                        [enc setBuffer:bSb offset:0 atIndex:4];
                        [enc setBytes:&pc length:20 atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(hV, 1, 2)
                            threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
                    }
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / iters;
            }
        };
        timeOne(0); double nsKq = timeOne(0);
        timeOne(1); double nsSol = timeOne(1);
        timeOne(2); double nsStp = timeOne(2);
        printf("  split: kq %8.1f µs | solve %8.1f µs | step %8.1f µs\n",
               nsKq / 1e3, nsSol / 1e3, nsStp / 1e3);
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

#ifndef QK_LIBRARY
int main(int argc, char** argv) {
    @autoreleasepool {
        std::string mode = argc > 1 ? argv[1] : "suite";

        if (mode == "list") {
            listTensors(argc > 2 ? argv[2] : "");
            return 0;
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
        // Layer count from the GGUF header (35B: 40, 80B: 48) — needed before
        // any engine exists to parse/validate the stage boundaries.
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
            // op4 load, idx in the n field, 4-byte status reply).
            // qkp3 = qkp2 + 5th header word topk: op1 to a LAST stage
            // replies n ids + topk (u32 id, f32 logit) pairs, descending.
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
                            // state save/load: idx rides in the n field; 4-byte status reply
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
            if (!runChain(prompt.data(), np, 0, ids.data())) return 1;
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

        if (mode == "serve-test2") {
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
                for (uint32_t sl = 0; sl < 2; sl++)
                    for (uint32_t i = 0; i < outCnt[sl]; i++) gen[sl].push_back(outTok[sl * ch + i]);
                if (steps % 64 == 0)
                    fprintf(stderr, "[t2] step %u gen0=%zu gen1=%zu\n", steps, gen[0].size(), gen[1].size());
            }
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            printf("serve-test2: OK gen0=%zu gen1=%zu tokens in %.1f ms (%u steps)\n",
                   gen[0].size(), gen[1].size(), ms, steps);
            printf("GEN0:"); for (uint32_t t : gen[0]) printf(" %u", t); printf("\n");
            printf("GEN1:"); for (uint32_t t : gen[1]) printf(" %u", t); printf("\n");
            qk_close(e);
            return 0;
        }

        if (mode == "serve-test") {
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
            uint32_t ch = qk_chunk(e);
            std::vector<std::vector<uint32_t>> gen(nSlots);
            std::vector<uint32_t> outTok((size_t)nSlots * ch), outCnt(nSlots);
            uint32_t finMask = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (uint32_t sl = 0; sl < nSlots; sl++)
                qk_slot_start(e, sl, prompt.data(), (uint32_t)prompt.size(), nGen, 0);
            while (qk_step_chunk(e, outTok.data(), outCnt.data(), &finMask) > 0)
                for (uint32_t sl = 0; sl < nSlots; sl++)
                    for (uint32_t i = 0; i < outCnt[sl]; i++) gen[sl].push_back(outTok[sl * ch + i]);
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            bool allEq = true;
            for (uint32_t sl = 1; sl < nSlots; sl++) if (gen[sl] != gen[0]) allEq = false;
            printf("serve-test: %u slots x prompt %zu -> %zu tokens each in %.1f ms (%.1f tok/s agg)\n",
                   nSlots, prompt.size(), gen[0].size(), ms,
                   (double)(gen[0].size() * nSlots) * 1000.0 / ms);
            if (nSlots > 1) printf("all slots identical: %s\n", allEq ? "YES" : "NO");
            printf("GEN:");
            for (uint32_t t : gen[0]) printf(" %u", t);
            printf("\n");
            qk_close(e);
            return 0;
        }

        if (mode == "cachetest") {
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
            auto run = [&](qk_engine* e, const std::vector<uint32_t>& ids, uint32_t gN, double& ms) {
                std::vector<uint32_t> out;
                qk_slot_start(e, 0, ids.data(), (uint32_t)ids.size(), gN, 0);
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
            auto R = run(eA, prompt, nGen, t1);
            std::vector<uint32_t> p2 = prompt;
            p2.insert(p2.end(), R.begin(), R.end());
            p2.insert(p2.end(), prompt.begin(), prompt.end());
            double tw;
            auto warm = run(eA, p2, nGen, tw);
            qk_close(eA);
            qk_engine* eB = qk_open(ggufPath(), &cfg, err, sizeof err);
            double tc;
            auto cold = run(eB, p2, nGen, tc);
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

        if (mode == "verify") {
            // Spec-decode P0 harness: compare oracle-draft batched verification
            // rounds with the serial greedy stream. This is the full-accept
            // path only: prefillBatchLast persists the round's final recurrent
            // state directly to the live slot. Partial acceptance needs the
            // scratch-state rollback path exercised by the later speccmp gate.
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
                const char* p = argv[4];
                while (*p) {
                    char* end = nullptr;
                    unsigned long k = strtoul(p, &end, 10);
                    if (end == p || k > UINT32_MAX) {
                        fprintf(stderr, "verify: bad K list '%s'\n", argv[4]);
                        return 1;
                    }
                    Ks.push_back((uint32_t)k);
                    p = end;
                    if (*p == ',') p++;
                    else if (*p) {
                        fprintf(stderr, "verify: bad K list '%s'\n", argv[4]);
                        return 1;
                    }
                }
            } else {
                // The default hp projection GEMM is exact through K=32 on the
                // gate stream. At K=64 its f16 input fragments can flip a
                // near-tie; QK_GEMM=scalar is the exact wider-K control.
                Ks = {2, 4, 8, 16, 32};
            }
            uint32_t tmax = argc > 5 ? (uint32_t)atoi(argv[5]) : 4096;
            qk_config cfg{1, tmax, 8};
            char err[256] = {0};
            qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
            if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }

            // Generate the serial reference stream and time only calls after
            // the first generated output (prompt ingestion is excluded).
            std::vector<uint32_t> serial = prompt;
            double serialMsTok = 0;
            {
                qk_slot_start(e, 0, prompt.data(), (uint32_t)prompt.size(), nGen, 0);
                uint32_t ch = qk_chunk(e), fin = 0;
                std::vector<uint32_t> out(ch), cnt(1);
                double decodeMs = 0;
                uint32_t decodeTok = 0;
                bool generated = false;
                while (true) {
                    auto t0 = std::chrono::steady_clock::now();
                    int active = qk_step_chunk(e, out.data(), cnt.data(), &fin);
                    double ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0).count();
                    if (active <= 0) break;
                    for (uint32_t i = 0; i < cnt[0]; i++) serial.push_back(out[i]);
                    if (generated) { decodeMs += ms; decodeTok += cnt[0]; }
                    if (cnt[0]) generated = true;
                    if (fin & 1u) break;
                }
                serialMsTok = decodeTok ? decodeMs / decodeTok : 0;
            }

            const uint32_t np = (uint32_t)prompt.size();
            const uint32_t ns = (uint32_t)serial.size();
            printf("verify: prompt %u, serial stream %u gen tokens, serial decode %.2f ms/tok\n",
                   np, ns - np, serialMsTok);
            if (ns < np + 4) {
                fprintf(stderr, "verify: stream too short (early EOS?)\n");
                qk_close(e);
                return 1;
            }

            bool allOk = true;
            printf("  %-4s %8s %10s %8s %10s   %s\n",
                   "K", "rounds", "round_ms", "ms/tok", "vs_serial", "exact");
            for (uint32_t K : Ks) {
                if (K < 1 || K > e->maxB) {
                    printf("  K=%u skipped (maxB=%u)\n", K, e->maxB);
                    continue;
                }

                // Reconstruct the prompt state from empty for each K.
                std::vector<float> dummy;
                for (uint32_t off = 0; off < np;) {
                    uint32_t n = std::min(e->maxB, np - off);
                    e->prefillBatchLast(prompt.data() + off, n, 0, dummy,
                                        /*wantLogits=*/false, off);
                    off += n;
                }

                // Feed serial[pos] as the first pending token. Per-position
                // argmax i must reproduce serial[pos+i+1]. Head and both
                // argmax reductions are encoded in the target command buffer.
                uint32_t pos = np, mismatches = 0, rounds = 0, committed = 0;
                uint32_t firstAt = 0, firstGot = 0, firstWant = 0;
                bool haveFirst = false;
                double totalMs = 0;
                std::vector<uint32_t> argmax(e->maxB);
                while (pos + 1 < ns) {
                    uint32_t n = std::min(K, ns - pos - 1);
                    auto t0 = std::chrono::steady_clock::now();
                    e->prefillBatchLast(&serial[pos], n, 0, dummy,
                                        /*wantLogits=*/false, pos, argmax.data());
                    totalMs += std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0).count();
                    rounds++;
                    for (uint32_t i = 0; i < n; i++) {
                        if (argmax[i] == serial[pos + i + 1]) continue;
                        if (!haveFirst) {
                            firstAt = pos + i;
                            firstGot = argmax[i];
                            firstWant = serial[pos + i + 1];
                            haveFirst = true;
                        }
                        mismatches++;
                    }
                    committed += n;
                    pos += n;
                }
                const double msTok = committed ? totalMs / committed : 0;
                const bool ok = mismatches == 0;
                allOk = allOk && ok;
                printf("  %-4u %8u %10.1f %8.2f %9.2fx   %s\n",
                       K, rounds, rounds ? totalMs / rounds : 0, msTok,
                       msTok > 0 ? serialMsTok / msTok : 0,
                       ok ? "EXACT" : "**MISMATCH**");
                if (!ok)
                    printf("      %u/%u positions diverged; first input-pos %u got %u want %u\n",
                           mismatches, committed, firstAt, firstGot, firstWant);
            }
            printf("verify: %s\n",
                   allOk ? "ORACLE SPEC ROUNDS TOKEN-EXACT" : "DIVERGENCE (see above)");
            qk_close(e);
            return allOk ? 0 : 1;
        }

        if (mode == "speccmp") {
            // Spec-decode P1 gate: force full and partial acceptance patterns
            // against a serial oracle. Every C-th draft token is corrupted;
            // C=0 always promotes scratch, C=1 rejects every first draft and
            // exercises the commit-pass path on every round.
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
            const uint32_t nGen = (uint32_t)atoi(argv[3]);
            const uint32_t K = argc > 4 ? (uint32_t)atoi(argv[4]) : 8;
            std::vector<uint32_t> corruptEvery{0, 4, 2, 1};
            if (argc > 5) {
                corruptEvery.clear();
                const char* p = argv[5];
                while (*p) {
                    char* end = nullptr;
                    unsigned long c = strtoul(p, &end, 10);
                    if (end == p || c > UINT32_MAX) {
                        fprintf(stderr, "speccmp: bad corruption list '%s'\n", argv[5]);
                        return 1;
                    }
                    corruptEvery.push_back((uint32_t)c);
                    p = end;
                    if (*p == ',') p++;
                    else if (*p) {
                        fprintf(stderr, "speccmp: bad corruption list '%s'\n", argv[5]);
                        return 1;
                    }
                }
            }

            qk_config cfg{1, 4096, 8};
            char err[256] = {0};
            qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
            if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
            if (K < 2 || K > e->maxB) {
                fprintf(stderr, "speccmp: K must be in [2,%u]\n", e->maxB);
                qk_close(e);
                return 1;
            }

            std::vector<uint32_t> serial = prompt;
            double serialMsTok = 0;
            {
                qk_slot_start(e, 0, prompt.data(), (uint32_t)prompt.size(), nGen, 0);
                const uint32_t ch = qk_chunk(e);
                uint32_t fin = 0;
                std::vector<uint32_t> out(ch), cnt(1);
                double decodeMs = 0;
                uint32_t decodeTok = 0;
                bool generated = false;
                while (true) {
                    auto t0 = std::chrono::steady_clock::now();
                    int active = qk_step_chunk(e, out.data(), cnt.data(), &fin);
                    double ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0).count();
                    if (active <= 0) break;
                    for (uint32_t i = 0; i < cnt[0]; i++) serial.push_back(out[i]);
                    if (generated) { decodeMs += ms; decodeTok += cnt[0]; }
                    if (cnt[0]) generated = true;
                    if (fin & 1u) break;
                }
                serialMsTok = decodeTok ? decodeMs / decodeTok : 0;
            }
            const uint32_t np = (uint32_t)prompt.size();
            const uint32_t sGen = (uint32_t)serial.size() - np;
            if (sGen < 4) {
                fprintf(stderr, "speccmp: stream too short (early EOS?)\n");
                qk_close(e);
                return 1;
            }
            printf("speccmp: prompt %u, serial %u gen tokens at %.2f ms/tok, K=%u\n",
                   np, sGen, serialMsTok, K);
            printf("  %-8s %7s %7s %7s %9s %8s %10s   %s\n",
                   "corrupt", "rounds", "avg_k", "full%", "commits", "ms/tok",
                   "vs_serial", "exact");

            bool allOk = true;
            std::vector<float> dummy;
            std::vector<uint32_t> argmax(e->maxB), input(e->maxB);
            for (uint32_t C : corruptEvery) {
                // Fresh live state. The final prompt argmax is the first
                // generated token and the pending input for round one.
                uint32_t next = 0;
                for (uint32_t off = 0; off < np;) {
                    uint32_t n = std::min(e->maxB, np - off);
                    bool last = off + n == np;
                    e->prefillBatchLast(prompt.data() + off, n, 0, dummy,
                                        /*wantLogits=*/false, off,
                                        last ? argmax.data() : nullptr);
                    if (last) next = argmax[n - 1];
                    off += n;
                }

                std::vector<uint32_t> output{next};
                uint32_t pos = np, rounds = 0, fullAccept = 0, commits = 0;
                uint64_t acceptedTotal = 0;
                double specMs = 0;
                bool hitEos = false;
                while (output.size() < sGen && !hitEos) {
                    uint32_t n = std::min(K, (uint32_t)serial.size() - pos);
                    if (!n) break;
                    input[0] = next;
                    for (uint32_t i = 1; i < n; i++) {
                        input[i] = serial[pos + i];
                        if (C && i % C == 0) input[i] = input[i] == 5 ? 6 : 5;
                    }

                    auto t0 = std::chrono::steady_clock::now();
                    e->verifyRound(input.data(), n, 0, pos, argmax.data());
                    uint32_t accepted = 1;
                    while (accepted < n && input[accepted] == argmax[accepted - 1]) accepted++;
                    if (accepted == n) {
                        e->promoteScratch(0);
                        fullAccept++;
                    } else {
                        // Live recurrent state stayed at `pos`; replay exactly
                        // the accepted prefix to commit it. KV suffix garbage
                        // is causally unreachable and will be overwritten.
                        e->prefillBatchLast(input.data(), accepted, 0, dummy,
                                            /*wantLogits=*/false, pos);
                        commits++;
                    }
                    specMs += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
                    rounds++;
                    acceptedTotal += accepted;

                    for (uint32_t i = 1; i < accepted; i++) output.push_back(input[i]);
                    if (argmax[accepted - 1] == e->eosTok) hitEos = true;
                    else output.push_back(argmax[accepted - 1]);
                    next = argmax[accepted - 1];
                    pos += accepted;
                }

                if (output.size() > sGen) output.resize(sGen);
                const bool ok = output.size() == sGen &&
                    std::equal(output.begin(), output.end(), serial.begin() + np);
                allOk = allOk && ok;
                printf("  C=%-6u %7u %7.2f %6.0f%% %9u %8.2f %9.2fx   %s\n",
                       C, rounds, rounds ? (double)acceptedTotal / rounds : 0,
                       rounds ? 100.0 * fullAccept / rounds : 0, commits,
                       output.empty() ? 0 : specMs / output.size(),
                       specMs > 0 ? serialMsTok * output.size() / specMs : 0,
                       ok ? "EXACT" : "**MISMATCH**");
                if (!ok) {
                    size_t d = 0;
                    while (d < output.size() && d < sGen && output[d] == serial[np + d]) d++;
                    printf("      first divergence at generated token %zu/%u\n", d, sGen);
                }
            }
            printf("speccmp: %s\n",
                   allOk ? "ROLLBACK TOKEN-EXACT (all corruption modes)"
                         : "DIVERGENCE (see above)");
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

            uint32_t cap = std::min<uint32_t>(e->maxB, ctx - 1);
            std::vector<uint32_t> sizes;
            for (uint32_t sz : {1u, 2u, 8u, 15u, 16u, 17u, 32u, 48u, 64u, 96u, 127u, 128u, 192u, 256u})
                if (sz <= cap) sizes.push_back(sz);
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
            // QK_PB_ONLY=<N>: run just that row; QK_PB_NOSERIAL=1: skip the
            // serial column (stage-isolation runs don't need it and it's hot)
            const char* pbOnly = getenv("QK_PB_ONLY");
            const bool pbNoSer = getenv("QK_PB_NOSERIAL") != nullptr;
            for (uint32_t N : {8u, 16u, 32u, 64u, 96u, 128u, 192u, 256u, 384u, 512u}) {
                if (N > cap) continue;
                if (pbOnly && N != (uint32_t)atoi(pbOnly)) continue;
                std::mt19937 rng(1234);
                std::vector<uint32_t> toks(N);
                for (uint32_t i = 0; i < N; i++) toks[i] = rng() % (e->vocab - 16) + 4;
                e->prefillBatchLast(toks.data(), N, 0, lB);   // warm
                double sMs = 0;
                if (!pbNoSer) {
                    e->serialPrefillLogits(toks.data(), N, 1, lS);
                    auto t0 = std::chrono::steady_clock::now();
                    e->serialPrefillLogits(toks.data(), N, 1, lS);
                    auto t1 = std::chrono::steady_clock::now();
                    sMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
                auto t1 = std::chrono::steady_clock::now();
                e->prefillBatchLast(toks.data(), N, 0, lB);
                auto t2 = std::chrono::steady_clock::now();
                double bMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
                printf("  %-6u %10.2f %10.2f %7.2fx %10.0f\n", N, sMs, bMs, sMs / bMs, N / (bMs / 1000.0));
            }
            qk_close(e);
            return 0;
        }

        if (mode == "prefilldecode") {
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

                std::vector<uint32_t> refSeq;
                qk_slot_start(e, 1, toks.data(), N, M, 0);
                while (e->stepChunk(outTok.data(), outCnt.data(), &fin) > 0)
                    for (uint32_t i = 0; i < outCnt[1]; i++) refSeq.push_back(outTok[(size_t)1 * ch + i]);

                std::vector<float> lB;
                e->prefillBatchLast(toks.data(), N, 0, lB);
                uint32_t tok0 = (uint32_t)(std::max_element(lB.begin(), lB.end()) - lB.begin());
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
                    s0.active = false;
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
        } else if (mode == "slotgemv") {
            ok = caseQ80Batch(c, argU(2, 8192), argU(3, 2048), argU(4, 8),
                              argU(5, 64), argU(6, 100));
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
        } else if (mode == "moegcmp") {
            ok = caseMoeGrp(c, argU(2, 0), argU(3, 128), argU(4, 50));
        } else if (mode == "block") {
            ok = caseBlock(c, argU(2, 0), argU(3, 3), argU(4, 200));
        } else if (mode == "ablock") {
            ok = caseABlock(c, argU(2, 3), argU(3, 3), argU(4, 200));
        } else if (mode == "bgemm") {
            ok = caseBGemm(c, argU(2, 8192), argU(3, 2048), argU(4, 128), argU(5, 50));
        } else if (mode == "dncmp") {
            ok = caseDnChunk(c, argU(2, 512), argU(3, 30));
        } else if (mode == "token") {
            if (argc < 4) {
                fprintf(stderr, "usage: qk token <ids-file> <nGen> [tmax]\n");
                return 1;
            }
            ok = caseToken(c, argv[2], argU(3, 12), argU(4, 128));
        } else {
            fprintf(stderr, "mode '%s' not ported to Metal yet "
                            "(M2: q8_0/q6_k/iq4_xs/iq3_xxs; M3: moe/block/ablock; "
                            "M4: token; M5: serve)\n", mode.c_str());
            return 1;
        }
        return ok ? 0 : 1;
    }
}
#endif  // QK_LIBRARY
