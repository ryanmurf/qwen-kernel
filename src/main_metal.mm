// qk — Qwen kernel harness, Metal host (Apple Silicon port of main.cpp).
// M1: fp16 GEMV. M2: quantized GEMV (Q8_0, Q6_K, IQ4_XS, IQ3_XXS) on raw
// ggml blocks, validated against CPU dequant reference and real tensors
// from the GGUF. (M3: fused blocks, M4: token loop, M5: qk.h engine.)
//
// Usage:
//   qk                        synthetic suite: f16, q8_0, q6_k, iq4_xs, iq3_xxs
//   qk f16|q8_0|q6_k|iq4_xs|iq3_xxs [M] [K] [iters]
//   qk slotgemv [M] [K] [B] [TPR] [iters]  Q8_0 batch comparison
//       QK_SLOT_FIXED=1 selects the fixed-K shipping/four-row pair (B=4)
//   qk normroutecmp [experts] [slots] [iters]  split normalized router
//   qk fasrvcmp [slots] [max-pos] [min-pos] [iters]  decode-attention schedules
//   qk fadecode [slots] [pos] [steps] [tmax] [min-pos]  steady deep decode
//   qk counters                 list hardware counter support
//   qk cachecmp [tmax=512]    packed/full prefix-state snapshot comparison
//   qk gguf <tensor> [iters]  real weights (QK_GGUF overrides model path)
//   qk list [filter]          list tensors in the GGUF
//   qk slotcmp <ids> <nGen> [nSlots] [tmax]  slot compaction/churn exactness
//
// Env: QK_DEVICE=<n> device index; QK_SHADER_DIR (dir with *.metal);
//      QK_GGUF=<path>; QK_KV_F16=1 or QK_KV_Q8=1 selects a lossy KV tier;
//      QK_PREFILL_REPACK=1 trades 1.27 GiB for K-major 35B prefill weights.
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
#include <memory>
#include <new>
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
    std::string argv0;
};

static void initMtl(MtlCtx& c, const char* argv0) {
    c.argv0 = argv0 ? argv0 : "";
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

static id<MTLCounterSet> timestampCounterSet(id<MTLDevice> dev) {
    for (id<MTLCounterSet> set in dev.counterSets)
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) return set;
    return nil;
}

static void printCounterInfo(const MtlCtx& c) {
    printf("counter sampling: stage=%d dispatch=%d draw=%d blit=%d\n",
           (int)[c.dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary],
           (int)[c.dev supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary],
           (int)[c.dev supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary],
           (int)[c.dev supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary]);
    for (id<MTLCounterSet> set in c.dev.counterSets) {
        printf("  %s:", set.name.UTF8String);
        for (id<MTLCounter> counter in set.counters)
            printf(" %s", counter.name.UTF8String);
        printf("\n");
    }
}

// M4 exposes only timestamp samples at compute-encoder boundaries.  This
// opt-in helper therefore splits one command buffer into labeled passes.  It
// deliberately allocates no Metal objects and never splits the encoder unless
// QK_COUNTERS=1: normal inference retains its original GPU command stream.
struct StageCounterTrace {
    id<MTLDevice> dev = nil;
    id<MTLCounterSampleBuffer> samples = nil;
    id<MTLFence> pendingFence = nil;
    std::vector<id<MTLFence>> fences;
    NSUInteger next = 0;
    std::vector<const char*> labels;

    explicit StageCounterTrace(MtlCtx& c, NSUInteger sampleCount = 2048) {
        const char* e = getenv("QK_COUNTERS");
        if (!e || atoi(e) == 0) return;
        dev = c.dev;
        if (![c.dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            fprintf(stderr, "[counter] stage-boundary sampling is unsupported\n");
            return;
        }
        id<MTLCounterSet> set = timestampCounterSet(c.dev);
        if (!set) {
            fprintf(stderr, "[counter] timestamp counter set is unavailable\n");
            return;
        }
        MTLCounterSampleBufferDescriptor* desc = [MTLCounterSampleBufferDescriptor new];
        desc.counterSet = set;
        desc.label = @"qk stage timestamps";
        desc.storageMode = MTLStorageModeShared;
        desc.sampleCount = sampleCount;
        NSError* err = nil;
        samples = [c.dev newCounterSampleBufferWithDescriptor:desc error:&err];
        if (!samples)
            fprintf(stderr, "[counter] sample-buffer allocation failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "unknown error");
    }

    bool enabled() const { return samples != nil; }

    id<MTLComputeCommandEncoder> begin(id<MTLCommandBuffer> cb) {
        if (next + 2 > samples.sampleCount) {
            fprintf(stderr, "[counter] sample capacity exhausted at %lu\n",
                    (unsigned long)next);
            exit(1);
        }
        MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
        pass.dispatchType = MTLDispatchTypeConcurrent;
        MTLComputePassSampleBufferAttachmentDescriptor* a =
            [pass.sampleBufferAttachments objectAtIndexedSubscript:0];
        a.sampleBuffer = samples;
        a.startOfEncoderSampleIndex = next++;
        a.endOfEncoderSampleIndex = next++;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:pass];
        if (pendingFence) [enc waitForFence:pendingFence];
        return enc;
    }

    void end(id<MTLComputeCommandEncoder> enc, const char* label) {
        id<MTLFence> fence = [dev newFence];
        [enc updateFence:fence];
        [enc endEncoding];
        fences.push_back(fence);
        pendingFence = fence;
        labels.push_back(label);
    }

    void discardEmpty(id<MTLComputeCommandEncoder> enc) {
        // nextPass starts the following pass eagerly.  A no-logit call has no
        // work after the final residual boundary; Metal may leave that empty
        // pass's timestamps at zero, so exclude its reserved pair.
        [enc endEncoding];
        next -= 2;
    }

    void report(double cbSeconds) const {
        if (!enabled() || labels.empty()) return;
        NSData* data = [samples resolveCounterRange:NSMakeRange(0, next)];
        if (!data || data.length < next * sizeof(MTLCounterResultTimestamp)) {
            fprintf(stderr, "[counter] failed to resolve %lu timestamp samples\n",
                    (unsigned long)next);
            return;
        }
        const auto* ts = (const MTLCounterResultTimestamp*)data.bytes;
        for (NSUInteger i = 0; i < next; ++i) {
            if (ts[i].timestamp == MTLCounterErrorValue || ts[i].timestamp == 0 ||
                (i && ts[i].timestamp < ts[i - 1].timestamp)) {
                fprintf(stderr, "[counter] GPU reported an invalid/non-monotonic "
                                "timestamp at %lu; trace skipped\n",
                        (unsigned long)i);
                return;
            }
        }
        const char* raw = getenv("QK_COUNTER_RAW");
        if (raw && atoi(raw) != 0) {
            fprintf(stderr, "[counter-raw] bytes=%lu samples=%lu stride=%lu\n",
                    (unsigned long)data.length, (unsigned long)next,
                    (unsigned long)(data.length / next));
            for (NSUInteger i = 0; i < std::min<NSUInteger>(next, 24); ++i)
                fprintf(stderr, "[counter-raw] %3lu %20llu\n", (unsigned long)i,
                        (unsigned long long)ts[i].timestamp);
        }
        const uint64_t span = ts[next - 1].timestamp - ts[0].timestamp;
        uint64_t active = 0, gaps = 0;
        std::map<std::string, uint64_t> totals;
        std::map<std::string, uint32_t> counts;
        uint64_t prevEnd = ts[0].timestamp;
        for (size_t i = 0; i < labels.size(); ++i) {
            const uint64_t start = std::max(ts[2 * i].timestamp, prevEnd);
            const uint64_t end = ts[2 * i + 1].timestamp;
            if (start > prevEnd) gaps += start - prevEnd;
            const uint64_t dt = end > start ? end - start : 0;
            active += dt;
            totals[labels[i]] += dt;
            counts[labels[i]]++;
            prevEnd = std::max(prevEnd, end);
        }
        // Apple GPU timestamps are currently nanoseconds.  Calibrate the
        // stage rows to the command-buffer timestamp anyway, so the report is
        // meaningful if a future device uses a different tick period.
        const double scaleMs = span ? cbSeconds * 1e3 / (double)span : 0.0;
        fprintf(stderr,
                "[counter] timestamp-only trace: %zu fenced passes (encoder splitting perturbs schedule)\n"
                "[counter] span %.3f ms | command-buffer %.3f ms | active %.3f ms",
                labels.size(), span * 1e-6, cbSeconds * 1e3, active * scaleMs);
        fprintf(stderr, " | gaps %.3f ms\n", gaps * scaleMs);
        for (const auto& [name, ticks] : totals)
            fprintf(stderr, "[counter] %-14s %8.3f ms  %5.1f%%  (%u passes)\n",
                    name.c_str(), ticks * scaleMs,
                    span ? 100.0 * ticks / (double)span : 0.0, counts[name]);
    }
};

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
// sourcePrefix/sourceTag create a separate source-level variant when a shader
// needs a compile-time define without exposing a mandatory function constant.
static id<MTLComputePipelineState> getPipeVariant(
        MtlCtx& c, const char* file, const char* fn, uint32_t tpr,
        const char* sourcePrefix = nullptr, const char* sourceTag = "") {
    const std::string libKey = std::string(file) + sourceTag;
    std::string key = libKey + ":" + fn + ":" + std::to_string(tpr);
    auto it = c.pipes.find(key);
    if (it != c.pipes.end()) return it->second;

    id<MTLLibrary> lib = nil;
    auto lit = c.libs.find(libKey);
    if (lit != c.libs.end()) {
        lib = lit->second;
    } else {
        std::string src = loadMetalSource(c.argv0.c_str(),
                                          (std::string(file) + ".metal").c_str());
        if (sourcePrefix) src.insert(0, sourcePrefix);
        NSError* err = nil;
        lib = [c.dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                  options:nil
                                    error:&err];
        if (!lib) {
            fprintf(stderr, "MSL compile %s: %s\n", file,
                    err.localizedDescription.UTF8String);
            exit(1);
        }
        c.libs[libKey] = lib;
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

static id<MTLComputePipelineState> getPipe(MtlCtx& c, const char* file,
                                           const char* fn, uint32_t tpr) {
    return getPipeVariant(c, file, fn, tpr);
}

// KV width is a source-level constant so an older host that does not know
// about the precision tiers can compile the same source with the f32 default.
// Current hosts still get a separately compiled, fully constant-folded library
// for each 1/2/4-byte representation.
static id<MTLComputePipelineState> getKvPipe(MtlCtx& c, const char* file,
                                             const char* fn, uint32_t kvBytes) {
    if (kvBytes != 1u && kvBytes != 2u && kvBytes != 4u) {
        fprintf(stderr, "invalid KV byte width %u\n", kvBytes);
        exit(1);
    }
    char prefix[48], tag[16];
    snprintf(prefix, sizeof prefix, "#define QK_KV_BYTES %uu\n", kvBytes);
    snprintf(tag, sizeof tag, ":kv%u", kvBytes);
    return getPipeVariant(c, file, fn, 0, prefix, tag);
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
                    uint32_t fixedNr0 = 0, uint32_t tgMemBytes = 0,
                    const char* shaderFile = nullptr) {
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

    id<MTLComputePipelineState> pso =
        getPipe(c, shaderFile ? shaderFile : kernelName, kernelName, tpr);

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

// Compare the shipping residual+norm+router fusion against an exact two-stage
// form that computes the RMS once and feeds the scale to every router row.
static bool caseNormRouteCmp(MtlCtx& c, uint32_t nExp, uint32_t slots,
                             uint32_t iters) {
    constexpr uint32_t n = 2048u, nFf = 512u, nsg = 4u;
    if ((nExp != 256u && nExp != 512u) || !slots || slots > 16u) {
        fprintf(stderr, "normroutecmp requires experts={256,512}, slots=1..16\n");
        return false;
    }
    std::mt19937 rng(0x4e525445u + nExp + 17u * slots);
    std::normal_distribution<float> nd(0.0f, 0.2f);
    std::vector<float> gi((size_t)nExp * n), gis(n), a((size_t)slots * n),
        b((size_t)slots * n), pn(n);
    for (float& v : gi) v = nd(rng);
    for (float& v : gis) v = nd(rng);
    for (float& v : a) v = nd(rng);
    for (float& v : b) v = nd(rng);
    for (float& v : pn) v = 1.0f + 0.1f * nd(rng);
    id<MTLBuffer> bGi = createBuf(c, gi.size() * 4u, gi.data(), true);
    id<MTLBuffer> bGis = createBuf(c, gis.size() * 4u, gis.data(), true);
    id<MTLBuffer> bA = createBuf(c, a.size() * 4u, a.data(), true);
    id<MTLBuffer> bB = createBuf(c, b.size() * 4u, b.data(), true);
    id<MTLBuffer> bPn = createBuf(c, pn.size() * 4u, pn.data(), true);
    const size_t vecBytes = (size_t)slots * n * 4u;
    const size_t logBytes = (size_t)slots * (nExp + 1u) * 4u;
    id<MTLBuffer> bYA = createBuf(c, vecBytes, nullptr, true);
    id<MTLBuffer> bYB = createBuf(c, vecBytes, nullptr, true);
    id<MTLBuffer> bXA = createBuf(c, vecBytes, nullptr, true);
    id<MTLBuffer> bXB = createBuf(c, vecBytes, nullptr, true);
    id<MTLBuffer> bLA = createBuf(c, logBytes, nullptr, true);
    id<MTLBuffer> bLB = createBuf(c, logBytes, nullptr, true);
    id<MTLBuffer> bScale = createBuf(c, (size_t)slots * 4u, nullptr, true);
    id<MTLComputePipelineState> pFuse =
        getPipe(c, "moe_logits", "moe_logits_addn", nsg);
    id<MTLComputePipelineState> pNorm =
        getPipe(c, "add_rmsnorm", "add_rmsnorm_sg", 0);
    id<MTLComputePipelineState> pLog =
        getPipe(c, "moe_logits", "moe_logits_scaled", nsg);
    struct { uint32_t ne, nf, nx, nu; float eps; }
        pc5{n, nFf, nExp, nExp == 512u ? 10u : 8u, 1e-6f};
    struct { uint32_t ne, nf, nx, nu; } pc4{pc5.ne, pc5.nf, pc5.nx, pc5.nu};
    struct { uint32_t n; float eps; } pcR{n, 1e-6f};
    auto bind = [](id<MTLComputeCommandEncoder> enc,
                   id<MTLComputePipelineState> p,
                   std::initializer_list<id<MTLBuffer>> bufs,
                   const void* pc, size_t pcBytes) {
        [enc setComputePipelineState:p];
        uint32_t i = 0;
        for (id<MTLBuffer> x : bufs) [enc setBuffer:x offset:0 atIndex:i++];
        [enc setBytes:pc length:pcBytes atIndex:i];
    };
    auto encodeFuse = [&](id<MTLComputeCommandEncoder> enc) {
        bind(enc, pFuse, {bGi, bGis, bA, bB, bPn, bLA, bYA, bXA},
             &pc5, sizeof(pc5));
        [enc dispatchThreadgroups:MTLSizeMake((nExp + 1u + nsg - 1u) / nsg, 1, slots)
            threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
    };
    auto encodeSplit = [&](id<MTLComputeCommandEncoder> enc) {
        bind(enc, pNorm, {bA, bB, bPn, bYB, bXB, bScale}, &pcR, sizeof(pcR));
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, slots)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        bind(enc, pLog, {bGi, bGis, bYB, bPn, bScale, bLB}, &pc4, sizeof(pc4));
        [enc dispatchThreadgroups:MTLSizeMake((nExp + 1u + nsg - 1u) / nsg, 1, slots)
            threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
    };
    auto submit = [&](bool split, uint32_t reps) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc =
                [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            for (uint32_t i = 0; i < reps; ++i) {
                if (split) encodeSplit(enc); else encodeFuse(enc);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "normroutecmp Metal failure: %s\n",
                        cb.error.localizedDescription.UTF8String);
                return -1.0;
            }
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps;
        }
    };
    if (submit(false, 1) < 0 || submit(true, 1) < 0) return false;
    auto bitdiff = [](id<MTLBuffer> x, id<MTLBuffer> y, size_t bytes) {
        const uint32_t* xp = (const uint32_t*)x.contents;
        const uint32_t* yp = (const uint32_t*)y.contents;
        uint64_t d = 0;
        for (size_t i = 0; i < bytes / 4u; ++i) d += xp[i] != yp[i];
        return d;
    };
    const uint64_t dy = bitdiff(bYA, bYB, vecBytes);
    const uint64_t dx = bitdiff(bXA, bXB, vecBytes);
    const uint64_t dl = bitdiff(bLA, bLB, logBytes);
    const bool pass = !dy && !dx && !dl;
    printf("\n== normroutecmp experts=%u slots=%u ==\n", nExp, slots);
    printf("  bitdiff y/xn/logits = %llu/%llu/%llu -> %s\n",
           (unsigned long long)dy, (unsigned long long)dx,
           (unsigned long long)dl, pass ? "PASS" : "FAIL");
    if (iters) {
        submit(false, 20); submit(true, 20);
        const double f0 = submit(false, iters), s0 = submit(true, iters);
        const double s1 = submit(true, iters), f1 = submit(false, iters);
        printf("  fused %.3f/%.3f us | split %.3f/%.3f us | %.3fx\n",
               f0, f1, s0, s1, (f0 + f1) / (s0 + s1));
    }
    return pass;
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
        (tpr != 16 && tpr != 32 && tpr != 64 && tpr != 128)) {
        fprintf(stderr, "slotgemv requires K%%32=0, B={2,4,8}, TPR={16,32,64,128}\n");
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
    const bool fixed = getenv("QK_SLOT_FIXED") && atoi(getenv("QK_SLOT_FIXED")) != 0 &&
                       (K == 2048u || K == 4096u) && B == 4u;
    id<MTLComputePipelineState> pRef = fixed
        ? getPipe(c, "gemv_q8_0_fixed", K == 2048u ? "gemv_q8_0_k2048"
                                                   : "gemv_q8_0_k4096", tpr)
        : getPipe(c, "gemv_q8_0", "gemv_q8_0", tpr);
    char fn[32];
    if (fixed) snprintf(fn, sizeof fn, "gemv_q8_0_b%u_k%u", B, K);
    else snprintf(fn, sizeof fn, "gemv_q8_0_b%u", B);
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
    const char* shaderFile = nullptr;
    uint32_t units = K, fixedNr0v = 0, tgMem = 0;
    switch (t->type) {
        case GGML_Q8_0:
            if (getenv("QK_GEMV_FIXED_K") && (K == 2048u || K == 4096u)) {
                kern = K == 2048u ? "gemv_q8_0_k2048" : "gemv_q8_0_k4096";
                shaderFile = "gemv_q8_0_fixed";
            } else {
                kern = "gemv_q8_0";
            }
            units = K / 32;
            break;
        case GGML_Q6_K:    kern = "gemv_q6_k";    units = K / 16; fixedNr0v = 2; break;
        case GGML_IQ4_XS:
            if (const char* v = getenv("QK_IQ4_FIXED_K");
                v && atoi(v) != 0 && (K == 2048u || K == 4096u)) {
                kern = K == 2048u ? "gemv_iq4_xs_k2048" : "gemv_iq4_xs_k4096";
                shaderFile = "gemv_iq4_xs_fixed";
            } else {
                kern = "gemv_iq4_xs";
            }
            units = K / 32; fixedNr0v = 2; tgMem = 128;
            break;
        case GGML_IQ3_XXS: kern = "gemv_iq3_xxs"; units = K / 32; fixedNr0v = 4; tgMem = 1152; break;
        case GGML_F16:     kern = "gemv_f16";     units = K / 8;  break;
        default:
            fprintf(stderr, "no Metal kernel for %s\n", ggmlTypeName(t->type));
            return false;
    }
    return runGemv(c, kern, t->data, (size_t)M * rowBytes, x, M, K, yref, iters,
                   units, 1e-2, fixedNr0v, tgMem, shaderFile);
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
    bool guFixed = guIq3 && nEmbd == 2048u && nFf == 512u;
    if (const char* v = getenv("QK_MOE_GU_FIXED"))
        guFixed = atoi(v) != 0 && guIq3 && nEmbd == 2048u && nFf == 512u;
    const char* guKernel = guFixed
        ? "moe_gateup_all_k2048_ff512"
        : (guIq4 ? "moe_gateup_all_iq4" : "moe_gateup_all");
    id<MTLComputePipelineState> pGu = getPipe(c, "moe_gateup_all", guKernel, nsg);
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

// Direct real-weight comparison of the decode gate+up kernel and its
// K=2048/n_ff=512 specialization, independent of router/down stages.
static bool caseMoeGuCmp(MtlCtx& c, uint32_t layer, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    char name[128];
    auto T = [&](const char* suffix) -> const GgufTensor* {
        snprintf(name, sizeof name, "blk.%u.%s", layer, suffix);
        return g.find(name);
    };
    const GgufTensor* tGE = T("ffn_gate_exps.weight");
    const GgufTensor* tUE = T("ffn_up_exps.weight");
    const GgufTensor* tGS = T("ffn_gate_shexp.weight");
    const GgufTensor* tUS = T("ffn_up_shexp.weight");
    if (!tGE || !tUE || !tGS || !tUS) {
        fprintf(stderr, "moegucmp missing gate/up tensors for layer %u\n", layer);
        return false;
    }
    const bool iq3 = tGE->type == GGML_IQ3_XXS && tUE->type == GGML_IQ3_XXS;
    const uint32_t K = (uint32_t)tGE->ne[0];
    const uint32_t ff = (uint32_t)tGE->ne[1];
    const uint32_t nExp = (uint32_t)tGE->ne[2];
    const uint32_t nUsed = (uint32_t)g.kvInt(
        g.kvStr("general.architecture", "") + ".expert_used_count", 8);
    if (!iq3 || tGS->type != GGML_Q8_0 || tUS->type != GGML_Q8_0 ||
        K != 2048u || ff != 512u || nUsed > 16u) {
        fprintf(stderr, "moegucmp requires IQ3_XXS K=2048, n_ff=512 target shapes\n");
        return false;
    }
    printf("\n== moegucmp blk.%u IQ3_XXS K=%u ff=%u experts=%u top-%u ==\n",
           layer, K, ff, nExp, nUsed);

    std::vector<float> x = randomX(K);
    struct SelH { uint32_t ids[16]; float w[16]; float wShared; float pad[7]; } sel{};
    for (uint32_t s = 0; s < nUsed; ++s) {
        sel.ids[s] = (17u + 29u * s) % nExp;
        sel.w[s] = 1.0f / nUsed;
    }
    sel.wShared = 0.5f;
    const size_t rbE = ggmlRowBytes(tGE->type, K);
    const size_t rbS = ggmlRowBytes(GGML_Q8_0, K);
    const size_t wEBytes = (size_t)nExp * ff * rbE;
    const size_t wSBytes = (size_t)ff * rbS;
    const size_t hCount = (size_t)(nUsed + 1u) * ff;
    id<MTLBuffer> bGE = createBuf(c, wEBytes, tGE->data, true);
    id<MTLBuffer> bUE = createBuf(c, wEBytes, tUE->data, true);
    id<MTLBuffer> bGS = createBuf(c, wSBytes, tGS->data, true);
    id<MTLBuffer> bUS = createBuf(c, wSBytes, tUS->data, true);
    id<MTLBuffer> bX = createBuf(c, (size_t)K * 4u, x.data(), true);
    id<MTLBuffer> bSel = createBuf(c, sizeof(sel), &sel, true);
    id<MTLBuffer> bSafe = createBuf(c, hCount * 4u, nullptr, true);
    id<MTLBuffer> bFixed = createBuf(c, hCount * 4u, nullptr, true);
    const uint32_t nsg = 4u;
    id<MTLComputePipelineState> pSafe =
        getPipe(c, "moe_gateup_all", "moe_gateup_all", nsg);
    id<MTLComputePipelineState> pFixed =
        getPipe(c, "moe_gateup_all", "moe_gateup_all_k2048_ff512", nsg);
    struct { uint32_t k, ff, ne, nu; } pc{K, ff, nExp, nUsed};
    const uint32_t wgs = (uint32_t)((hCount + nsg - 1u) / nsg);

    auto encode = [&](id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState> pso, id<MTLBuffer> h) {
        [enc setComputePipelineState:pso];
        id<MTLBuffer> bufs[] = {bGE, bUE, bGS, bUS, bX, bSel, h};
        for (uint32_t i = 0; i < 7u; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
        [enc setBytes:&pc length:sizeof(pc) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
    };
    auto once = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> h) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode(enc, pso, h);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
    };
    once(pSafe, bSafe);
    once(pFixed, bFixed);
    const uint32_t* a = (const uint32_t*)bSafe.contents;
    const uint32_t* b = (const uint32_t*)bFixed.contents;
    size_t mismatches = 0;
    for (size_t i = 0; i < hCount; ++i) mismatches += a[i] != b[i];
    printf("fixed-vs-safe: %zu/%zu bit mismatches -> %s\n",
           mismatches, hCount, mismatches ? "FAIL" : "EXACT");

    auto run = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> h) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < iters; ++i) encode(enc, pso, h);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / iters;
        }
    };
    if (iters) {
        run(pSafe, bSafe); run(pFixed, bFixed);
        const double s0 = run(pSafe, bSafe), f0 = run(pFixed, bFixed);
        const double f1 = run(pFixed, bFixed), s1 = run(pSafe, bSafe);
        printf("gpu us/iter: safe %.1f / %.1f | fixed %.1f / %.1f\n", s0, s1, f0, f1);
    }
    return mismatches == 0;
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
    const GgufTensor* tDE = T("ffn_down_exps.weight");
    const GgufTensor* tDS = T("ffn_down_shexp.weight");
    if (!tGE || !tUE || !tGS || !tUS || !tDE || !tDS) return false;
    if (tGE->type != GGML_IQ3_XXS || tUE->type != GGML_IQ3_XXS ||
        tGS->type != GGML_Q8_0 || tUS->type != GGML_Q8_0 ||
        (tDE->type != GGML_IQ4_XS && tDE->type != GGML_Q6_K) ||
        tDS->type != GGML_Q8_0) {
        fprintf(stderr, "layer %u tensor types don't match the compiled kernels\n", layer);
        return false;
    }
    const uint32_t nEmbd = (uint32_t)tGE->ne[0];
    const uint32_t nFf   = (uint32_t)tGE->ne[1];
    const uint32_t nExp  = (uint32_t)tGE->ne[2];
    const uint32_t nUsed = 8;
    const uint32_t hs = (nUsed + 1) * nFf;
    const bool hotRoute = getenv("QK_MOEG_HOT") && atoi(getenv("QK_MOEG_HOT")) != 0;
    printf("\n== moegcmp blk.%u  n=%u  n_embd=%u n_ff=%u experts=%u top-%u + shared (%s) ==\n",
           layer, n, nEmbd, nFf, nExp, nUsed, hotRoute ? "hot routes" : "uniform routes");

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
            if (hotRoute) {
                sel[t].ids[s] = s;
            } else {
                uint32_t j = s + rng() % (nExp - s);
                std::swap(pool[s], pool[j]);
                sel[t].ids[s] = pool[s];
            }
            sel[t].w[s] = 0.125f;
        }
        sel[t].wShared = 0.5f;
    }

    const size_t rbGE = ggmlRowBytes(GGML_IQ3_XXS, nEmbd);
    const size_t rbGS = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbDE = ggmlRowBytes((GgmlType)tDE->type, nFf);
    const size_t rbDS = ggmlRowBytes(GGML_Q8_0, nFf);
    const size_t szGE = (size_t)nExp * nFf * rbGE, szGS = (size_t)nFf * rbGS;
    const size_t szDE = (size_t)nExp * nEmbd * rbDE, szDS = (size_t)nEmbd * rbDS;
    id<MTLBuffer> bGE = createBuf(c, szGE, tGE->data, true), bUE = createBuf(c, szGE, tUE->data, true);
    id<MTLBuffer> bGS = createBuf(c, szGS, tGS->data, true), bUS = createBuf(c, szGS, tUS->data, true);
    id<MTLBuffer> bDE = createBuf(c, szDE, tDE->data, true);
    id<MTLBuffer> bDS = createBuf(c, szDS, tDS->data, true);
    id<MTLBuffer> bX = createBuf(c, xs.size() * 4, xs.data(), true);
    id<MTLBuffer> bSel = createBuf(c, (size_t)n * 160, sel.data(), true);
    id<MTLBuffer> bH0 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH1 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH2 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH3 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH4 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH5 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bH6 = createBuf(c, (size_t)n * hs * 4, nullptr, true);
    id<MTLBuffer> bStart = createBuf(c, (size_t)(nExp + 2) * 4, nullptr, true);
    id<MTLBuffer> bATok = createBuf(c, (size_t)n * (nUsed + 1) * 4, nullptr, true);
    id<MTLBuffer> bASlot = createBuf(c, (size_t)n * (nUsed + 1) * 4, nullptr, true);
    const uint32_t workCap = nExp + (n * nUsed + 31) / 32 + (n + 31) / 32;
    id<MTLBuffer> bWork = createBuf(c, (size_t)workCap * 2 * 4, nullptr, true);
    const size_t dCount = (size_t)n * (nUsed + 1) * nEmbd;
    id<MTLBuffer> bD5 = createBuf(c, dCount * 4, nullptr, true);
    id<MTLBuffer> bD6 = createBuf(c, dCount * 4, nullptr, true);

    const uint32_t nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;
    const uint32_t thrN = nsg * 32;
    id<MTLComputePipelineState> pGu  = getPipe(c, "moe_gateup_all", "moe_gateup_all", nsg);
    id<MTLComputePipelineState> pGrp = getPipe(c, "moe_grouped", "moe_group", 0);
    id<MTLComputePipelineState> pGrpWork = getPipe(c, "moe_grouped", "moe_group_work", 0);
    id<MTLComputePipelineState> pG1  = getPipe(c, "moe_grouped", "moe_gu_grouped", nsg);
    id<MTLComputePipelineState> pG2  = getPipe(c, "moe_grouped", "moe_gu_grouped2", 0);
    id<MTLComputePipelineState> pG3  = getPipe(c, "moe_grouped", "moe_gu_grouped3", 0);
    id<MTLComputePipelineState> pG4  = getPipe(c, "moe_grouped", "moe_gu_grouped4", 0);
    id<MTLComputePipelineState> pG5  = getPipe(c, "moe_grouped", "moe_gu_grouped5", 0);
    id<MTLComputePipelineState> pG6  = getPipe(c, "moe_grouped", "moe_gu_grouped5_work", 0);
    const bool downQ6 = tDE->type == GGML_Q6_K;
    id<MTLComputePipelineState> pD5 = getPipe(c, "moe_down_grouped",
        downQ6 ? "moe_down_grouped_p_q6k" : "moe_down_grouped_p_iq4", 0);
    id<MTLComputePipelineState> pD6 = getPipe(c, "moe_down_grouped",
        downQ6 ? "moe_down_grouped_p_work_q6k" : "moe_down_grouped_p_work_iq4", 0);

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
    auto runAll = [&](bool bench) -> std::array<double, 7> {
        std::array<double, 7> ms{0, 0, 0, 0, 0, 0, 0};
        for (int which = 0; which < 7; which++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc =
                    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                for (uint32_t it = 0; it < (bench ? iters : 1); it++) {
                    if (which == 0) {
                        dsp(enc, pGu, {bGE, bUE, bGS, bUS, bX, bSel, bH0}, &pcv, 16,
                            (hs + nsg - 1) / nsg, thrN, n);
                    } else {
                        if (which == 6)
                            dsp(enc, pGrpWork, {bSel, bStart, bATok, bASlot, bWork},
                                &pcg, 20, 1, 256, 1);
                        else
                            dsp(enc, pGrp, {bSel, bStart, bATok, bASlot},
                                &pcg, 20, 1, 256, 1);
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
                        else if (which == 5)
                            dsp(enc, pG5, {bGE, bUE, bGS, bUS, bX, bStart, bATok, bASlot, bH5},
                                &pcv, 16, (nExp + 1) * (nFf / 32), 128, (n + 31) / 32);
                        else
                            dsp(enc, pG6,
                                {bGE, bUE, bGS, bUS, bX, bStart, bATok, bASlot, bH6, bWork},
                                &pcv, 16, workCap * (nFf / 32), 128, 1);
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
    const float* h6 = (const float*)bH6.contents;
    double rms = 0;
    for (size_t i = 0; i < (size_t)n * hs; i++) rms += (double)h0[i] * h0[i];
    rms = std::sqrt(rms / ((size_t)n * hs));
    const double floorD = std::max(1e-4, 1e-3 * rms);
    uint32_t diff1 = 0, diff6 = 0;
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
        if (h6[i] != h5[i]) diff6++;
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
    printf("v6 (v5 compact work):   %u/%zu entries differ from v5 -> %s\n",
           diff6, (size_t)n * hs, diff6 ? "FAIL" : "PASS");

    if (iters > 0) {
        runAll(true);
        auto ms = runAll(true);
        printf("bench (%u iters): ungrouped %.3f ms | v1 %.3f ms | v2 %.3f ms | v3 %.3f ms | v4 %.3f ms | v5 %.3f ms | v6 %.3f ms\n",
               iters, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], ms[6]);
    }
    auto runDown = [&](bool bench) -> std::array<double, 2> {
        std::array<double, 2> ms{0, 0};
        for (uint32_t which = 0; which < 2; ++which) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc =
                    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                for (uint32_t it = 0; it < (bench ? iters : 1); ++it) {
                    if (which == 0)
                        dsp(enc, pD5, {bDE, bDS, bH5, bStart, bATok, bASlot, bD5},
                            &pcv, 16, (nExp + 1) * (nEmbd / 64), 128, (n + 31) / 32);
                    else
                        dsp(enc, pD6,
                            {bDE, bDS, bH5, bStart, bATok, bASlot, bD6, bWork},
                            &pcv, 16, workCap * (nEmbd / 64), 128, 1);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                ms[which] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 /
                            (bench ? iters : 1);
            }
        }
        return ms;
    };
    runDown(false);
    const float* d5 = (const float*)bD5.contents;
    const float* d6 = (const float*)bD6.contents;
    size_t diffD = 0;
    for (size_t i = 0; i < dCount; ++i)
        if (d5[i] != d6[i]) ++diffD;
    printf("down v6 vs v5 (%s):     %zu/%zu entries differ -> %s\n",
           downQ6 ? "Q6_K" : "IQ4_XS", diffD, dCount, diffD ? "FAIL" : "PASS");
    if (iters > 0) {
        runDown(true);
        auto ms = runDown(true);
        printf("down bench (%u iters): v5 %.3f ms | v6 %.3f ms\n",
               iters, ms[0], ms[1]);
    }
    return diff1 == 0 && diff6 == 0 && diffD == 0;
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
    const bool dnDecodeOpt = !getenv("QK_DN_DECODE") ||
                             atoi(getenv("QK_DN_DECODE")) != 0;
    const bool dnDecodeResident = dnDecodeOpt &&
        (!getenv("QK_DN_DECODE_RES") || atoi(getenv("QK_DN_DECODE_RES")) != 0);
    const char* dnStepFn = dnDecodeResident ? "dn_step_res8" : "dn_step";
    id<MTLComputePipelineState> pStep = getPipe(c, "dn_step", dnStepFn, 0);
    int dnDecodeSplitMode = dnDecodeOpt ? -1 : 0;
    if (const char* v = getenv("QK_DN_DECODE_SPLIT")) dnDecodeSplitMode = atoi(v) != 0;
    const bool dnDecodeSimd = getenv("QK_DN_DECODE_SIMD") &&
                              atoi(getenv("QK_DN_DECODE_SIMD")) != 0;
    if (dnDecodeSimd) dnDecodeSplitMode = 1;
    const bool dnDecodeSplit = dnDecodeSplitMode != 0;  // single-slot harness
    id<MTLComputePipelineState> pStepPrep = dnDecodeSplit
        ? getPipe(c, "dn_step", "dn_step_prep", 0) : nil;
    const char* dnStateFn = dnDecodeResident ? "dn_step_state_res8" : "dn_step_state";
    if (dnDecodeSimd) dnStateFn = "dn_step_state_rowsimd";
    id<MTLComputePipelineState> pStepState = dnDecodeSplit
        ? getPipe(c, "dn_step", dnStateFn, 0) : nil;
    id<MTLComputePipelineState> pGemvO = getPipe(c, "gemv_q8_0", "gemv_q8_0", 128);  // K = d_inner
    id<MTLComputePipelineState> pAddN  = getPipe(c, "add_rmsnorm", "add_rmsnorm", 0);
    id<MTLComputePipelineState> pAdd   = getPipe(c, "vec_add", "vec_add", 0);
    id<MTLComputePipelineState> pMoeL  = getPipe(c, "moe_logits", "moe_logits", nsg);
    id<MTLComputePipelineState> pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    id<MTLComputePipelineState> pMoeLA = getPipe(c, "moe_logits", "moe_logits_addn", nsg);
    bool moeGuFixed = !moe.guIq4 && moe.nFf == 512u;
    if (const char* v = getenv("QK_MOE_GU_FIXED"))
        moeGuFixed = atoi(v) != 0 && !moe.guIq4 && moe.nFf == 512u;
    id<MTLComputePipelineState> pMoeGu = getPipe(
        c, "moe_gateup_all",
        moeGuFixed ? "moe_gateup_all_k2048_ff512" : "moe_gateup_all", nsg);
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
        if (dnDecodeSplit) {
            dsp(enc, pStepPrep, {bQkv, bConvSt, bKer}, &pcStep, 20, hK, dS);
            bar(enc);
            dsp(enc, pStepState, {bQkv, bGb, bS, bZ, bSN, bAtt},
                &pcStep, 20, hV, dS);
        } else {
            dsp(enc, pStep, {bQkv, bConvSt, bKer, bGb, bS, bZ, bSN, bAtt},
                &pcStep, 20, hK, dS);
        }
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
        printf("gpu: %8.1f µs/block | %u dispatches, 1 submit\n",
               ns / 1e3, dnDecodeSplit ? 12u : 11u);
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
    const bool kvQ8 = getenv("QK_KV_Q8") && atoi(getenv("QK_KV_Q8")) != 0;
    const bool kvF16 = !kvQ8 && getenv("QK_KV_F16") &&
                       atoi(getenv("QK_KV_F16")) != 0;
    const uint32_t kvBytes = kvQ8 ? 1u : kvF16 ? 2u : 4u;
    const uint32_t kvRowBytes = kvQ8 ? 272u : dh * kvBytes;
    const float eps = 1e-6f;
    printf("\n== ablock blk.%u (full attn)  n_embd=%u heads=%ux%u kv=%u rot=%u %s ==\n",
           layer, nEmbd, hQ, dh, hKV, nRot,
           kvQ8 ? "KV=q8-row" : kvF16 ? "KV=f16" : "KV=f32");

    const size_t rbQ8e = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbQ8a = ggmlRowBytes(GGML_Q8_0, atN);

    const uint32_t nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;
    id<MTLComputePipelineState> pRms   = getPipe(c, "rmsnorm", "rmsnorm", 0);
    id<MTLComputePipelineState> pGemvA = getPipe(c, "gemv_q8_0", "gemv_q8_0", 64);
    id<MTLComputePipelineState> pPrep  = getKvPipe(c, "fa_prep", "fa_prep", kvBytes);
    id<MTLComputePipelineState> pAttn  = getKvPipe(c, "fa_attn", "fa_attn", kvBytes);
    id<MTLComputePipelineState> pGemvO = getPipe(c, "gemv_q8_0", "gemv_q8_0", 128);
    id<MTLComputePipelineState> pAddN  = getPipe(c, "add_rmsnorm", "add_rmsnorm", 0);
    id<MTLComputePipelineState> pAdd   = getPipe(c, "vec_add", "vec_add", 0);
    id<MTLComputePipelineState> pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    id<MTLComputePipelineState> pMoeLA = getPipe(c, "moe_logits", "moe_logits_addn", nsg);
    bool moeGuFixed = !moe.guIq4 && moe.nFf == 512u;
    if (const char* v = getenv("QK_MOE_GU_FIXED"))
        moeGuFixed = atoi(v) != 0 && !moe.guIq4 && moe.nFf == 512u;
    id<MTLComputePipelineState> pMoeGu = getPipe(
        c, "moe_gateup_all",
        moeGuFixed ? "moe_gateup_all_k2048_ff512" : "moe_gateup_all", nsg);
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
    id<MTLBuffer> bKC = createBuf(c, (size_t)hKV * tmax * kvRowBytes, nullptr, true);
    id<MTLBuffer> bVC = createBuf(c, (size_t)hKV * tmax * kvRowBytes, nullptr, true);
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
    auto q8RowRef = [&](float* v) {
        float amax = 0.0f;
        for (uint32_t j = 0; j < dh; ++j) amax = std::max(amax, std::fabs(v[j]));
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        for (uint32_t j = 0; j < dh; ++j) {
            const int q = d > 0.0f
                ? std::clamp((int)std::lrint(v[j] / d), -127, 127) : 0;
            v[j] = d * (float)q;
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
            float* vd = &vcCpu[((size_t)h * tmax + t) * dh];
            if (kvQ8) {
                memcpy(vd, &vin[(size_t)h * dh], dh * 4);
                q8RowRef(kd);
                q8RowRef(vd);
            } else if (kvF16) {
                for (uint32_t j = 0; j < dh; ++j) {
                    kd[j] = qk_f16_to_f32(qk_f32_to_f16(kd[j]));
                    vd[j] = qk_f16_to_f32(qk_f32_to_f16(vin[(size_t)h * dh + j]));
                }
            } else {
                memcpy(vd, &vin[(size_t)h * dh], dh * 4);
            }
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
    const bool kvQ8 = getenv("QK_KV_Q8") && atoi(getenv("QK_KV_Q8")) != 0;
    const bool kvF16 = !kvQ8 && getenv("QK_KV_F16") &&
                       atoi(getenv("QK_KV_F16")) != 0;
    const uint32_t kvBytes = kvQ8 ? 1u : kvF16 ? 2u : 4u;
    const uint32_t kvRowBytes = kvQ8 ? 272u : dh * kvBytes;
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
    printf("token mode: %zu prompt ids, gen %u, vocab %u, tmax %u, %s KV\n",
           promptIds.size(), nGen, vocab, tmax,
           kvQ8 ? "q8-row" : kvF16 ? "f16" : "f32");

    const uint32_t nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;
    id<MTLComputePipelineState> pRms   = getPipe(c, "rmsnorm", "rmsnorm", 0);
    id<MTLComputePipelineState> pGemvA = getPipe(c, "gemv_q8_0", "gemv_q8_0", 64);
    id<MTLComputePipelineState> pAb    = getPipe(c, "dn_ab", "dn_ab", nsg);
    const bool dnDecodeOpt = !getenv("QK_DN_DECODE") ||
                             atoi(getenv("QK_DN_DECODE")) != 0;
    const bool dnDecodeResident = dnDecodeOpt &&
        (!getenv("QK_DN_DECODE_RES") || atoi(getenv("QK_DN_DECODE_RES")) != 0);
    const char* dnStepFn = dnDecodeResident ? "dn_step_res8" : "dn_step";
    id<MTLComputePipelineState> pStep = getPipe(c, "dn_step", dnStepFn, 0);
    int dnDecodeSplitMode = dnDecodeOpt ? -1 : 0;
    if (const char* v = getenv("QK_DN_DECODE_SPLIT")) dnDecodeSplitMode = atoi(v) != 0;
    const bool dnDecodeSimd = getenv("QK_DN_DECODE_SIMD") &&
                              atoi(getenv("QK_DN_DECODE_SIMD")) != 0;
    if (dnDecodeSimd) dnDecodeSplitMode = 1;
    const bool dnDecodeSplit = dnDecodeSplitMode != 0;  // token harness is one stream
    id<MTLComputePipelineState> pStepPrep = dnDecodeSplit
        ? getPipe(c, "dn_step", "dn_step_prep", 0) : nil;
    const char* dnStateFn = dnDecodeResident ? "dn_step_state_res8" : "dn_step_state";
    if (dnDecodeSimd) dnStateFn = "dn_step_state_rowsimd";
    id<MTLComputePipelineState> pStepState = dnDecodeSplit
        ? getPipe(c, "dn_step", dnStateFn, 0) : nil;
    id<MTLComputePipelineState> pGemvO = getPipe(c, "gemv_q8_0", "gemv_q8_0", 128);
    id<MTLComputePipelineState> pAddN  = getPipe(c, "add_rmsnorm", "add_rmsnorm", 0);
    id<MTLComputePipelineState> pPrep  = getKvPipe(c, "fa_prep", "fa_prep", kvBytes);
    id<MTLComputePipelineState> pAttn  = getKvPipe(c, "fa_attn", "fa_attn", kvBytes);
    id<MTLComputePipelineState> pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    id<MTLComputePipelineState> pMoeLA = getPipe(c, "moe_logits", "moe_logits_addn", nsg);
    bool moeGuFixed = ffE == 512u;
    if (const char* v = getenv("QK_MOE_GU_FIXED"))
        moeGuFixed = atoi(v) != 0 && ffE == 512u;
    id<MTLComputePipelineState> pMoeGu = getPipe(
        c, "moe_gateup_all",
        moeGuFixed ? "moe_gateup_all_k2048_ff512" : "moe_gateup_all", nsg);
    id<MTLComputePipelineState> pMoeD4 = getPipe(c, "moe_down_all", "moe_down_all_iq4", nsg);
    id<MTLComputePipelineState> pMoeD6 = getPipe(c, "moe_down_all", "moe_down_all_q6k", nsg);
    bool headFixedK = true;
    if (const char* v = getenv("QK_HEAD_FIXED_K")) headFixedK = atoi(v) != 0;
    id<MTLComputePipelineState> pHead = headFixedK
        ? getPipe(c, "gemv_q6_k_fixed", "gemv_q6_k_k2048", 2)
        : getPipe(c, "gemv_q6_k", "gemv_q6_k", 2);
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
            L.kc = W(nullptr, (size_t)hKV * tmax * kvRowBytes);
            L.vc = W(nullptr, (size_t)hKV * tmax * kvRowBytes);
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
                if (dnDecodeSplit) {
                    dsp(enc, pStepPrep, {bBig, L.convSt, L.ker},
                        &pcStep, 20, hK, dS);
                    bar(enc);
                    dsp(enc, pStepState, {bBig, bGb, L.S, bMid, L.sn, bAtt},
                        &pcStep, 20, hV, dS);
                } else {
                    dsp(enc, pStep, {bBig, L.convSt, L.ker, bGb, L.S, bMid, L.sn, bAtt},
                        &pcStep, 20, hK, dS);
                }
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
    bool prefillMulti = true;
    bool slotCompact = true;
    bool pcacheCompact = true;
    bool kvF16 = false;          // explicit lossy capacity tier; f32 compute
    bool kvQ8 = false;           // aligned row-Q8 storage; f32 dequant/compute
    uint32_t kvBytes = 4;
    uint32_t kvRowBytes = dh * 4;
    bool slotsNeedCompact = false;
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
        pPrefillGemvA4, pPrefillGemvO4,
        pAb, pStep, pStepPrep, pStepState, pStepStateSimd, pPrep, pAttn,
        pAttnSplit, pAttnReduce, pMoeS,
        pMoeLA, pMoeLS, pAddNSg, pMoeGu, pMoeD4, pMoeD6, pAddN, pHead, pHeadTop,
        pHeadTop64, pHeadF8, pHeadF16, pHeadF32, pHeadTopReduce, pAm1, pAm2, pEmb,
        pPrepB, pAttnB, pAttnBM, pAbB, pConvB, pStepB, pGateB,
        pGemmB, pGemmBAligned, pGemmBK;
    // Adaptive decode attention either coarsens f32 producer scheduling or
    // shares compact-cache reads across GQA heads. Both retain one original
    // 256-key partial per query head/chunk and the shipping reducer.
    id<MTLComputePipelineState> pAttnCompact, pAttnMulti,
        pAttnQ8G2, pAttnQ8G4, pAttnQ8G8,
        pAttnF16G2, pAttnF16G4, pAttnF16G8,
        pAttnF32G2, pAttnF32G4, pAttnF32G8;
    bool faDecodeSplit = true, faDecodeWork = true;
    bool faF16Gqa = true, faF32Gqa = true;
    int faFanOverride = -1;
    int moeNormSplitMode = -1;  // -1 auto, 0 fused recompute, 1 split/reuse
    // IQ4_XS twins for the 80B: dense-proj gemv/gemm + routed gate/up.
    id<MTLComputePipelineState> pGemv4, pGemv4O, pGemmB4, pGemmB4Aligned,
        pMoeGu4, pMoeGuG4i, pMoeGuG5i;
    // chunked delta rule (prefill DN): parallel kq/solve + chunk-serial step
    id<MTLComputePipelineState> pDnKq, pDnSolve, pDnStepC;
    id<MTLBuffer> bbDnKQ, bbDnUW, bbDnAtt, bbDnEl;
    bool dnChunk = true;   // QK_DN_CHUNK=0 falls back to dn_step_batch
    uint32_t dnStepPW = 8;  // resident row-panel width; 0 restores streamed control
    int dnDecodeSplitMode = -1;  // -1 auto schedule, 0 fused rollback, 1 forced split
    // DeltaNet GQA k-pairing: 0 = h % hK (qwen35moe), else h / kDiv
    // (qwen3next: hV/hK = 2). Set from general.architecture at open.
    uint32_t dnKDiv = 0;
    std::vector<id<MTLBuffer>> extraBufs;   // host-built weights (ssm_ba split)
    struct WeightWin { const uint8_t* base; size_t len; id<MTLBuffer> buf; };
    std::vector<WeightWin> gwin;            // no-copy windows when the file > maxBufferLength
    uint32_t gemmThreads = 256;
    uint32_t gemmBM = 128, gemmBN = 64;
    bool prefillGemvBatch = false;
    bool gemmAligned = true;
    uint32_t faQTM = 8, faThreads = 256, faHeadGroup = 2;
    bool headGemm = true;
    uint32_t headGemmN = 4;
    bool headF16 = false;  // opt-in half operands/f32 accumulation for batched head
    bool headFixedK = true;
    bool slotBatch = false;
    uint32_t slotTprA = 64, slotTprO = 128;
    uint32_t gemvTprA = 16, gemvTprO = 128;
    bool gemvFast = true;
    bool iq4FixedK = true;
    uint32_t slotBatchN = 0;
    bool prefillRepack = false;
    bool prefillRepackUse = true;

    struct Layer {
        bool rec = false, downQ6 = false;
        // per-projection quant flags: true = IQ4_XS weight (dispatch the *4
        // pipe). p1 = qkv|q, p2 = z|k, p3 = v, wo = ssm_out|attn_output.
        bool iq4P1 = false, iq4P2 = false, iq4P3 = false, iq4Wo = false;
        WB aNorm{nil}, pn{nil};
        WB qkvW{nil}, zW{nil}, alW{nil}, beW{nil}, dt{nil}, av{nil}, ker{nil},
           sn{nil}, outW{nil};                                       // rec
        WB wq{nil}, wk{nil}, wv{nil}, qn{nil}, kn{nil}, wo{nil};     // attn
        // Prefill-only K-major copies. Decode keeps the native GGUF row-major
        // tensors above so every GEMV continues to stream complete rows.
        WB qkvP{nil}, zP{nil}, outP{nil}, wqP{nil}, wkP{nil}, wvP{nil}, woP{nil};
        WB mgi{nil}, mgis{nil}, mge{nil}, mue{nil}, mde{nil}, mgs{nil},
           mus{nil}, mds{nil};                                       // moe
        id<MTLBuffer> st1, st2;      // per-slot state: rec=(conv,S) attn=(kc,vc)
        size_t ps1 = 0, ps2 = 0;     // per-slot byte stride
        size_t carryOff = 0;          // compact offset in batched-prefill conv carry
    };
    std::vector<Layer> layers;

    id<MTLBuffer> bXin, bXn, bBig, bMid, bKin, bVin, bGb, bAtt, bAttnOut, bY, bXn2,
        bML, bMH, bMSel, bMY, bLogits, bRope, bAV, bAI,
        bTok, bRbScratch, bSlotIn, bSlotPos, bPartial, bFaWork, bMoeScale;
    WB bONorm{nil}, bHeadW{nil}, bEmbdW{nil};

    uint32_t maxB = 0;
    double lastCbGpu = 0, lastCbWall = 0;
    id<MTLBuffer> bbXin, bbXn, bbBig, bbMid, bbKin, bbVin, bbGb, bbConvOut, bbO,
        bbAtt, bbAttnOut, bbY, bbXn2, bbML, bbMH, bbMSel, bbMY, bbLogits, bbIds, bbCarry,
        bbAV, bbAI, bbTok, bbHeadV, bbHeadI;
    id<MTLComputePipelineState> pRms, pMoeGrp, pMoeGrpLive, pMoeGrpWork,
        pMoeGuG, pMoeGuG2, pMoeGuG3, pMoeGuG4, pMoeGuG4Live,
        pMoeGuG5, pMoeGuG5Live, pMoeGuG5Work,
        pMoeDG4, pMoeDG6, pMoeDG4Live, pMoeDG6Live,
        pMoeDGH4, pMoeDGH6, pMoeDGP4, pMoeDGP6,
        pMoeDGP4Live, pMoeDGP6Live, pMoeDGP4Work, pMoeDGP6Work,
        pMoeDR, pLogG;
    id<MTLBuffer> bbStart, bbATok, bbASlot, bbLive, bbWork, bbMDy;
    // Grouped (decode-once) MoE gate+up for prefill chunks. Variants:
    // 1 = v1 read-once (SLOWER — kept as bit-exact control); 2 = v2 f32
    // narrow; 3 = v3 f16 64x32 (llama mul_mm_id class — fastest, opt-in);
    // 4 = v4 f32 32x32 (exact-class, default); 5 = packed-f16 with compact
    // (expert,token-tile) work pairs (`QK_MOE_WORK=0` restores grid-z);
    // 6 is retained as an explicit compact-work alias.
    // Grouping only pays once
    // experts see enough tokens: default fires at n >= moeGroupN (192) with
    // variant 4; QK_MOE_GROUPED forces a variant at ALL n (0 disables),
    // QK_MOE_GROUP_N overrides the threshold.
    int moeGrouped = 4;
    uint32_t moeGroupN = 192;
    bool moeWork = true;     // compact packed-v5 prefill tiles; env 0 is rollback
    int moeSlotGrouped = 0;  // 35B default v4; env 0..5 selects/rolls back
    int moeSlotParts = 3;    // bit 0=gate/up, bit 1=down
    bool moeSlotLive = true;
    uint32_t moeSlotMin = 8;
    bool moeSlotSameOnly = true;

    struct Slot {
        bool active = false;
        // Public slot ids are stable client identities. Unsplit serving packs
        // live model state into physical stripes [0,nActive), avoiding decode
        // work for holes left by staggered completion/cancellation.
        uint32_t phys = UINT32_MAX;
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
    bool specOn = false, specScratch = false;
    uint32_t specL = 6, specK = 8;
    std::vector<uint32_t> specToks, specAm;
    uint32_t specDraft(uint32_t slot);

    struct CacheEntry {
        std::vector<uint32_t> tokens;
        // Packed host snapshot: recurrent stripes in full, then only the live
        // K/V prefix for each attention kv-head. Allocated on first save and
        // resized on overwrite, so empty cache entries cost no payload memory.
        std::unique_ptr<uint8_t[]> snap;
        size_t snapBytes = 0;
        uint32_t snapTok = 0;
        uint64_t lru = 0;
        bool valid = false;
    };
    std::vector<CacheEntry> pcache;
    std::vector<size_t> snapOff1, snapOff2;
    size_t snapSize = 0;
    uint64_t lruClock = 0;

    bool open(const char* path, const qk_config& cfg, char* err, size_t errLen);
    int stepChunk(uint32_t* outTok, uint32_t* outCnt, uint32_t* outFin);
    void prefillPendingSlots();
    void prefillBatchLast(const uint32_t* toks, uint32_t n, uint32_t slot,
                          std::vector<float>& logits, bool wantLogits = true, uint32_t base = 0,
                          uint32_t* argmaxOut = nullptr, const float* hiddenIn = nullptr,
                          float* hiddenOut = nullptr, bool scratchState = false,
                          uint32_t nSeq = 1, const uint32_t* seqSlots = nullptr,
                          const uint32_t* seqBases = nullptr);
    int stageRun(uint32_t slot, const uint32_t* toks, const float* hiddenIn, uint32_t n,
                 uint32_t base, float* hiddenOut, uint32_t* idsOut);
    // Top-k (ids, logits) of the final position's row after a last-stage
    // stageRun — the split driver's sampling hook (see qk_stage_topk in qk.h).
    int stageTopK(uint32_t k, uint32_t* idsOut, float* valsOut);
    uint32_t serialPrefillLogits(const uint32_t* toks, uint32_t n, uint32_t slot,
                                 std::vector<float>& logits);
    void resetStripe(uint32_t stripe);
    void snapshotSlot(uint32_t slot);
    int matchPrefix(const uint32_t* prompt, uint32_t n);
    void restoreInto(uint32_t slot, int cacheIdx);
    void copyStripes(uint32_t stripe, uint8_t* snap, bool save, uint32_t nTok = 0);
    size_t packedSnapSize(uint32_t nTok) const;
    bool copyPackedStripes(uint32_t stripe, CacheEntry& entry, bool save, uint32_t nTok);
    uint32_t physicalSlot(uint32_t slot) const;
    uint32_t acquirePhysicalSlot(uint32_t slot);
    void moveSlotState(uint32_t fromStripe, uint32_t toStripe, uint32_t nTok);
    void compactActiveSlots();
    // The recurrent scratch stripe is index nSlots. Verification advances
    // DeltaNet/conv there while attention KV writes remain on the live slot.
    void copyDnStripes(uint32_t fromStripe, uint32_t toStripe);
    void verifyRound(const uint32_t* toks, uint32_t n, uint32_t slot, uint32_t base,
                     uint32_t* outIds);
    void promoteScratch(uint32_t slot) { copyDnStripes(nSlots, physicalSlot(slot)); }
    void encodeStep(id<MTLComputeCommandEncoder> enc, uint32_t zdim,
                    bool equivalentSlots = false);
};

void qk_engine::resetStripe(uint32_t stripe) {
    for (auto& L : layers) {   // GPU is idle at every call site (engine is synchronous)
        if (!L.st1) continue;  // layer outside this stage's [lFirst, lEnd)
        memset((uint8_t*)L.st1.contents + (size_t)stripe * L.ps1, 0, L.ps1);
        memset((uint8_t*)L.st2.contents + (size_t)stripe * L.ps2, 0, L.ps2);
    }
}

void qk_engine::copyStripes(uint32_t stripe, uint8_t* snap, bool save, uint32_t nTok) {
    for (uint32_t il = 0; il < nLayer; il++) {
        Layer& L = layers[il];
        if (!L.st1) continue;
        uint8_t* s1 = (uint8_t*)L.st1.contents + (size_t)stripe * L.ps1;
        uint8_t* s2 = (uint8_t*)L.st2.contents + (size_t)stripe * L.ps2;
        // Attention KV stripes are [hKV][tmax][dh]: with a live token count,
        // copy only each kv-heads first nTok rows so snapshot cost and
        // resident pages track the conversation, not capacity. Recurrent
        // (DeltaNet/conv) stripes always copy whole.
        if (!L.rec && nTok && nTok < nCtx) {
            size_t headBytes = L.ps1 / hKV;
            size_t liveBytes = (size_t)nTok * kvRowBytes;
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

size_t qk_engine::packedSnapSize(uint32_t nTok) const {
    const uint32_t liveTok = nTok ? std::min(nTok, nCtx) : nCtx;
    size_t bytes = 0;
    for (const Layer& L : layers) {
        if (!L.st1) continue;
        if (L.rec) bytes += L.ps1 + L.ps2;
        else bytes += (size_t)2 * hKV * liveTok * kvRowBytes;
    }
    return bytes;
}

bool qk_engine::copyPackedStripes(uint32_t stripe, CacheEntry& entry,
                                  bool save, uint32_t nTok) {
    const uint32_t liveTok = nTok ? std::min(nTok, nCtx) : nCtx;
    if (!pcacheCompact) {
        if (save) {
            if (!entry.snap || entry.snapBytes != snapSize) {
                std::unique_ptr<uint8_t[]> fresh(new (std::nothrow) uint8_t[snapSize]);
                if (!fresh) return false;
                entry.snap = std::move(fresh);
                entry.snapBytes = snapSize;
            }
            entry.snapTok = liveTok;
        } else if (!entry.snap || entry.snapBytes != snapSize || liveTok > entry.snapTok) {
            return false;
        }
        copyStripes(stripe, entry.snap.get(), save, nTok);
        return true;
    }
    if (save) {
        const size_t bytes = packedSnapSize(liveTok);
        if (!entry.snap || entry.snapBytes != bytes) {
            std::unique_ptr<uint8_t[]> fresh(new (std::nothrow) uint8_t[bytes]);
            if (!fresh) return false;
            entry.snap = std::move(fresh);
            entry.snapBytes = bytes;
        }
        entry.snapTok = liveTok;
    } else if (!entry.snap || liveTok > entry.snapTok ||
               entry.snapBytes != packedSnapSize(entry.snapTok)) {
        return false;
    }

    uint8_t* p = entry.snap.get();
    const size_t storedLiveBytes = (size_t)entry.snapTok * kvRowBytes;
    const size_t copyLiveBytes = (size_t)liveTok * kvRowBytes;
    for (Layer& L : layers) {
        if (!L.st1) continue;
        uint8_t* s1 = (uint8_t*)L.st1.contents + (size_t)stripe * L.ps1;
        uint8_t* s2 = (uint8_t*)L.st2.contents + (size_t)stripe * L.ps2;
        if (L.rec) {
            if (save) {
                memcpy(p, s1, L.ps1); p += L.ps1;
                memcpy(p, s2, L.ps2); p += L.ps2;
            } else {
                memcpy(s1, p, L.ps1); p += L.ps1;
                memcpy(s2, p, L.ps2); p += L.ps2;
            }
            continue;
        }
        const size_t headStride = L.ps1 / hKV;
        for (uint8_t* state : {s1, s2}) {
            for (uint32_t h = 0; h < hKV; ++h) {
                if (save) memcpy(p, state + (size_t)h * headStride, storedLiveBytes);
                else memcpy(state + (size_t)h * headStride, p, copyLiveBytes);
                p += storedLiveBytes;
            }
        }
    }
    if ((size_t)(p - entry.snap.get()) != entry.snapBytes) {
        fprintf(stderr, "packed prefix snapshot size invariant failed\n");
        abort();
    }
    return true;
}

uint32_t qk_engine::physicalSlot(uint32_t slot) const {
    if (splitStage() || !slotCompact) return slot;
    if (slot >= nSlots || slots[slot].phys >= nSlots) {
        fprintf(stderr, "physicalSlot: logical slot %u has no live stripe\n", slot);
        abort();
    }
    return slots[slot].phys;
}

void qk_engine::moveSlotState(uint32_t fromStripe, uint32_t toStripe, uint32_t nTok) {
    if (fromStripe == toStripe) return;
    // Compaction always moves toward a lower, vacant stripe while the GPU is
    // idle. Recurrent state is fixed-size; attention only needs the live KV
    // prefix, not the potentially enormous configured context capacity.
    for (Layer& L : layers) {
        if (!L.st1) continue;
        uint8_t* from1 = (uint8_t*)L.st1.contents + (size_t)fromStripe * L.ps1;
        uint8_t* from2 = (uint8_t*)L.st2.contents + (size_t)fromStripe * L.ps2;
        uint8_t* to1 = (uint8_t*)L.st1.contents + (size_t)toStripe * L.ps1;
        uint8_t* to2 = (uint8_t*)L.st2.contents + (size_t)toStripe * L.ps2;
        if (L.rec) {
            memcpy(to1, from1, L.ps1);
            memcpy(to2, from2, L.ps2);
        } else if (nTok) {
            const size_t headBytes = L.ps1 / hKV;
            const size_t liveBytes =
                std::min((size_t)nTok, (size_t)nCtx) * kvRowBytes;
            for (uint32_t h = 0; h < hKV; ++h) {
                const size_t off = (size_t)h * headBytes;
                memcpy(to1 + off, from1 + off, liveBytes);
                memcpy(to2 + off, from2 + off, liveBytes);
            }
        }
    }
}

void qk_engine::compactActiveSlots() {
    if (splitStage() || !slotCompact || !slotsNeedCompact) return;
    std::array<int32_t, 16> owner;
    owner.fill(-1);
    uint32_t nLive = 0;
    for (uint32_t s = 0; s < nSlots; ++s) {
        const Slot& sl = slots[s];
        if (!sl.active || sl.finPending) continue;
        if (sl.phys >= nSlots || owner[sl.phys] >= 0) {
            fprintf(stderr, "compactActiveSlots: invalid/duplicate stripe for slot %u\n", s);
            abort();
        }
        owner[sl.phys] = (int32_t)s;
        ++nLive;
    }
    // Fill each hole below nLive with one high stripe. This changes physical
    // row order (which is fully mapped) but copies at most one state per hole,
    // instead of shifting every later client and its potentially large KV.
    for (uint32_t target = 0; target < nLive; ++target) {
        if (owner[target] >= 0) continue;
        uint32_t source = nSlots;
        while (source > nLive) {
            --source;
            if (owner[source] >= 0) break;
        }
        if (source < nLive || owner[source] < 0) {
            fprintf(stderr, "compactActiveSlots: no source for hole %u/%u\n", target, nLive);
            abort();
        }
        Slot& sl = slots[(uint32_t)owner[source]];
        // During serial prompt ingestion `cursor` advances while `pos` stays
        // at the last batched boundary, so cursor is the live KV length. Once
        // generation begins, pos is the authoritative fed-token count.
        const uint32_t liveTok = std::max(
            sl.pos, std::min(sl.cursor, (uint32_t)sl.prompt.size()));
        moveSlotState(source, target, liveTok);
        sl.phys = target;
        owner[target] = owner[source];
        owner[source] = -1;
    }
    slotsNeedCompact = false;
}

uint32_t qk_engine::acquirePhysicalSlot(uint32_t slot) {
    if (splitStage() || !slotCompact) {
        slots[slot].phys = slot;
        return slot;
    }
    compactActiveSlots();
    uint32_t n = 0;
    for (const Slot& sl : slots) if (sl.active && !sl.finPending) ++n;
    if (n >= nSlots) abort();
    slots[slot].phys = n;
    return n;
}

void qk_engine::copyDnStripes(uint32_t fromStripe, uint32_t toStripe) {
    // Every caller is synchronous: the preceding command buffer has completed
    // and the next one has not been committed, so shared-memory copies are
    // coherent without another Metal submission.
    if ((fromStripe == nSlots || toStripe == nSlots) && !specScratch) abort();
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
    const uint32_t stripe = physicalSlot(slot);
    copyDnStripes(stripe, nSlots);  // seed scratch from live state at `base`
    std::vector<float> dummy;
    prefillBatchLast(toks, n, stripe, dummy, /*wantLogits=*/false, base, outIds,
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
    if (!copyPackedStripes(physicalSlot(slot), e, /*save=*/true,
                           pcacheCompact ? sl.pos : 0)) {
        fprintf(stderr, "prefix cache snapshot allocation/copy failed (%u tokens)\n", sl.pos);
        return;
    }
    size_t pp = std::min((size_t)sl.pos, sl.prompt.size());
    e.tokens.assign(sl.prompt.begin(), sl.prompt.begin() + pp);
    e.tokens.insert(e.tokens.end(), sl.genTokens.begin(), sl.genTokens.begin() + fedGen);
    e.lru = ++lruClock;
    e.valid = true;
    if (getenv("QK_MEM_STATS"))
        fprintf(stderr, "[pcache] save entry=%d tokens=%u payload=%.1f MiB (%s)\n",
                idx, (uint32_t)e.tokens.size(), e.snapBytes / 1048576.0,
                pcacheCompact ? "packed" : "full");
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
    CacheEntry& e = pcache[cacheIdx];
    if (pcacheCompact && e.tokens.size() != e.snapTok) {
        fprintf(stderr, "prefix cache token/state length mismatch for entry %d\n", cacheIdx);
        abort();
    }
    if (!copyPackedStripes(physicalSlot(slot), e, /*save=*/false,
                           (uint32_t)e.tokens.size())) {
        fprintf(stderr, "prefix cache restore invariant failed for entry %d\n", cacheIdx);
        abort();
    }
}

bool qk_engine::open(const char* path, const qk_config& cfg, char* err, size_t errLen) {
    auto fail = [&](const char* m) { if (err && errLen) snprintf(err, errLen, "%s", m); return false; };
    nSlots = cfg.n_slots; nCtx = cfg.n_ctx; chunkN = cfg.chunk;
    shareFork = getenv("QK_FORK") != nullptr;
    prefillMulti = !getenv("QK_PREFILL_MULTI") || atoi(getenv("QK_PREFILL_MULTI")) != 0;
    slotCompact = !getenv("QK_SLOT_COMPACT") || atoi(getenv("QK_SLOT_COMPACT")) != 0;
    pcacheCompact = !getenv("QK_PCACHE_COMPACT") ||
                    atoi(getenv("QK_PCACHE_COMPACT")) != 0;
    kvQ8 = getenv("QK_KV_Q8") && atoi(getenv("QK_KV_Q8")) != 0;
    kvF16 = !kvQ8 && getenv("QK_KV_F16") && atoi(getenv("QK_KV_F16")) != 0;
    prefillRepack = getenv("QK_PREFILL_REPACK") &&
                    atoi(getenv("QK_PREFILL_REPACK")) != 0;
    if (const char* v = getenv("QK_PREFILL_REPACK_USE"))
        prefillRepackUse = atoi(v) != 0;
    kvBytes = kvQ8 ? 1u : kvF16 ? 2u : 4u;
    kvRowBytes = kvQ8 ? 272u : dh * kvBytes;
    specOn = getenv("QK_SPEC") != nullptr;
    specScratch = specOn || getenv("QK_SPEC_SCRATCH") != nullptr;
    if (nSlots < 1 || nSlots > 16 || nCtx < 64 || nCtx > 65536 || chunkN < 1 || chunkN > 32)
        return fail("qk_open: bad config");
    // Prefer the shader snapshot installed beside the running executable.
    // Shared-library consumers without an adjacent snapshot still fall back
    // to the working-tree path in loadMetalSource.
    NSString* exePath = [NSBundle mainBundle].executablePath;
    initMtl(c, exePath ? exePath.fileSystemRepresentation : "libqk");
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
    if (prefillRepack && guIq4) {
        fprintf(stderr, "qk_open: QK_PREFILL_REPACK is Q8-only; disabled for IQ4 model\n");
        prefillRepack = false;
    }
    const size_t rbQ8e = ggmlRowBytes(GGML_Q8_0, nEmbd);
    const size_t rbQ8i = ggmlRowBytes(GGML_Q8_0, dIn);
    const size_t rbE = ggmlRowBytes(GGML_Q6_K, nEmbd);
    const uint32_t nB = nSlots, tmax = nCtx;
    nsg = getenv("QK_MOE_NSG") ? (uint32_t)atoi(getenv("QK_MOE_NSG")) : 4;

    auto readGemvTpr = [](const char* name, uint32_t dflt) {
        const char* v = getenv(name);
        const uint32_t x = v ? (uint32_t)atoi(v) : dflt;
        return x >= 8u && x <= 256u && (x & (x - 1u)) == 0u ? x : dflt;
    };
    if (const char* v = getenv("QK_GEMV_FAST")) gemvFast = atoi(v) != 0;
    if (!gemvFast) gemvTprA = 64;
    gemvTprA = readGemvTpr("QK_GEMV_TPR_A", gemvTprA);
    gemvTprO = readGemvTpr("QK_GEMV_TPR_O", gemvTprO);
    pRms   = getPipe(c, "rmsnorm", "rmsnorm", 0);
    pGemvA = gemvFast
        ? getPipe(c, "gemv_q8_0_fixed", "gemv_q8_0_k2048", gemvTprA)
        : getPipe(c, "gemv_q8_0", "gemv_q8_0", gemvTprA);
    pGemvO = gemvFast
        ? getPipe(c, "gemv_q8_0_fixed", "gemv_q8_0_k4096", gemvTprO)
        : getPipe(c, "gemv_q8_0", "gemv_q8_0", gemvTprO);
    prefillGemvBatch = gemvFast && gemvTprA == 16u && gemvTprO == 128u;
    if (const char* v = getenv("QK_PREFILL_GEMV_BATCH"))
        prefillGemvBatch = prefillGemvBatch && atoi(v) != 0;
    if (prefillGemvBatch) {
        // Exact small-prefill path: reuse each decoded Q8 block across four
        // token rows while retaining the shipping GEMV's TPR/reduction order.
        // The 80B is predominantly IQ4 but retains a Q8 attention-V projection,
        // so its mixed-format path needs these pipelines too.
        pPrefillGemvA4 = getPipe(c, "gemv_q8_0_batch", "gemv_q8_0_b4_k2048", gemvTprA);
        pPrefillGemvO4 = getPipe(c, "gemv_q8_0_batch", "gemv_q8_0_b4_k4096", gemvTprO);
    }
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
    const bool dnDecodeOpt = !getenv("QK_DN_DECODE") ||
                             atoi(getenv("QK_DN_DECODE")) != 0;
    const bool dnDecodeResident = dnDecodeOpt &&
        (!getenv("QK_DN_DECODE_RES") || atoi(getenv("QK_DN_DECODE_RES")) != 0);
    const char* dnStepFn = dnDecodeResident ? "dn_step_res8" : "dn_step";
    pStep = getPipe(c, "dn_step", dnStepFn, 0);
    dnDecodeSplitMode = dnDecodeOpt ? -1 : 0;
    if (const char* v = getenv("QK_DN_DECODE_SPLIT")) dnDecodeSplitMode = atoi(v) != 0;
    const bool dnDecodeSimd = getenv("QK_DN_DECODE_SIMD") &&
                              atoi(getenv("QK_DN_DECODE_SIMD")) != 0;
    if (dnDecodeSimd) dnDecodeSplitMode = 1;
    if (dnDecodeSplitMode != 0) {
        pStepPrep = getPipe(c, "dn_step", "dn_step_prep", 0);
        const char* stateFn = dnDecodeResident ? "dn_step_state_res8" : "dn_step_state";
        if (dnDecodeSimd) stateFn = "dn_step_state_rowsimd";
        pStepState = getPipe(c, "dn_step", stateFn, 0);
        if (dnDecodeSplitMode < 0)
            pStepStateSimd = getPipe(c, "dn_step", "dn_step_state_rowsimd", 0);
    }
    pPrep  = getKvPipe(c, "fa_srv", "fa_prep_srv", kvBytes);
    pAttn  = getKvPipe(c, "fa_srv", "fa_attn_srv", kvBytes);
    pAttnSplit  = getKvPipe(c, "fa_srv",
        kvQ8 ? "fa_attn_srv_split_q8" : "fa_attn_srv_split", kvBytes);
    pAttnReduce = getPipe(c, "fa_srv", "fa_attn_srv_reduce", 0);
    pAttnCompact = getKvPipe(c, "fa_srv", "fa_attn_srv_split_compact", kvBytes);
    pAttnMulti = getKvPipe(c, "fa_srv", "fa_attn_srv_split_multi", kvBytes);
    if (kvQ8) {
        pAttnQ8G2 = getKvPipe(c, "fa_srv", "fa_attn_srv_split_q8_gqa2", kvBytes);
        pAttnQ8G4 = getKvPipe(c, "fa_srv", "fa_attn_srv_split_q8_gqa4", kvBytes);
        pAttnQ8G8 = getKvPipe(c, "fa_srv", "fa_attn_srv_split_q8_gqa8", kvBytes);
    } else if (kvF16) {
        pAttnF16G2 = getKvPipe(c, "fa_srv", "fa_attn_srv_split_f16_gqa2", kvBytes);
        pAttnF16G4 = getKvPipe(c, "fa_srv", "fa_attn_srv_split_f16_gqa4", kvBytes);
        pAttnF16G8 = getKvPipe(c, "fa_srv", "fa_attn_srv_split_f16_gqa8", kvBytes);
    } else {
        pAttnF32G2 = getKvPipe(c, "fa_srv", "fa_attn_srv_split_f32_gqa2", kvBytes);
        pAttnF32G4 = getKvPipe(c, "fa_srv", "fa_attn_srv_split_f32_gqa4", kvBytes);
        pAttnF32G8 = getKvPipe(c, "fa_srv", "fa_attn_srv_split_f32_gqa8", kvBytes);
    }
    if (const char* v = getenv("QK_FA_SPLIT")) faDecodeSplit = atoi(v) != 0;
    if (const char* v = getenv("QK_FA_WORK")) faDecodeWork = atoi(v) != 0;
    if (const char* v = getenv("QK_FA_F16_GQA")) faF16Gqa = atoi(v) != 0;
    if (const char* v = getenv("QK_FA_F32_GQA")) faF32Gqa = atoi(v) != 0;
    if (const char* v = getenv("QK_FA_FAN")) {
        const int x = atoi(v);
        if (x == 1 || x == 4 || x == 8 || x == 16 || x == 32 || x == 64)
            faFanOverride = x;
        else if (x != 0)
            fprintf(stderr, "QK_FA_FAN=%d ignored (expected 1,4,8,16,32,64)\n", x);
    }
    pMoeS  = getPipe(c, "moe_select", "moe_select", 0);
    pMoeLA = getPipe(c, "moe_logits", "moe_logits_addn", nsg);
    if (const char* v = getenv("QK_MOE_NORM_SPLIT"))
        moeNormSplitMode = atoi(v) != 0;
    pMoeLS = nil;
    pAddNSg = nil;
    if (nSlots >= 8u || moeNormSplitMode == 1) {
        pMoeLS = getPipe(c, "moe_logits", "moe_logits_scaled", nsg);
        pAddNSg = getPipe(c, "add_rmsnorm", "add_rmsnorm_sg", 0);
    }
    bool moeGuFixed = !guIq4 && ffE == 512u;
    if (const char* v = getenv("QK_MOE_GU_FIXED"))
        moeGuFixed = atoi(v) != 0 && !guIq4 && ffE == 512u;
    pMoeGu = getPipe(c, "moe_gateup_all",
                     moeGuFixed ? "moe_gateup_all_k2048_ff512" : "moe_gateup_all", nsg);
    pMoeD4 = getPipe(c, "moe_down_all", "moe_down_all_iq4", nsg);
    pMoeD6 = getPipe(c, "moe_down_all", "moe_down_all_q6k", nsg);
    pAddN  = getPipe(c, "add_rmsnorm", "add_rmsnorm", 0);
    if (const char* v = getenv("QK_HEAD_GEMM")) headGemm = atoi(v) != 0;
    if (const char* v = getenv("QK_HEAD_GEMM_N")) {
        const uint32_t x = (uint32_t)atoi(v);
        if (x >= 1u && x <= 64u) headGemmN = x;
    }
    if (const char* v = getenv("QK_HEAD_F16")) headF16 = atoi(v) != 0;
    if (const char* v = getenv("QK_HEAD_FIXED_K")) headFixedK = atoi(v) != 0;
    pHead = headFixedK
        ? getPipe(c, "gemv_q6_k_fixed", "gemv_q6_k_k2048", 2)
        : getPipe(c, "gemv_q6_k", "gemv_q6_k", 2);
    pHeadTop = getPipe(c, "head_q6", "head_q6_gemm_b8_top1_f32", 0);
    pHeadTop64 = getPipe(c, "head_q6", "head_q6_gemm_b8_top1_f32_m64", 0);
    if (headF16) {
        pHeadF8 = getPipe(c, "head_q6", "head_q6_gemm_b8_top1_f16", 0);
        pHeadF16 = getPipe(c, "head_q6", "head_q6_gemm_b16_top1_f16", 0);
        pHeadF32 = getPipe(c, "head_q6", "head_q6_gemm_b32_top1_f16", 0);
    }
    pHeadTopReduce = getPipe(c, "head_q6", "head_top1_reduce_batch", 0);
    pAm1   = getPipe(c, "argmax", "argmax1", 0);
    pAm2   = getPipe(c, "argmax", "argmax2", 0);
    pEmb   = getPipe(c, embQ8 ? "embed_q8_0" : "embed_q6k", embQ8 ? "embed_q8_0" : "embed_q6k", 0);
    pPrepB = getKvPipe(c, "fa_batch", "fa_prep_batch", kvBytes);
    pAttnB = getKvPipe(c, "fa_batch", "fa_attn_batch", kvBytes);
    // Two-Q-head Q8/K64/S16 geometry is the prefill-attention default. `exact`
    // retains the geometry but restores scalar softmax summation; `q16`
    // restores the original Q16/K64/S8 kernel for full rollback.
    const char* faGeom = getenv("QK_FA_GEOM");
    if (kvQ8) {
        faQTM = 8; faThreads = 512; faHeadGroup = 2;
        pAttnBM = getKvPipe(c, "fa_gqa", "fa_attn_batch_gqa2_q8", kvBytes);
    } else if (kvF16) {
        // MSL cannot directly load half storage into a float matrix. The f16
        // twin expands one K/V fragment per simdgroup into float TG scratch,
        // retaining the default GQA reuse and f32 MMA accumulation.
        faQTM = 8; faThreads = 512; faHeadGroup = 2;
        pAttnBM = getKvPipe(c, "fa_gqa", "fa_attn_batch_gqa2_f16", kvBytes);
    } else if (faGeom && !strcmp(faGeom, "q16")) {
        faQTM = 16; faThreads = 256; faHeadGroup = 1;
        pAttnBM = getKvPipe(c, "fa_batch", "fa_attn_batch_mma", kvBytes);
    } else if (faGeom && !strcmp(faGeom, "exact")) {
        faQTM = 8; faThreads = 512; faHeadGroup = 2;
        pAttnBM = getKvPipe(c, "fa_gqa", "fa_attn_batch_gqa2_exact", kvBytes);
    } else {
        faQTM = 8; faThreads = 512; faHeadGroup = 2;
        pAttnBM = getKvPipe(c, "fa_gqa", "fa_attn_batch_gqa2", kvBytes);
    }
    pAbB   = getPipe(c, "dn_batch", "dn_ab_batch", nsg);
    pConvB = getPipe(c, "dn_batch", "dn_conv_batch", 0);
    pStepB = getPipe(c, "dn_batch", "dn_step_batch", 0);
    pGateB = getPipe(c, "dn_batch", "dn_gate_batch", nsg);
    if (const char* v = getenv("QK_DN_CHUNK")) dnChunk = atoi(v) != 0;
    if (const char* v = getenv("QK_DN_STEP_RES")) {
        const uint32_t x = (uint32_t)atoi(v);
        if (x == 0u || x == 8u) dnStepPW = x;
    }
    if (dnChunk) {
        pDnKq    = getPipe(c, "dn_chunk", "dn_chunk_kq", 0);
        pDnSolve = getPipe(c, "dn_chunk", "dn_chunk_solve", 0);
        const char* stepFn = dnStepPW ? "dn_chunk_step_res8_s8" : "dn_chunk_step";
        pDnStepC = getPipe(c, "dn_chunk", stepFn, dS);   // fc(0)=dS
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
        if (const char* v = getenv("QK_GEMM_ALIGNED")) gemmAligned = atoi(v) != 0;
        if (gemmAligned && !strcmp(fn, "gemm_q8_0_hp"))
            pGemmBAligned = getPipe(c, "gemm_q8_0", "gemm_q8_0_hp_aligned", 0);
        if (prefillRepack && !strcmp(fn, "gemm_q8_0_hp"))
            pGemmBK = getPipe(c, "gemm_q8_0", "gemm_q8_0_hp_kmajor", 0);
        gemmThreads = !strcmp(fn, "gemm_q8_0_sg") ? 128 : 256;  // sg is 4-simd; scalar+h are 256
        if (!strcmp(fn, "gemm_q8_0_h2")) { gemmBM = 64; gemmBN = 128; }
        if (!strcmp(fn, "gemm_q8_0_hp")) { gemmBM = 64; gemmBN = 32; gemmThreads = 128; }
    }
    pMoeGrp  = getPipe(c, "moe_grouped", "moe_group", 0);
    pMoeGrpLive = getPipe(c, "moe_grouped", "moe_group_live", 0);
    pMoeGrpWork = getPipe(c, "moe_grouped", "moe_group_work", 0);
    pMoeGuG  = getPipe(c, "moe_grouped", "moe_gu_grouped", nsg);
    pMoeGuG2 = getPipe(c, "moe_grouped", "moe_gu_grouped2", 0);
    pMoeGuG3 = getPipe(c, "moe_grouped", "moe_gu_grouped3", 0);
    pMoeGuG4 = getPipe(c, "moe_grouped", "moe_gu_grouped4", 0);
    pMoeGuG4Live = getPipe(c, "moe_grouped", "moe_gu_grouped4_live", 0);
    pMoeGuG5 = getPipe(c, "moe_grouped", "moe_gu_grouped5", 0);
    pMoeGuG5Live = getPipe(c, "moe_grouped", "moe_gu_grouped5_live", 0);
    pMoeGuG5Work = getPipe(c, "moe_grouped", "moe_gu_grouped5_work", 0);
    if (guIq4) {
        if (const char* v = getenv("QK_IQ4_FIXED_K")) iq4FixedK = atoi(v) != 0;
        pGemv4 = iq4FixedK
            ? getPipe(c, "gemv_iq4_xs_fixed", "gemv_iq4_xs_k2048", 2)
            : getPipe(c, "gemv_iq4_xs", "gemv_iq4_xs", 2);
        pGemv4O = iq4FixedK
            ? getPipe(c, "gemv_iq4_xs_fixed", "gemv_iq4_xs_k4096", 2)
            : pGemv4;
        pGemmB4   = getPipe(c, "gemm_iq4_xs", "gemm_iq4_xs_hp", 0);
        if (gemmAligned)
            pGemmB4Aligned = getPipe(c, "gemm_iq4_xs", "gemm_iq4_xs_hp_aligned", 0);
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
    pMoeDGP4Work = getPipe(c, "moe_down_grouped", "moe_down_grouped_p_work_iq4", 0);
    pMoeDGP6Work = getPipe(c, "moe_down_grouped", "moe_down_grouped_p_work_q6k", 0);
    pMoeDR   = getPipe(c, "moe_down_grouped", "moe_down_reduce", 0);
    pLogG    = getPipe(c, "moe_logits", "moe_logits_gemm", 0);
    if (const char* mg = getenv("QK_MOE_GROUPED")) { moeGrouped = atoi(mg); moeGroupN = 0; }
    if (const char* mn = getenv("QK_MOE_GROUP_N")) moeGroupN = (uint32_t)atoi(mn);
    if (const char* mw = getenv("QK_MOE_WORK")) moeWork = atoi(mw) != 0;
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

    // Decode and batched-prefill activations have the same strictly ordered
    // phase graph and are never in flight together (every engine API call is
    // synchronous).  The default arena below colors those lifetimes onto a
    // small set of whole MTLBuffers, avoiding offset plumbing in every shader.
    // Keep an independent-allocation mode as a correctness/perf rollback.
    const bool scratchAlias = !getenv("QK_SCRATCH_ALIAS") ||
                              atoi(getenv("QK_SCRATCH_ALIAS")) != 0;
    if (!scratchAlias) {
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
        bAV = createBuf(c, (size_t)nB * 64 * 4, nullptr, true);
        bAI = createBuf(c, (size_t)nB * 64 * 4, nullptr, true);
        bTok = createBuf(c, (size_t)nB * 4, nullptr, true);
    }
    bMoeScale = pMoeLS
        ? createBuf(c, (size_t)nB * sizeof(float), nullptr, true) : nil;
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
    bEmbdW = WOFF(tEmbd);
    bRope = createBuf(c, (size_t)tmax * (nRot / 2) * 2 * 4, nullptr, true);
    bRbScratch = createBuf(c, 64, nullptr, true);
    bSlotIn = createBuf(c, (size_t)nB * 4, nullptr, true);
    bSlotPos = createBuf(c, (size_t)nB * 4, nullptr, true);
    // The arena allocation below also aliases decode-attention partials with
    // the full decode-logit row: attention is finished before the head runs.
    const uint32_t maxChunks = (nCtx + 255u) / 256u;
    const size_t partialBytes = !splitStage() && faDecodeSplit
        ? (size_t)nB * hQ * maxChunks * (dh + 2) * 4 : 0;
    const size_t decodeLogitBytes = splitStage() ? 0 : (size_t)nB * vocab * 4;
    bFaWork = partialBytes
        ? createBuf(c, (size_t)nB * maxChunks * 2u * sizeof(uint32_t), nullptr, true) : nil;
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

    // Unsplit 35B admission amortizes dense/MoE weights through 1024-row
    // chunks. Split and IQ4/80B engines retain the lean 128-row allocation;
    // QK_MAXB remains an explicit (context-clamped) override.
    uint32_t cap = !splitStage() && !guIq4 ? 1024u : 128u;
    if (const char* v = getenv("QK_MAXB")) {
        long x = atol(v);
        if (x >= 16 && x <= 1024) cap = (uint32_t)x;
    }
    cap = (uint32_t)std::min<uint64_t>(cap, (uint64_t)nCtx * nSlots);
    maxB = cap;
    const uint32_t headLogitRows = headGemm
        ? std::min(cap, std::max(1u, headGemmN - 1u)) : cap;
    const size_t tiles32 = (vocab + 31u) / 32u;
    const size_t tiles64 = (vocab + 63u) / 64u;
    const size_t headPairs = lastStage() && headGemm
        ? std::max((size_t)std::min(cap, 7u) * tiles32, (size_t)cap * tiles64) : 0;
    const size_t sXin = (size_t)cap * nEmbd * 4;
    const size_t sXn = sXin;
    const size_t sBig = (size_t)cap * chQkv * 4;
    const size_t sMid = (size_t)cap * dIn * 4;
    const size_t sKin = (size_t)cap * hKV * dh * 4;
    const size_t sVin = sKin;
    const size_t sGb = (size_t)cap * 2 * hV * 4;
    const size_t sConvOut = sBig;
    const size_t sO = (size_t)((cap + 63) / 64 * 64) * dIn * 4;
    const size_t sAtt = sMid;
    const size_t sAttnOut = sXin;
    const size_t sY = sXin;
    const size_t sXn2 = sXin;
    const size_t sML = (size_t)cap * (nExp + 1) * 4;
    const size_t sMH = (size_t)cap * (nUsed + 1) * ffE * 4;
    const size_t sMSel = (size_t)cap * 160;
    const size_t sMY = sXin;
    const size_t sLogits = lastStage() ? (size_t)headLogitRows * vocab * 4 : 0;
    const size_t sAV = lastStage() ? (size_t)cap * 64 * 4 : 0;
    const size_t sAI = sAV;
    const size_t sTok = lastStage() ? (size_t)cap * 4 : 0;
    const size_t sIds = (size_t)cap * 4;
    uint32_t nRecOwned = 0;
    for (uint32_t il = lFirst; il < lEnd; ++il) {
        char rn[64];
        snprintf(rn, sizeof rn, "blk.%u.ssm_a", il);
        nRecOwned += g.find(rn) != nullptr;
    }
    const size_t carryStride = (size_t)nSlots * chQkv * 3 * 4;
    const size_t sCarry = (size_t)nRecOwned * carryStride;
    const size_t nChMax = (cap + 63) / 64;
    const size_t sDnKQ = dnChunk ? nChMax * hK * 2 * 64 * 64 * 4 : 0;
    const size_t sDnUW = dnChunk ? nChMax * hV * 4 * 64 * (size_t)dS * 4 : 0;
    const size_t sDnAtt = dnChunk ? nChMax * hV * 64 * 64 * 4 : 0;
    const size_t sDnEl = dnChunk ? nChMax * hV * 4 : 0;
    const size_t sMDy = (size_t)cap * (nUsed + 1) * nEmbd * 4;

    if (scratchAlias) {
        // Whole-buffer lifetime coloring.  Existing buffer barriers separate
        // every transition listed here; no kernel ever observes two logical
        // members of one group at the same time.  Keeping offset zero also
        // preserves all shader indexing and Metal alignment assumptions.
        size_t logical = 0, physical = 0;
        auto group = [&](std::initializer_list<size_t> sizes) -> id<MTLBuffer> {
            size_t mx = 0;
            for (size_t z : sizes) { logical += z; mx = std::max(mx, z); }
            mx = std::max(mx, sizeof(uint32_t));
            physical += mx;
            return createBuf(c, mx, nullptr, true);
        };
        const size_t dXin = (size_t)nB * nEmbd * 4;
        const size_t dXn = dXin;
        const size_t dBig = (size_t)nB * chQkv * 4;
        const size_t dMid = (size_t)nB * dIn * 4;
        const size_t dKin = (size_t)nB * hKV * dh * 4;
        const size_t dVin = dKin;
        const size_t dGb = (size_t)nB * 2 * hV * 4;
        const size_t dAtt = dMid;
        const size_t dAttnOut = dXin;
        const size_t dY = dXin;
        const size_t dXn2 = dXin;
        const size_t dML = (size_t)nB * (nExp + 1) * 4;
        const size_t dMH = (size_t)nB * (nUsed + 1) * ffE * 4;
        const size_t dMSel = (size_t)nB * 160;
        const size_t dMY = dXin;
        const size_t dAV = (size_t)nB * 64 * 4;
        const size_t dAI = dAV;
        const size_t dTok = (size_t)nB * 4;

        bbXin = group({sXin, dXin});
        bbXn = bbML = bbDnUW = bbMDy =
            group({sXn, sML, sDnUW, sMDy, dXn, dML});
        bbBig = bbDnKQ = bbMY = bbHeadI =
            group({sBig, sDnKQ, sMY, headPairs * sizeof(uint32_t), dBig, dMY});
        bbMid = bbY = bbLogits =
            group({sMid, sY, sLogits, dMid, dY});
        bbKin = bbAV = group({sKin, sAV, dKin, dAV});
        bbVin = bbAI = group({sVin, sAI, dVin, dAI});
        bbGb = bbMSel = bbTok = group({sGb, sMSel, sTok, dGb, dMSel, dTok});
        bbConvOut = bbAttnOut = bbMH = bbIds =
            group({sConvOut, sAttnOut, sMH, sIds, dAttnOut, dMH,
                   dnChunk ? sO : 0});
        bbO = dnChunk ? bbConvOut : group({sO});
        bbAtt = bbDnAtt = bbXn2 =
            group({sAtt, sDnAtt, sXn2, dAtt, dXn2});
        bbCarry = bbHeadV = bPartial = bLogits =
            group({sCarry, headPairs * sizeof(float), partialBytes, decodeLogitBytes});

        // Decode uses the same phase-colored storage as prefill.  Synchronous
        // API waits guarantee that the two command-buffer families never
        // overlap, while encodeStep's grouped MoE intentionally uses bbMDy.
        bXin = bbXin; bXn = bbXn; bBig = bbBig; bMid = bbMid;
        bKin = bbKin; bVin = bbVin; bGb = bbGb; bAtt = bbAtt;
        bAttnOut = bbAttnOut; bY = bbY; bXn2 = bbXn2; bML = bbML;
        bMH = bbMH; bMSel = bbMSel; bMY = bbMY;
        bAV = bbAV; bAI = bbAI; bTok = bbTok;
        if (!partialBytes) bPartial = nil;
        if (!decodeLogitBytes) bLogits = nil;
        if (!headPairs) { bbHeadV = nil; bbHeadI = nil; }
        if (!sLogits) bbLogits = nil;
        if (!sAV) { bbAV = nil; bbAI = nil; bbTok = nil; }
        if (!sDnKQ) { bbDnKQ = nil; bbDnUW = nil; bbDnAtt = nil; }
        if (getenv("QK_MEM_STATS"))
            fprintf(stderr, "qk_open: scratch aliases %.1f -> %.1f MiB (saved %.1f MiB)\n",
                    logical / 1048576.0, physical / 1048576.0,
                    (logical - physical) / 1048576.0);
    } else {
        bbXin = createBuf(c, sXin, nullptr, true);
        bbXn = createBuf(c, sXn, nullptr, true);
        bbBig = createBuf(c, sBig, nullptr, true);
        bbMid = createBuf(c, sMid, nullptr, true);
        bbKin = createBuf(c, sKin, nullptr, true);
        bbVin = createBuf(c, sVin, nullptr, true);
        bbGb = createBuf(c, sGb, nullptr, true);
        bbConvOut = createBuf(c, sConvOut, nullptr, true);
        bbO = createBuf(c, sO, nullptr, true);
        bbAtt = createBuf(c, sAtt, nullptr, true);
        bbAttnOut = createBuf(c, sAttnOut, nullptr, true);
        bbY = createBuf(c, sY, nullptr, true);
        bbXn2 = createBuf(c, sXn2, nullptr, true);
        bbML = createBuf(c, sML, nullptr, true);
        bbMH = createBuf(c, sMH, nullptr, true);
        bbMSel = createBuf(c, sMSel, nullptr, true);
        bbMY = createBuf(c, sMY, nullptr, true);
        bbLogits = sLogits ? createBuf(c, sLogits, nullptr, true) : nil;
        bbAV = sAV ? createBuf(c, sAV, nullptr, true) : nil;
        bbAI = sAI ? createBuf(c, sAI, nullptr, true) : nil;
        bbTok = sTok ? createBuf(c, sTok, nullptr, true) : nil;
        bbHeadV = headPairs ? createBuf(c, headPairs * sizeof(float), nullptr, true) : nil;
        bbHeadI = headPairs ? createBuf(c, headPairs * sizeof(uint32_t), nullptr, true) : nil;
        bbIds = createBuf(c, sIds, nullptr, true);
        bbMDy = createBuf(c, sMDy, nullptr, true);
        bbCarry = sCarry ? createBuf(c, sCarry, nullptr, true) : nil;
        bPartial = partialBytes ? createBuf(c, partialBytes, nullptr, true) : nil;
        bLogits = decodeLogitBytes ? createBuf(c, decodeLogitBytes, nullptr, true) : nil;
        if (dnChunk) {
            bbDnKQ = createBuf(c, sDnKQ, nullptr, true);
            bbDnUW = createBuf(c, sDnUW, nullptr, true);
            bbDnAtt = createBuf(c, sDnAtt, nullptr, true);
        }
        if (getenv("QK_MEM_STATS"))
            fprintf(stderr, "qk_open: scratch aliases disabled\n");
    }
    bbStart = createBuf(c, (size_t)(nExp + 2) * 4, nullptr, true);
    bbATok = createBuf(c, (size_t)cap * (nUsed + 1) * 4, nullptr, true);
    bbASlot = createBuf(c, (size_t)cap * (nUsed + 1) * 4, nullptr, true);
    bbLive = createBuf(c, ((size_t)cap * nUsed + 1) * 4, nullptr, true);
    const size_t workCap = nExp + ((size_t)cap * nUsed + 31) / 32 + (cap + 31) / 32;
    bbWork = createBuf(c, workCap * 2 * 4, nullptr, true);
    if (sCarry) memset(bbCarry.contents, 0, sCarry);
    if (dnChunk) {   // chunked-DN scratch: per (chunk, head) tiles, reused per layer
        bbDnEl = createBuf(c, sDnEl, nullptr, true);
    }

    layers.resize(nLayer);
    char nb[128];
    size_t carryCursor = 0;
    size_t specStripeBytes = 0;
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
            L.carryOff = carryCursor;
            carryCursor += carryStride;
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
            specStripeBytes += L.ps1 + L.ps2;
        } else {
            L.wq = Wd("attn_q.weight", L.iq4P1);
            L.wk = Wd("attn_k.weight", L.iq4P2);
            L.wv = Wd("attn_v.weight", L.iq4P3);
            L.qn = W(T("attn_q_norm.weight"), dh * 4);
            L.kn = W(T("attn_k_norm.weight"), dh * 4);
            L.wo = Wd("attn_output.weight", L.iq4Wo);
            L.ps1 = (size_t)hKV * tmax * kvRowBytes;
            L.ps2 = (size_t)hKV * tmax * kvRowBytes;
        }
        // +slack: fa_attn_batch_mma reads full 8-key MMA tiles, so at near-full
        // context (nk ~ tmax) the last slot's K/V load over-reads a few rows
        // past the buffer. One KBM(64)-key block of zeroed slack keeps it
        // in-bounds and finite (over-read rows are P=0, so contribute nothing).
        const size_t stSlack = L.rec ? 0 : (size_t)64 * kvRowBytes;
        // Spec-enabled recurrent layers reserve one shared verification
        // scratch stripe at index nSlots. Normal engines and attention (whose
        // positional KV is rollback-free) need only the live slot stripes.
        const size_t nStateStripes = nB + (L.rec && specScratch ? 1u : 0u);
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
    if (prefillRepack && pGemmBK) {
        auto align256 = [](size_t x) { return (x + 255u) & ~(size_t)255u; };
        size_t packedBytes = 0;
        auto reserveQ8 = [&](bool iq4, uint32_t M, uint32_t K) {
            if (!iq4) packedBytes = align256(packedBytes) +
                                    (size_t)M * (K / 32u) * sizeof(block_q8_0);
        };
        for (uint32_t il = lFirst; il < lEnd; ++il) {
            const Layer& L = layers[il];
            if (L.rec) {
                reserveQ8(L.iq4P1, chQkv, nEmbd);
                reserveQ8(L.iq4P2, dIn, nEmbd);
                reserveQ8(L.iq4Wo, nEmbd, dIn);
            } else {
                reserveQ8(L.iq4P1, chQkv, nEmbd);
                reserveQ8(L.iq4P2, hKV * dh, nEmbd);
                reserveQ8(L.iq4P3, hKV * dh, nEmbd);
                reserveQ8(L.iq4Wo, nEmbd, dIn);
            }
        }
        if (packedBytes > c.dev.maxBufferLength) {
            fprintf(stderr,
                    "qk_open: QK_PREFILL_REPACK needs %.2f GiB, above maxBufferLength; disabled\n",
                    packedBytes / 1073741824.0);
            prefillRepack = false;
        } else {
            const auto t0 = std::chrono::steady_clock::now();
            id<MTLBuffer> packed = createBuf(c, packedBytes, nullptr, true);
            size_t cursor = 0;
            auto packQ8 = [&](WB src, WB& dst, bool iq4, uint32_t M, uint32_t K) {
                if (iq4) return;
                cursor = align256(cursor);
                dst = WB{packed, (NSUInteger)cursor};
                const uint32_t kb = K / 32u;
                const block_q8_0* s = (const block_q8_0*)
                    ((const uint8_t*)src.b.contents + src.o);
                block_q8_0* d = (block_q8_0*)((uint8_t*)packed.contents + cursor);
                for (uint32_t b = 0; b < kb; ++b)
                    for (uint32_t m = 0; m < M; ++m)
                        d[(size_t)b * M + m] = s[(size_t)m * kb + b];
                cursor += (size_t)M * kb * sizeof(block_q8_0);
            };
            for (uint32_t il = lFirst; il < lEnd; ++il) {
                Layer& L = layers[il];
                if (L.rec) {
                    packQ8(L.qkvW, L.qkvP, L.iq4P1, chQkv, nEmbd);
                    packQ8(L.zW, L.zP, L.iq4P2, dIn, nEmbd);
                    packQ8(L.outW, L.outP, L.iq4Wo, nEmbd, dIn);
                } else {
                    packQ8(L.wq, L.wqP, L.iq4P1, chQkv, nEmbd);
                    packQ8(L.wk, L.wkP, L.iq4P2, hKV * dh, nEmbd);
                    packQ8(L.wv, L.wvP, L.iq4P3, hKV * dh, nEmbd);
                    packQ8(L.wo, L.woP, L.iq4Wo, nEmbd, dIn);
                }
            }
            if (cursor != packedBytes) return fail("qk_open: prefill repack size mismatch");
            extraBufs.push_back(packed);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "qk_open: prefill K-major Q8 %.2f GiB packed in %.1f ms\n",
                    packedBytes / 1073741824.0, ms);
        }
    }
    if (carryCursor != sCarry) return fail("qk_open: recurrent carry layout mismatch");
    if (getenv("QK_MEM_STATS"))
        fprintf(stderr,
                "qk_open: recurrent carry %u/%u layers %.1f MiB; spec stripe %s %.1f MiB\n",
                nRecOwned, lEnd - lFirst, sCarry / 1048576.0,
                specScratch ? "allocated" : "lazy-saved", specStripeBytes / 1048576.0);

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
    if (getenv("QK_MEM_STATS"))
        fprintf(stderr, "qk_open: prefix snapshots %s/lazy (full capacity %.2f GiB x %u entries)\n",
                pcacheCompact ? "packed" : "full", snapSize / 1073741824.0, PCACHE_N);

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
    for (uint32_t s = 0; s < nSlots; ++s) slots[s].phys = s;
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
    // dspY adds the chunk/work-item (y) grid axis the plain dsp lacks.
    auto dspY = [&](id<MTLComputePipelineState> pso, std::initializer_list<WB> bufs,
                    const void* pc, uint32_t pcSize, uint32_t wgs, uint32_t ydim,
                    uint32_t thr, uint32_t gridZ = 0) {
        [enc setComputePipelineState:pso];
        uint32_t i = 0;
        for (const WB& b : bufs) [enc setBuffer:b.b offset:b.o atIndex:i++];
        [enc setBytes:pc length:pcSize atIndex:i];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, ydim, gridZ ? gridZ : zdim)
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
    // Adaptive split-K schedule. A 256-key producer maximizes parallelism for
    // one/few live trajectories; balanced deep batches already expose ample
    // head/slot parallelism and instead win by processing several chunks in each
    // producer. Highly skewed batches use a compact (slot,group) work list.
    uint32_t maxPos = 0, minChunks = UINT32_MAX, liveChunks = 0;
    uint32_t slotChunks[16]{};
    { const uint32_t* sp = (const uint32_t*)bSlotPos.contents;
      for (uint32_t s = 0; s < zdim; ++s) {
          maxPos = std::max(maxPos, sp[s]);
          slotChunks[s] = (sp[s] + 1u + 255u) / 256u;
          minChunks = std::min(minChunks, slotChunks[s]);
          liveChunks += slotChunks[s];
      } }
    const uint32_t splitChunks = (maxPos + 1u + 255u) / 256u;
    auto nearestFaFan = [](uint32_t chunkWork) {
        const uint32_t target = std::max(1u, (chunkWork + 8u) / 16u);
        static constexpr uint32_t fan[] = {1u, 4u, 8u, 16u, 32u, 64u};
        uint32_t best = 1u, bd = UINT32_MAX;
        for (uint32_t f : fan) {
            const uint32_t d = f > target ? f - target : target - f;
            if (d <= bd) { best = f; bd = d; }
        }
        return best;
    };
    auto faFanForGroups = [&]() {
        static constexpr uint32_t fan[] = {8u, 16u, 32u, 64u};
        uint32_t best = 8u, bd = UINT32_MAX;
        for (uint32_t f : fan) {
            uint32_t groups = 0;
            for (uint32_t s = 0; s < zdim; ++s)
                groups += std::max(1u, (slotChunks[s] + f / 2u) / f);
            const uint32_t d = groups > 16u ? groups - 16u : 16u - groups;
            if (d <= bd) { best = f; bd = d; }
        }
        return best;
    };
    uint32_t faFan = 1u;
    const bool balanced = minChunks * 2u >= splitChunks;
    if (splitChunks >= 16u && balanced) {
        faFan = nearestFaFan(liveChunks);
    } else if (splitChunks >= 16u) {
        uint32_t deep = 0;
        bool bimodal = true;
        for (uint32_t s = 0; s < zdim; ++s) {
            if (slotChunks[s] * 4u >= splitChunks * 3u) deep++;
            else if (slotChunks[s] * 4u > splitChunks) bimodal = false;
        }
        if (bimodal && (uint64_t)deep * splitChunks >= 96u)
            faFan = faFanForGroups();
    }
    if (zdim == 1u && splitChunks <= 32u) faFan = 1u;
    // Fan-4 only tied at the first balanced crossover in full-engine tests;
    // require the first materially winning fan unless explicitly overridden.
    if (faFan < 8u) faFan = 1u;
    if (faFanOverride > 0) faFan = (uint32_t)faFanOverride;
    if (kvQ8 || (kvF16 && faF16Gqa) || (!kvF16 && !kvQ8 && faF32Gqa))
        faFan = 1u;
    uint32_t decodeHeadGroup = 1u;
    if (kvQ8) {
        const uint32_t qPerKv = hQ / hKV;
        if (liveChunks >= 64u && qPerKv % 8u == 0u) decodeHeadGroup = 8u;
        else if ((liveChunks >= 16u || (zdim >= 4u && liveChunks >= 8u)) &&
                 qPerKv % 4u == 0u)
            decodeHeadGroup = 4u;
        else if (liveChunks >= 8u && qPerKv % 2u == 0u) decodeHeadGroup = 2u;
    } else if (kvF16 && faF16Gqa) {
        const uint32_t qPerKv = hQ / hKV;
        if (balanced && liveChunks >= 128u && qPerKv % 8u == 0u)
            decodeHeadGroup = 8u;
        else if (liveChunks >= 32u && qPerKv % 4u == 0u) decodeHeadGroup = 4u;
        else if (qPerKv % 2u == 0u) decodeHeadGroup = 2u;
    } else if (!kvF16 && !kvQ8 && faF32Gqa) {
        const uint32_t qPerKv = hQ / hKV;
        if (liveChunks >= 128u && qPerKv % 8u == 0u)
            decodeHeadGroup = 8u;
        else if (liveChunks >= 32u && qPerKv % 4u == 0u) decodeHeadGroup = 4u;
        else if (qPerKv % 2u == 0u) decodeHeadGroup = 2u;
    }
    const bool faUseWork = !kvQ8 && faDecodeSplit && faDecodeWork && (faFan > 1u ||
        (uint64_t)liveChunks * 4u < (uint64_t)zdim * splitChunks);
    uint32_t faWorkN = 0;
    if (faUseWork) {
        uint32_t* work = (uint32_t*)bFaWork.contents;
        if (faFan == 1u) {
            for (uint32_t s = 0; s < zdim; ++s)
                for (uint32_t c = 0; c < slotChunks[s]; ++c)
                    work[faWorkN++] = (s << 16u) | c;
        } else {
            for (uint32_t s = 0; s < zdim; ++s) {
                const uint32_t ng = std::max(
                    1u, (slotChunks[s] + faFan / 2u) / faFan);
                for (uint32_t g = 0; g < ng; ++g) {
                    const uint32_t first = (uint64_t)g * slotChunks[s] / ng;
                    const uint32_t end = (uint64_t)(g + 1u) * slotChunks[s] / ng;
                    work[2u * faWorkN] = (s << 16u) | g;
                    work[2u * faWorkN + 1u] = (first << 16u) | (end - first);
                    ++faWorkN;
                }
            }
        }
    }
    if (getenv("QK_FA_LOG")) {
        static uint64_t lastSig = UINT64_MAX;
        const uint64_t sig = (uint64_t)zdim << 56u | (uint64_t)splitChunks << 40u |
            (uint64_t)liveChunks << 16u | (uint64_t)faFan << 8u |
            (uint64_t)decodeHeadGroup << 4u | faUseWork;
        if (sig != lastSig) {
            fprintf(stderr, "[fa] slots=%u chunks=%u..%u live=%u fan=%u hg=%u work=%u/%u\n",
                    zdim, minChunks, splitChunks, liveChunks, faFan, decodeHeadGroup,
                    faUseWork ? faWorkN : 0u, zdim * splitChunks);
            lastSig = sig;
        }
    }
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
                             8, bwgs(chQkv, slotB ? slotTprA : gemvTprA), 256,
                             slotB ? zdim / slotBatchN : 0);
            if (L.iq4P2) dsp(pGemv4, {L.zW, bXn, bMid}, &pcZ, 8, dIn / 4, 64);
            else         dsp(slotB ? pGemvAB : pGemvA, {L.zW, bXn, bMid}, &pcZ,
                             8, bwgs(dIn, slotB ? slotTprA : gemvTprA), 256,
                             slotB ? zdim / slotBatchN : 0);
            dsp(pAb, {bXn, L.alW, L.beW, L.dt, L.av, bGb}, &pcAb, 8,
                (2 * hV + nsg - 1) / nsg, thrN);
            bar();
            const bool useDnSplit = dnDecodeSplitMode == 1 ||
                (dnDecodeSplitMode < 0 && (zdim <= 4u || zdim >= 7u));
            if (useDnSplit) {
                dsp(pStepPrep, {bBig, L.st1, L.ker}, &pcStep, 20, hK, dS);
                bar();
                id<MTLComputePipelineState> statePipe =
                    dnDecodeSplitMode < 0 && zdim >= 7u ? pStepStateSimd : pStepState;
                dsp(statePipe, {bBig, bGb, L.st2, bMid, L.sn, bAtt},
                    &pcStep, 20, hV, dS);
            } else {
                dsp(pStep, {bBig, L.st1, L.ker, bGb, L.st2, bMid, L.sn, bAtt},
                    &pcStep, 20, hK, dS);
            }
            bar();
            if (L.iq4Wo) dsp(pGemv4O, {L.outW, bAtt, bAttnOut}, &pcWo, 8, nEmbd / 4, 64);
            else         dsp(slotB ? pGemvOB : pGemvO, {L.outW, bAtt, bAttnOut}, &pcWo,
                             8, bwgs(nEmbd, slotB ? slotTprO : gemvTprO), 256,
                             slotB ? zdim / slotBatchN : 0);
        } else {
            if (L.iq4P1) dsp(pGemv4, {L.wq, bXn, bBig}, &pcQkv, 8, chQkv / 4, 64);
            else         dsp(slotB ? pGemvAB : pGemvA, {L.wq, bXn, bBig}, &pcQkv,
                             8, bwgs(chQkv, slotB ? slotTprA : gemvTprA), 256,
                             slotB ? zdim / slotBatchN : 0);
            if (L.iq4P2) dsp(pGemv4, {L.wk, bXn, bKin}, &pcKV, 8, hKV * dh / 4, 64);
            else         dsp(slotB ? pGemvAB : pGemvA, {L.wk, bXn, bKin}, &pcKV,
                             8, bwgs(hKV * dh, slotB ? slotTprA : gemvTprA), 256,
                             slotB ? zdim / slotBatchN : 0);
            if (L.iq4P3) dsp(pGemv4, {L.wv, bXn, bVin}, &pcKV, 8, hKV * dh / 4, 64);
            else         dsp(slotB ? pGemvAB : pGemvA, {L.wv, bXn, bVin}, &pcKV,
                             8, bwgs(hKV * dh, slotB ? slotTprA : gemvTprA), 256,
                             slotB ? zdim / slotBatchN : 0);
            bar();
            dsp(pPrep, {bBig, bKin, bVin, L.qn, L.kn, bMid, L.st1, L.st2, bRope, bSlotPos},
                &pcFa, 32, hQ + 2 * hKV, 256);
            bar();
            const bool useFaSplit = faDecodeSplit && (kvQ8 || (kvF16 && faF16Gqa) ||
                (!kvF16 && !kvQ8 && faF32Gqa) ||
                !(zdim == 1u && splitChunks == 1u && faFanOverride < 0));
            if (useFaSplit && !faUseWork) {
                id<MTLComputePipelineState> splitPipe = pAttnSplit;
                if (kvQ8 && decodeHeadGroup == 2u) splitPipe = pAttnQ8G2;
                else if (kvQ8 && decodeHeadGroup == 4u) splitPipe = pAttnQ8G4;
                else if (kvQ8 && decodeHeadGroup == 8u) splitPipe = pAttnQ8G8;
                else if (kvF16 && decodeHeadGroup == 2u) splitPipe = pAttnF16G2;
                else if (kvF16 && decodeHeadGroup == 4u) splitPipe = pAttnF16G4;
                else if (kvF16 && decodeHeadGroup == 8u) splitPipe = pAttnF16G8;
                else if (decodeHeadGroup == 2u) splitPipe = pAttnF32G2;
                else if (decodeHeadGroup == 4u) splitPipe = pAttnF32G4;
                else if (decodeHeadGroup == 8u) splitPipe = pAttnF32G8;
                dspY(splitPipe, {bMid, L.st1, L.st2, bPartial, bSlotPos},
                     &pcFaSplit, 36, hQ / decodeHeadGroup, splitChunks, 256);
                bar();
                dsp(pAttnReduce, {bPartial, bBig, bAtt, bSlotPos}, &pcFaSplit, 36, hQ, 256);
            } else if (useFaSplit) {
                dspY(faFan == 1u ? pAttnCompact : pAttnMulti,
                     {bMid, L.st1, L.st2, bPartial, bSlotPos, bFaWork},
                     &pcFaSplit, 36, hQ, faWorkN, 256, 1);
                bar();
                dsp(pAttnReduce, {bPartial, bBig, bAtt, bSlotPos},
                    &pcFaSplit, 36, hQ, 256);
            } else {
                dsp(pAttn, {bMid, L.st1, L.st2, bBig, bAtt, bSlotPos}, &pcFa, 32, hQ, 256);
            }
            bar();
            if (L.iq4Wo) dsp(pGemv4O, {L.wo, bAtt, bAttnOut}, &pcWo, 8, nEmbd / 4, 64);
            else         dsp(slotB ? pGemvOB : pGemvO, {L.wo, bAtt, bAttnOut}, &pcWo,
                             8, bwgs(nEmbd, slotB ? slotTprO : gemvTprO), 256,
                             slotB ? zdim / slotBatchN : 0);
        }
        bar();
        const bool splitMoeNorm = moeNormSplitMode == 1 ||
            (moeNormSplitMode < 0 && zdim >= 8u);
        if (splitMoeNorm) {
            dsp(pAddNSg, {bXin, bAttnOut, L.pn, bY, bXn2, bMoeScale},
                &pcRms, 8, 1, 32);
            bar();
            dsp(pMoeLS, {L.mgi, L.mgis, bY, L.pn, bMoeScale, bML},
                &pcv, 16, (nExp + 1 + nsg - 1) / nsg, thrN);
        } else {
            dsp(pMoeLA, {L.mgi, L.mgis, bXin, bAttnOut, L.pn, bML, bY, bXn2},
                &pcv5, 20, (nExp + 1 + nsg - 1) / nsg, thrN);
        }
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

void qk_engine::prefillPendingSlots() {
    // 80B's IQ4 grouped-MoE path changes activation arithmetic when prompt
    // rows are unioned (the recurrent state diverges immediately after layer
    // 0). Its production split path is identity-scheduled; keep unsplit 80B
    // on serial prefill as well until a per-sequence IQ4 MoE union exists.
    if (!prefillMulti || guIq4 || nSlots < 2 || maxB < 384) return;
    std::vector<float> dummy;

    // Coalesce only chunks large enough that each sequence independently uses
    // the same exact grouped-MoE regime as serial prefill. Smaller unions can
    // be much faster by crossing that threshold in aggregate, but recurrent
    // trajectories then differ and are not eligible for the exact default.
    while (true) {
        struct Ready { uint32_t logical, rem; };
        std::vector<Ready> ready;
        for (uint32_t s = 0; s < nSlots; ++s) {
            const Slot& sl = slots[s];
            if (!sl.active || sl.finPending || sl.prompt.empty()) continue;
            const uint32_t target = (uint32_t)sl.prompt.size() - 1u;
            if (target > sl.cursor && target - sl.cursor >= 192u)
                ready.push_back({s, target - sl.cursor});
        }
        std::sort(ready.begin(), ready.end(), [](const Ready& a, const Ready& b) {
            return a.rem > b.rem;
        });

        uint32_t B = 0, Tn = 0;
        for (uint32_t candidate : {8u, 4u, 2u}) {
            if (candidate > ready.size()) continue;
            uint32_t minRem = ready[0].rem;
            for (uint32_t q = 1; q < candidate; ++q)
                minRem = std::min(minRem, ready[q].rem);
            const uint32_t lim = std::min(maxB / candidate, minRem);
            const uint32_t tail = minRem - lim;
            // Avoid creating a 16..191-token tail that cannot be coalesced and
            // would require one separate prefill per client. A narrower union
            // can often consume the whole remaining prefix instead.
            if (tail >= 16u && tail < 192u) continue;
            if (lim >= 192u) { B = candidate; Tn = lim; break; }
        }
        if (!B) break;

        std::vector<uint32_t> ids((size_t)B * Tn), stripes(B), bases(B);
        for (uint32_t q = 0; q < B; ++q) {
            Slot& sl = slots[ready[q].logical];
            memcpy(ids.data() + (size_t)q * Tn, sl.prompt.data() + sl.cursor,
                   (size_t)Tn * sizeof(uint32_t));
            stripes[q] = physicalSlot(ready[q].logical);
            bases[q] = sl.cursor;
        }
        prefillBatchLast(ids.data(), B * Tn, 0, dummy, /*wantLogits=*/false, 0,
                         nullptr, nullptr, nullptr, false, B,
                         stripes.data(), bases.data());
        for (uint32_t q = 0; q < B; ++q) {
            Slot& sl = slots[ready[q].logical];
            sl.cursor += Tn;
            sl.pos += Tn;
        }
    }

    // Finish each residual batchable prefix through the existing single-slot
    // path. At most 191 tokens remain after a coalesced round; tails below 16
    // stay in the serial multi-slot step, matching the legacy policy.
    for (uint32_t s = 0; s < nSlots; ++s) {
        Slot& sl = slots[s];
        if (!sl.active || sl.finPending || sl.prompt.empty()) continue;
        const uint32_t target = (uint32_t)sl.prompt.size() - 1u;
        while (target > sl.cursor && target - sl.cursor >= 16u) {
            const uint32_t cn = std::min(maxB, target - sl.cursor);
            prefillBatchLast(sl.prompt.data() + sl.cursor, cn, physicalSlot(s), dummy,
                             /*wantLogits=*/false, sl.cursor);
            sl.cursor += cn;
            sl.pos += cn;
        }
    }
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
            slotsNeedCompact = true;
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
    compactActiveSlots();
    prefillPendingSlots();

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
                    prefillBatchLast(specToks.data(), accepted, physicalSlot((uint32_t)only), dummy,
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
                    slotsNeedCompact = true;
                    finish((uint32_t)only);
                    break;
                }
            }
        }
    }

    // Serial fallback for slots with no high-confidence draft and for all
    // multi-slot steps. It is also responsible for prompt ingestion.
    for (uint32_t step = 0; step < chunkN; step++) {
        // A slot may have completed in the preceding iteration. Move only its
        // surviving neighbors now, after every bTok row from that dispatch has
        // been consumed.
        compactActiveSlots();
        int nAct = 0;
        uint32_t maxZ = 0;
        bool need = false;
        for (uint32_t p = 0; p < nSlots; ++p) {
            slotIn[p] = 0;
            slotPos[p] = 0;
        }
        for (uint32_t s = 0; s < nSlots; s++) {
            Slot& sl = slots[s];
            if (!sl.active || sl.finPending) continue;
            const uint32_t p = physicalSlot(s);
            nAct++;
            maxZ = std::max(maxZ, p + 1);
            if (sl.cursor < sl.prompt.size()) {
                slotIn[p] = sl.prompt[sl.cursor];
                slotPos[p] = sl.cursor;
                need = true;
            } else {
                slotIn[p] = sl.last;
                slotPos[p] = sl.pos;
                if (outCnt[s] < chunkN) need = true;
            }
        }
        if (!nAct || !need) break;

        bool equivalentSlots = moeSlotGrouped && maxZ >= 2 && nAct == (int)maxZ;
        if (equivalentSlots) {
            const Slot* r = nullptr;
            for (uint32_t s = 0; s < nSlots; ++s)
                if (slots[s].active && !slots[s].finPending) { r = &slots[s]; break; }
            for (uint32_t s = 0; s < nSlots; ++s) {
                const Slot& q = slots[s];
                if (!q.active || q.finPending || &q == r) continue;
                if (q.equivId != r->equivId || q.cursor != r->cursor ||
                    q.pos != r->pos || q.genTokens.size() != r->genTokens.size() ||
                    q.last != r->last) {
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
            uint32_t sampled = tok[physicalSlot(s)];
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
                slotsNeedCompact = true;
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
                slotsNeedCompact = true;
                finish(s);
            }
        }
    }
    return activeAtEntry;
}

void qk_engine::prefillBatchLast(const uint32_t* toks, uint32_t n, uint32_t slot,
                                 std::vector<float>& logits, bool wantLogits, uint32_t base,
                                 uint32_t* argmaxOut, const float* hiddenIn, float* hiddenOut,
                                 bool scratchState, uint32_t nSeq,
                                 const uint32_t* seqSlots, const uint32_t* seqBases) {
    const uint32_t seqN = nSeq ? n / nSeq : 0;
    const uint32_t seqPad = (seqN + 63u) & ~63u;
    auto stateSlotFor = [&](uint32_t q) {
        return nSeq == 1 ? slot : seqSlots[q];
    };
    auto baseFor = [&](uint32_t q) {
        return nSeq == 1 ? base : seqBases[q];
    };
    bool badSeq = nSeq < 1 || nSeq > nSlots || !seqN || seqN * nSeq != n ||
                  (size_t)nSeq * seqPad > ((maxB + 63u) & ~63u) ||
                  (scratchState && !specScratch) ||
                  (nSeq > 1 && (!seqSlots || !seqBases || hiddenIn || wantLogits ||
                                argmaxOut || scratchState));
    if (!badSeq) {
        for (uint32_t q = 0; q < nSeq; ++q)
            badSeq |= stateSlotFor(q) >= nSlots || (size_t)baseFor(q) + seqN > nCtx;
    }
    if ((!toks && !hiddenIn) || n < 1 || n > maxB || slot >= nSlots || badSeq ||
        (hiddenIn && firstStage()) || (!hiddenIn && !firstStage()) ||
        ((wantLogits || argmaxOut) && !lastStage())) {
        fprintf(stderr, "prefillBatchLast: bad args n=%u seq=%u slot=%u base=%u "
                        "stage=[%u,%u)\n", n, nSeq, slot, base, lFirst, lEnd);
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
    const bool faMma = !getenv("QK_FA_MMA") || atoi(getenv("QK_FA_MMA")) != 0;
    if (wantLogits) logits.resize(vocab);
    for (uint32_t q = 0; q < nSeq; ++q)
        if (baseFor(q) == 0) resetStripe(stateSlotFor(q));
    // Seed each sequence's DeltaNet conv carry (plain UMA memcpy; GPU idle).
    for (uint32_t il = lFirst; il < lEnd; il++) {
        if (!layers[il].rec) continue;
        for (uint32_t q = 0; q < nSeq; ++q) {
            const uint32_t stateSlot = scratchState ? nSlots : stateSlotFor(q);
            uint8_t* dst = (uint8_t*)bbCarry.contents +
                layers[il].carryOff + (size_t)q * chQkv * 3 * 4;
            if (baseFor(q) == 0) memset(dst, 0, (size_t)chQkv * 3 * 4);
            else memcpy(dst, (uint8_t*)layers[il].st1.contents +
                             (size_t)stateSlot * layers[il].ps1,
                        (size_t)chQkv * 3 * 4);
        }
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        StageCounterTrace counter(c);
        id<MTLComputeCommandEncoder> enc = counter.enabled()
            ? counter.begin(cb)
            : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        auto nextPass = [&](const char* label) {
            if (!counter.enabled()) return;
            counter.end(enc, label);
            enc = counter.begin(cb);
        };
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
            pcConvB{chQkv, dS, 2 * hK * dS, eps, seqN};
        struct { uint32_t dState, hK_, hV_, Tn, kd; }
            pcStepB{dS, hK, hV, seqN, dnKDiv};
        struct { uint32_t dState, hK_, hV_, Tn, kd; }
            pcDnC{dS, hK, hV, seqN, dnKDiv};
        const uint32_t nChDn = (seqN + 63) / 64;
        struct { uint32_t dState, hV_; float e; uint32_t Tn; } pcGateB{dS, hV, eps, n};
        struct { uint32_t a, b, cc, d; } pcv{nEmbd, ffE, nExp, nUsed};
        struct { uint32_t a, b, cc, d; float e; } pcv5{nEmbd, ffE, nExp, nUsed, eps};
        struct { uint32_t tmax, dh_, nRot_, hQ_, hKV_; float e, fb; uint32_t base_, Tn, qbase; }
            pcFaB{nCtx, dh, nRot, hQ, hKV, eps, kFreqBase, baseFor(0), seqN, 0};
        struct { uint32_t k, idx, pr; float e; } pcE{nEmbd, 0, 1, eps};
        struct { uint32_t M, K, N; } pcG;

        // projection: tiled GEMM for n>=48; exact weight reuse below it.
        auto proj = [&](WB W, WB WP, id<MTLBuffer> X, id<MTLBuffer> Y,
                        uint32_t M, uint32_t K, bool isOut, bool iq4) {
            if (skProj) return;
            if (n >= 48) {
                pcG = {M, K, n};
                if (iq4)
                    dspz(pGemmB4Aligned && M % 64u == 0u && n % 32u == 0u
                             ? pGemmB4Aligned : pGemmB4,
                         {W, X, Y}, &pcG, 12, (M + 63) / 64, 128, (n + 31) / 32);
                else if (prefillRepack && prefillRepackUse && WP.b && pGemmBK &&
                         M % 64u == 0u && n % 32u == 0u)
                    dspz(pGemmBK, {WP, X, Y}, &pcG, 12,
                         M / 64u, 128, n / 32u);
                else
                    dspz(pGemmBAligned && M % 64u == 0u && n % 32u == 0u
                             ? pGemmBAligned : pGemmB,
                         {W, X, Y}, &pcG, 12, (M + gemmBM - 1) / gemmBM, gemmThreads,
                         (n + gemmBN - 1) / gemmBN);
            } else {
                pcP = {M, K};
                if (iq4)
                    dspz(isOut ? pGemv4O : pGemv4, {W, X, Y}, &pcP, 8,
                         (M + 3) / 4, 64, n);
                else {
                    const uint32_t tpr = isOut ? gemvTprO : gemvTprA;
                    const uint32_t wgs = (M + 256u / tpr - 1u) / (256u / tpr);
                    const uint32_t n4 = prefillGemvBatch ? n & ~3u : 0u;
                    if (n4)
                        dspz(isOut ? pPrefillGemvO4 : pPrefillGemvA4,
                             {W, X, Y}, &pcP, 8, wgs, 256, n4 / 4u);
                    if (n4 < n)
                        dspz(isOut ? pGemvO : pGemvA,
                             {W,
                              WB{X, (NSUInteger)((size_t)n4 * K * sizeof(float))},
                              WB{Y, (NSUInteger)((size_t)n4 * M * sizeof(float))}},
                             &pcP, 8, wgs, 256, n - n4);
                }
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
        nextPass("input");
        // Preserve the per-sequence numerical regime when rows from several
        // prompts share the activation matrix. Crossing the grouped-MoE
        // threshold only in aggregate changes recurrent trajectories.
        const bool useGrouped = moeGrouped && seqN >= moeGroupN;
        for (uint32_t il = lFirst; il < lEnd; il++) {
            Layer& L = layers[il];
            if (L.rec) {
                proj(L.qkvW, L.qkvP, bbXn, bbBig, chQkv, nEmbd, false, L.iq4P1);
                proj(L.zW, L.zP, bbXn, bbMid, dIn, nEmbd, false, L.iq4P2);
                dspz(pAbB, {bbXn, L.alW, L.beW, L.dt, L.av, bbGb}, &pcAbB, 12,
                     (2 * hV + nsg - 1) / nsg, thrN, n);
                bar();
                nextPass("rec-proj");
                if (!skDn) {
                for (uint32_t q = 0; q < nSeq; ++q) {
                    const uint32_t stateSlot = scratchState ? nSlots : stateSlotFor(q);
                    const size_t ro = (size_t)q * seqN;
                    const NSUInteger carryOff =
                        L.carryOff + (NSUInteger)q * chQkv * 3 * 4;
                    dspz(pConvB,
                         {WB{bbCarry, carryOff},
                          WB{bbBig, (NSUInteger)(ro * chQkv * 4)}, L.ker,
                          WB{bbConvOut, (NSUInteger)(ro * chQkv * 4)},
                          WB{L.st1, (NSUInteger)((size_t)stateSlot * L.ps1)}},
                         &pcConvB, 20, chQkv / dS, dS, seqN);
                }
                bar();
                nextPass("dn-conv");
                if (dnChunk) {
                    const size_t kqStride = (size_t)nChDn * hK * 2 * 64 * 64 * 4;
                    const size_t uwStride = (size_t)nChDn * hV * 4 * 64 * dS * 4;
                    const size_t atStride = (size_t)nChDn * hV * 64 * 64 * 4;
                    const size_t elStride = (size_t)nChDn * hV * 4;
                    for (uint32_t q = 0; q < nSeq; ++q) {
                        const size_t ro = (size_t)q * seqN;
                        dspz(pDnKq,
                             {WB{bbConvOut, (NSUInteger)(ro * chQkv * 4)},
                              WB{bbDnKQ, (NSUInteger)(q * kqStride)}},
                             &pcDnC, 20, hK, dS, nChDn);
                    }
                    bar();
                    nextPass("dn-kq");
                    for (uint32_t q = 0; q < nSeq; ++q) {
                        const size_t ro = (size_t)q * seqN;
                        dspz(pDnSolve,
                             {WB{bbConvOut, (NSUInteger)(ro * chQkv * 4)},
                              WB{bbGb, (NSUInteger)(ro * 2 * hV * 4)},
                              WB{bbDnKQ, (NSUInteger)(q * kqStride)},
                              WB{bbDnUW, (NSUInteger)(q * uwStride)},
                              WB{bbDnAtt, (NSUInteger)(q * atStride)},
                              WB{bbDnEl, (NSUInteger)(q * elStride)}},
                             &pcDnC, 20, hV, dS, nChDn);
                    }
                    bar();
                    nextPass("dn-solve");
                    for (uint32_t q = 0; q < nSeq; ++q) {
                        const uint32_t stateSlot = scratchState ? nSlots : stateSlotFor(q);
                        const size_t oo = (size_t)q * seqPad;
                        dspz(pDnStepC,
                             {WB{bbDnUW, (NSUInteger)(q * uwStride)},
                              WB{bbDnAtt, (NSUInteger)(q * atStride)},
                              WB{bbDnEl, (NSUInteger)(q * elStride)},
                              WB{bbO, (NSUInteger)(oo * dIn * 4)},
                              WB{L.st2, (NSUInteger)((size_t)stateSlot * L.ps2)}},
                             &pcDnC, 20, hV, dnStepPW ? 256u : dS,
                             dnStepPW ? dS / dnStepPW : 2u);
                    }
                } else {
                    for (uint32_t q = 0; q < nSeq; ++q) {
                        const uint32_t stateSlot = scratchState ? nSlots : stateSlotFor(q);
                        const size_t ro = (size_t)q * seqN;
                        const size_t oo = (size_t)q * seqPad;
                        dspz(pStepB,
                             {WB{bbConvOut, (NSUInteger)(ro * chQkv * 4)},
                              WB{bbGb, (NSUInteger)(ro * 2 * hV * 4)},
                              WB{bbO, (NSUInteger)(oo * dIn * 4)},
                              WB{L.st2, (NSUInteger)((size_t)stateSlot * L.ps2)}},
                             &pcStepB, 20, hV, dS, 1);
                    }
                }
                bar();
                nextPass("dn-step");
                for (uint32_t q = 0; q < nSeq; ++q) {
                    const size_t ro = (size_t)q * seqN;
                    const size_t oo = (size_t)q * seqPad;
                    dspz(pGateB,
                         {WB{bbO, (NSUInteger)(oo * dIn * 4)}, L.sn,
                          WB{bbMid, (NSUInteger)(ro * dIn * 4)},
                          WB{bbAtt, (NSUInteger)(ro * dIn * 4)}},
                         &pcGateB, 16, (hV + nsg - 1) / nsg, thrN, seqN);
                }
                bar();
                nextPass("dn-gate");
                }
                proj(L.outW, L.outP, bbAtt, bbAttnOut, nEmbd, dIn, true, L.iq4Wo);
            } else {
                proj(L.wq, L.wqP, bbXn, bbBig, chQkv, nEmbd, false, L.iq4P1);
                proj(L.wk, L.wkP, bbXn, bbKin, hKV * dh, nEmbd, false, L.iq4P2);
                proj(L.wv, L.wvP, bbXn, bbVin, hKV * dh, nEmbd, false, L.iq4P3);
                bar();
                nextPass("attn-proj");
                if (!skAttn) {
                for (uint32_t q = 0; q < nSeq; ++q) {
                    const uint32_t stateSlot = stateSlotFor(q);
                    const size_t ro = (size_t)q * seqN;
                    pcFaB.base_ = baseFor(q);
                    dspz(pPrepB,
                         {WB{bbBig, (NSUInteger)(ro * chQkv * 4)},
                          WB{bbKin, (NSUInteger)(ro * hKV * dh * 4)},
                          WB{bbVin, (NSUInteger)(ro * hKV * dh * 4)}, L.qn, L.kn,
                          WB{bbMid, (NSUInteger)(ro * dIn * 4)},
                          WB{L.st1, (NSUInteger)((size_t)stateSlot * L.ps1)},
                          WB{L.st2, (NSUInteger)((size_t)stateSlot * L.ps2)}, bRope},
                         &pcFaB, 40, hQ + 2 * hKV, 256, seqN);
                }
                bar();
                nextPass("attn-prep");
                for (uint32_t q = 0; q < nSeq; ++q) {
                    const uint32_t stateSlot = stateSlotFor(q);
                    const size_t ro = (size_t)q * seqN;
                    pcFaB.base_ = baseFor(q);
                    if (faMma)
                        dspz(pAttnBM,
                             {WB{bbMid, (NSUInteger)(ro * dIn * 4)},
                              WB{L.st1, (NSUInteger)((size_t)stateSlot * L.ps1)},
                              WB{L.st2, (NSUInteger)((size_t)stateSlot * L.ps2)},
                              WB{bbBig, (NSUInteger)(ro * chQkv * 4)},
                              WB{bbAtt, (NSUInteger)(ro * dIn * 4)}},
                             &pcFaB, 40, hQ / faHeadGroup, faThreads,
                             (seqN + faQTM - 1u) / faQTM);
                    else
                        dspz(pAttnB,
                             {WB{bbMid, (NSUInteger)(ro * dIn * 4)},
                              WB{L.st1, (NSUInteger)((size_t)stateSlot * L.ps1)},
                              WB{L.st2, (NSUInteger)((size_t)stateSlot * L.ps2)},
                              WB{bbBig, (NSUInteger)(ro * chQkv * 4)},
                              WB{bbAtt, (NSUInteger)(ro * dIn * 4)}},
                             &pcFaB, 40, hQ, 256, seqN);
                }
                bar();
                nextPass("attention");
                }
                proj(L.wo, L.woP, bbAtt, bbAttnOut, nEmbd, dIn, true, L.iq4Wo);
            }
            bar();
            nextPass(L.rec ? "out-proj" : "attn-out");
            if (useGrouped) {
                // grouped regime: residual+norm via add_rmsnorm (same y/xn2 as
                // the fused kernel), router logits as one f32 GEMM
                dspz(pAddN, {bbXin, bbAttnOut, L.pn, bbY, bbXn2}, &pcRms, 8, 1, 256, n);
                bar();
                nextPass("residual");
                struct { uint32_t M, K, N; } pcLg{nExp + 1, nEmbd, n};
                dspz(pLogG, {L.mgi, L.mgis, bbXn2, bbML}, &pcLg, 12, (nExp + 1 + 31) / 32, 128,
                     (n + 31) / 32);
            } else {
                dspz(pMoeLA, {L.mgi, L.mgis, bbXin, bbAttnOut, L.pn, bbML, bbY, bbXn2},
                     &pcv5, 20, (nExp + 1 + nsg - 1) / nsg, thrN, n);
            }
            bar();
            nextPass(useGrouped ? "router" : "router+resid");
            if (!skMoe) {
            dspz(pMoeS, {bbML, bbMSel}, &pcv, 16, 1, 32, n);
            bar();
            nextPass("select");
            if (useGrouped) {
                struct { uint32_t a, b, cc, d, n; } pcg{nEmbd, ffE, nExp, nUsed, n};
                const bool compactWork = (moeGrouped == 5 || moeGrouped == 6) &&
                                         !guIq4 && moeWork;
                const uint32_t workCap = nExp + (n * nUsed + 31) / 32 + (n + 31) / 32;
                if (compactWork)
                    dspz(pMoeGrpWork, {bbMSel, bbStart, bbATok, bbASlot, bbWork},
                         &pcg, 20, 1, 256, 1);
                else
                    dspz(pMoeGrp, {bbMSel, bbStart, bbATok, bbASlot},
                         &pcg, 20, 1, 256, 1);
                bar();
                nextPass("moe-group");
                if (!skGu) {
                if (compactWork)
                    dspz(pMoeGuG5Work,
                         {L.mge, L.mue, L.mgs, L.mus, bbXn2, bbStart, bbATok,
                          bbASlot, bbMH, bbWork}, &pcv, 16,
                         workCap * (ffE / 32), 128, 1);
                else if (moeGrouped == 5 || moeGrouped == 6)
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
                nextPass("moe-gateup");
                if (!skDown) {
                if (compactWork)
                    dspz(L.downQ6 ? pMoeDGP6Work : pMoeDGP4Work,
                         {L.mde, L.mds, bbMH, bbStart, bbATok, bbASlot, bbMDy, bbWork},
                         &pcv, 16, workCap * (nEmbd / 64), 128, 1);
                else if (moeGrouped == 5 || moeGrouped == 6)
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
                nextPass("moe-down");
                dspz(pMoeDR, {bbMDy, bbMSel, bbMY}, &pcv, 16, (nEmbd + 255) / 256, 256, n);
                }
            } else {
                if (!skGu)
                dspz(guIq4 ? pMoeGu4 : pMoeGu,
                     {L.mge, L.mue, L.mgs, L.mus, bbXn2, bbMSel, bbMH}, &pcv, 16,
                     ((nUsed + 1) * ffE + nsg - 1) / nsg, thrN, n);
                bar();
                nextPass("moe-gateup");
                if (!skDown)
                dspz(L.downQ6 ? pMoeD6 : pMoeD4, {L.mde, L.mds, bbMH, bbMSel, bbMY},
                     &pcv, 16, (nEmbd + nsg - 1) / nsg, thrN, n);
            }
            bar();
            nextPass(useGrouped ? "moe-reduce" : "moe-down");
            }
            WB nextNorm = il + 1 < lEnd ? layers[il + 1].aNorm : bONorm;
            dspz(pAddN, {bbY, bbMY, nextNorm, bbXin, bbXn}, &pcRms, 8, 1, 256, n);
            bar();
            nextPass("residual");
        }
        if (wantLogits || argmaxOut) {
            // per-position ids (argmaxOut: stageRun wire contract) need the
            // head for every row; logits-only callers need just the LAST row
            // — the z=n head is ~417 MB of Q6_K weight reads PER ROW.
            const uint32_t nh = argmaxOut ? n : 1;
            const bool tiledHead = argmaxOut && headGemm && nh >= headGemmN;
            if (tiledHead) {
                const bool f16 = headF16 && nh >= 4u;
                const bool wide = f16 || nh >= 8u;
                const uint32_t headRows = wide ? 64u : 32u;
                const uint32_t headCols = f16 ? (nh <= 8u ? 8u : nh <= 16u ? 16u : 32u)
                                              : 8u;
                const uint32_t headTiles = (vocab + headRows - 1u) / headRows;
                const uint32_t materialize = (!scratchState || wantLogits) ? 1u : 0u;
                struct { uint32_t m, k, nn, tiles, materialize; }
                    pcHead{vocab, nEmbd, nh, headTiles, materialize};
                id<MTLComputePipelineState> hp = f16
                    ? (headCols == 8u ? pHeadF8 : headCols == 16u ? pHeadF16 : pHeadF32)
                    : (wide ? pHeadTop64 : pHeadTop);
                dspz(hp,
                     {bHeadW, bbXn, bbHeadV, bbHeadI, bbLogits},
                     &pcHead, 20, headTiles, wide ? 256 : 128,
                     (nh + headCols - 1u) / headCols);
                bar();
                nextPass("head");
                struct { uint32_t tiles, nn; } pcRed{headTiles, nh};
                dspz(pHeadTopReduce, {bbHeadV, bbHeadI, bbTok}, &pcRed, 8, 1, 256, nh);
                lastRunRows = 1;
            } else {
                lastRunRows = nh;
                struct { uint32_t m, k; } pcHead{vocab, nEmbd};
                dspz(pHead,
                     {bHeadW, nh == 1 ? WB{bbXn, (NSUInteger)(n - 1) * nEmbd * 4}
                                      : WB{bbXn},
                      bbLogits},
                     &pcHead, 8, (vocab + 3) / 4, 64, nh);
                if (argmaxOut) {
                    bar();
                    nextPass("head");
                    const uint32_t amWgs = (vocab + 4095) / 4096;
                    struct { uint32_t nn, span; } pcAm{vocab, 4096};
                    struct { uint32_t m, pos; } pcAm2{amWgs, 0};
                    dspz(pAm1, {bbLogits, bbAV, bbAI}, &pcAm, 8, amWgs, 256, n);
                    bar();
                    nextPass("argmax-1");
                    dspz(pAm2, {bbAV, bbAI, bbTok, bRbScratch}, &pcAm2, 8, 1, 256, n);
                }
            }
        }
        if (counter.enabled()) {
            if (wantLogits || argmaxOut) counter.end(enc, "head-tail");
            else counter.discardEmpty(enc);
        } else {
            [enc endEncoding];
        }
        auto tw0 = std::chrono::steady_clock::now();
        [cb commit];
        [cb waitUntilCompleted];
        lastCbGpu = cb.GPUEndTime - cb.GPUStartTime;
        lastCbWall = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
        counter.report(lastCbGpu);
    }
    if (wantLogits)
        memcpy(logits.data(),
               (uint8_t*)bbLogits.contents + (size_t)(lastRunRows - 1u) * vocab * 4,
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
    resetStripe(slot);
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
    return e->copyPackedStripes(slot, e->pcache[idx], /*save=*/true, n_tok) ? 0 : -2;
}

int qk_state_load(qk_engine* e, uint32_t slot, uint32_t idx, uint32_t n_tok) {
    if (!e || slot >= e->nSlots || idx >= e->pcache.size()) return -1;
    return e->copyPackedStripes(slot, e->pcache[idx], /*save=*/false, n_tok) ? 0 : -2;
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
    const uint32_t stripe = e->acquirePhysicalSlot(slot);
    int cidx = e->matchPrefix(prompt, n_prompt);
    uint32_t start;
    if (cidx >= 0) {
        e->restoreInto(slot, cidx);
        start = (uint32_t)e->pcache[cidx].tokens.size();
    } else {
        e->resetStripe(stripe);
        start = 0;
    }
    uint32_t target = n_prompt >= 1 ? n_prompt - 1 : 0;
    uint32_t done = start;
    const bool deferPrefill = e->prefillMulti && !e->guIq4 && e->nSlots > 1 &&
                              e->maxB >= 384 &&
                              snap_prefix == 0 && !e->shareFork &&
                              !getenv("QK_NO_BATCH");
    auto batch_to = [&](uint32_t limit) {
        while (limit > done && limit - done >= 16) {
            uint32_t chunk = std::min(e->maxB, limit - done);
            std::vector<float> unused;
            e->prefillBatchLast(prompt + done, chunk, stripe, unused,
                                /*wantLogits=*/false, /*base=*/done);
            done += chunk;
        }
    };
    uint32_t snapAt = (snap_prefix > start && snap_prefix < target) ? snap_prefix : 0;
    uint32_t snapPos = 0;
    if (!deferPrefill && !getenv("QK_NO_BATCH")) {
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
        e->slotsNeedCompact = true;
        e->slots[slot].prompt.clear();
        e->slots[slot].equivId = 0;
        e->slots[slot].resetSpec();
        e->compactActiveSlots();
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
    if (gv && !strcmp(gv, "hpa")) fn = "gemm_q8_0_hp_aligned";
    if (gv && !strcmp(gv, "kma")) fn = "gemm_q8_0_hp_kmajor";
    bool scalar = !strcmp(fn, "gemm_q8_0");
    const bool aligned = !strcmp(fn, "gemm_q8_0_hp_aligned");
    const bool kmajor = !strcmp(fn, "gemm_q8_0_hp_kmajor");
    const bool hp = !strcmp(fn, "gemm_q8_0_hp") || !strcmp(fn, "gemm_q8_0_hp_aligned") || kmajor;
    const uint32_t tBM = !strcmp(fn, "gemm_q8_0_h2") ? 64 : hp ? 64 : 128;
    const uint32_t tBN = !strcmp(fn, "gemm_q8_0_h2") ? 128 : hp ? 32 : 64;
    id<MTLComputePipelineState> pso = getPipe(c, "gemm_q8_0", fn, 0);
    printf("variant: %s\n", fn);
    std::vector<block_q8_0> kmajorBlocks;
    const block_q8_0* candidateBlocks = blocks.data();
    if (kmajor) {
        const uint32_t kb = K / 32u;
        kmajorBlocks.resize(nb);
        for (uint32_t b = 0; b < kb; ++b)
            for (uint32_t m = 0; m < M; ++m)
                kmajorBlocks[(size_t)b * M + m] = blocks[(size_t)m * kb + b];
        candidateBlocks = kmajorBlocks.data();
    }
    id<MTLBuffer> bW = createBuf(c, nb * sizeof(block_q8_0), candidateBlocks);
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
    if (aligned || kmajor) {
        id<MTLBuffer> bSafe = createBuf(c, yref.size() * 4);
        id<MTLBuffer> bSafeW = kmajor
            ? createBuf(c, nb * sizeof(block_q8_0), blocks.data()) : bW;
        id<MTLComputePipelineState> pSafe =
            getPipe(c, "gemm_q8_0", "gemm_q8_0_hp", 0);
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pSafe];
            [enc setBuffer:bSafeW offset:0 atIndex:0];
            [enc setBuffer:bX offset:0 atIndex:1];
            [enc setBuffer:bSafe offset:0 atIndex:2];
            [enc setBytes:&pc length:12 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((M + 63u) / 64u, 1, (N + 31u) / 32u)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        const uint32_t* ya = (const uint32_t*)bY.contents;
        const uint32_t* ys = (const uint32_t*)bSafe.contents;
        size_t mismatches = 0;
        for (size_t i = 0; i < yref.size(); ++i) mismatches += ya[i] != ys[i];
        printf("%s-vs-safe: %zu/%zu bit mismatches -> %s\n",
               kmajor ? "kmajor" : "aligned", mismatches, yref.size(),
               mismatches ? "FAIL" : "EXACT");
        if (kmajor && iters) {
            auto timed = [&](id<MTLComputePipelineState> pipe, id<MTLBuffer> weights,
                             id<MTLBuffer> out) -> double {
                @autoreleasepool {
                    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    for (uint32_t i = 0; i < iters; ++i) {
                        [enc setComputePipelineState:pipe];
                        [enc setBuffer:weights offset:0 atIndex:0];
                        [enc setBuffer:bX offset:0 atIndex:1];
                        [enc setBuffer:out offset:0 atIndex:2];
                        [enc setBytes:&pc length:12 atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake((M + 63u) / 64u, 1,
                                                              (N + 31u) / 32u)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / iters;
                }
            };
            const double a0 = timed(pSafe, bSafeW, bSafe);
            const double b0 = timed(pso, bW, bY);
            const double b1 = timed(pso, bW, bY);
            const double a1 = timed(pSafe, bSafeW, bSafe);
            printf("layout bracket us: row %.1f / kmajor %.1f %.1f / row %.1f\n",
                   a0, b0, b1, a1);
        }
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

// Real-weight Q8_0 GEMV comparison.  Report the compiler's sub-ULP reassociation
// from fixed-trip-count unrolling; the production TPR choice is separately
// required to preserve the token stream through the full gates.
static bool caseQ8GemvCmp(MtlCtx& c, const std::string& tensorName,
                          uint32_t tpr, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    const GgufTensor* t = g.find(tensorName);
    if (!t || t->type != GGML_Q8_0 || t->ne[2] != 1) {
        fprintf(stderr, "q8gemvcmp requires a 2D Q8_0 tensor\n");
        return false;
    }
    const uint32_t K = (uint32_t)t->ne[0];
    const uint32_t M = (uint32_t)t->ne[1];
    if ((K != 2048u && K != 4096u) || tpr < 8u || tpr > 256u ||
        (tpr & (tpr - 1u)) != 0u || M % (256u / tpr) != 0u) {
        fprintf(stderr, "q8gemvcmp requires K={2048,4096}, power-of-two TPR, and full row tiles\n");
        return false;
    }
    const size_t wBytes = (size_t)M * ggmlRowBytes(t->type, K);
    printf("\n== q8gemvcmp %s W[%u,%u], TPR=%u (W %.1f MiB) ==\n",
           tensorName.c_str(), M, K, tpr, (double)wBytes / (1 << 20));

    std::mt19937 rng(0x51384745u);
    std::normal_distribution<float> nd(0.f, 0.35f);
    std::vector<float> x(K);
    for (float& v : x) v = nd(rng);

    id<MTLBuffer> bW = createBuf(c, wBytes, t->data);
    id<MTLBuffer> bX = createBuf(c, (size_t)K * 4u, x.data());
    id<MTLBuffer> bSafe = createBuf(c, (size_t)M * 4u);
    id<MTLBuffer> bFixed = createBuf(c, (size_t)M * 4u);
    id<MTLComputePipelineState> pSafe = getPipe(c, "gemv_q8_0", "gemv_q8_0", tpr);
    id<MTLComputePipelineState> pFixed = getPipe(
        c, "gemv_q8_0_fixed", K == 2048u ? "gemv_q8_0_k2048" : "gemv_q8_0_k4096", tpr);
    const uint32_t mk[2] = {M, K};
    const uint32_t wgs = M / (256u / tpr);

    auto encode = [&](id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState> pso, id<MTLBuffer> y) {
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:y offset:0 atIndex:2];
        [enc setBytes:mk length:sizeof(mk) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto once = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode(enc, pso, y);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
    };
    once(pSafe, bSafe);
    once(pFixed, bFixed);
    const uint32_t* ys = (const uint32_t*)bSafe.contents;
    const uint32_t* yf = (const uint32_t*)bFixed.contents;
    size_t mismatches = 0;
    double rms = 0.0;
    const float* fs = (const float*)ys;
    const float* ff = (const float*)yf;
    for (uint32_t i = 0; i < M; ++i) rms += (double)fs[i] * fs[i];
    rms = std::sqrt(rms / M);
    double maxAbs = 0.0;
    for (uint32_t i = 0; i < M; ++i) {
        mismatches += ys[i] != yf[i];
        const double d = std::fabs((double)fs[i] - ff[i]);
        maxAbs = std::max(maxAbs, d);
    }
    const double scaled = maxAbs / std::max(1e-7, rms);
    const bool close = scaled <= 2e-6;
    printf("fixed-vs-safe: %zu/%u bit mismatches, max_abs %.3g, max_abs/rms %.3g -> %s\n",
           mismatches, M, maxAbs, scaled, mismatches ? (close ? "CLOSE" : "FAIL") : "EXACT");

    auto run = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < iters; ++i) encode(enc, pso, y);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / iters;
        }
    };
    if (iters) {
        run(pSafe, bSafe); run(pFixed, bFixed);
        const double s0 = run(pSafe, bSafe), f0 = run(pFixed, bFixed);
        const double f1 = run(pFixed, bFixed), s1 = run(pSafe, bSafe);
        printf("gpu us/iter: safe %.2f / %.2f | fixed %.2f / %.2f\n", s0, s1, f0, f1);
    }
    return close;
}

// Real-weight IQ4_XS GEMV comparison for the two fixed inner dimensions used
// by every 80B dense projection.  Both paths use the shipping NSG=2 geometry.
static bool caseIQ4GemvCmp(MtlCtx& c, const std::string& tensorName,
                           uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    const GgufTensor* t = g.find(tensorName);
    if (!t || t->type != GGML_IQ4_XS || t->ne[2] != 1) {
        fprintf(stderr, "iq4gemvcmp requires a 2D IQ4_XS tensor\n");
        return false;
    }
    const uint32_t K = (uint32_t)t->ne[0];
    const uint32_t M = (uint32_t)t->ne[1];
    if ((K != 2048u && K != 4096u) || M % 4u != 0u) {
        fprintf(stderr, "iq4gemvcmp requires K={2048,4096} and M%%4=0\n");
        return false;
    }
    const size_t wBytes = (size_t)M * ggmlRowBytes(t->type, K);
    printf("\n== iq4gemvcmp %s W[%u,%u] (W %.1f MiB) ==\n",
           tensorName.c_str(), M, K, (double)wBytes / (1 << 20));

    std::mt19937 rng(0x49344756u);
    std::normal_distribution<float> nd(0.f, 0.35f);
    std::vector<float> x(K);
    for (float& v : x) v = nd(rng);

    id<MTLBuffer> bW = createBuf(c, wBytes, t->data);
    id<MTLBuffer> bX = createBuf(c, (size_t)K * 4u, x.data());
    id<MTLBuffer> bSafe = createBuf(c, (size_t)M * 4u);
    id<MTLBuffer> bFixed = createBuf(c, (size_t)M * 4u);
    id<MTLComputePipelineState> pSafe = getPipe(c, "gemv_iq4_xs", "gemv_iq4_xs", 2);
    id<MTLComputePipelineState> pFixed = getPipe(
        c, "gemv_iq4_xs_fixed",
        K == 2048u ? "gemv_iq4_xs_k2048" : "gemv_iq4_xs_k4096", 2);
    const uint32_t mk[2] = {M, K};

    auto encode = [&](id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState> pso, id<MTLBuffer> y) {
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:y offset:0 atIndex:2];
        [enc setBytes:mk length:sizeof(mk) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M / 4u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    };
    auto once = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode(enc, pso, y);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
    };
    once(pSafe, bSafe);
    once(pFixed, bFixed);
    const uint32_t* ys = (const uint32_t*)bSafe.contents;
    const uint32_t* yf = (const uint32_t*)bFixed.contents;
    const float* fs = (const float*)ys;
    const float* ff = (const float*)yf;
    size_t mismatches = 0;
    double rms = 0.0, maxAbs = 0.0;
    for (uint32_t i = 0; i < M; ++i) rms += (double)fs[i] * fs[i];
    rms = std::sqrt(rms / M);
    for (uint32_t i = 0; i < M; ++i) {
        mismatches += ys[i] != yf[i];
        maxAbs = std::max(maxAbs, std::fabs((double)fs[i] - ff[i]));
    }
    const double scaled = maxAbs / std::max(1e-7, rms);
    const bool close = scaled <= 2e-6;
    printf("fixed-vs-safe: %zu/%u bit mismatches, max_abs %.3g, max_abs/rms %.3g -> %s\n",
           mismatches, M, maxAbs, scaled, mismatches ? (close ? "CLOSE" : "FAIL") : "EXACT");

    auto run = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < iters; ++i) encode(enc, pso, y);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / iters;
        }
    };
    if (iters) {
        run(pSafe, bSafe); run(pFixed, bFixed);
        const double s0 = run(pSafe, bSafe), f0 = run(pFixed, bFixed);
        const double f1 = run(pFixed, bFixed), s1 = run(pSafe, bSafe);
        printf("gpu us/iter: safe %.2f / %.2f | fixed %.2f / %.2f\n", s0, s1, f0, f1);
    }
    return close;
}

// Real Q6_K output-head comparison for fixed K=2048 and NSG sweeps.
static bool caseQ6GemvCmp(MtlCtx& c, uint32_t nsg, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    const GgufTensor* t = g.find("output.weight");
    if (!t || t->type != GGML_Q6_K || t->ne[2] != 1) {
        fprintf(stderr, "q6gemvcmp requires a 2D Q6_K output.weight\n");
        return false;
    }
    const uint32_t K = (uint32_t)t->ne[0];
    const uint32_t M = (uint32_t)t->ne[1];
    if (K != 2048u || (nsg != 1u && nsg != 2u && nsg != 4u && nsg != 8u) ||
        M % (2u * nsg) != 0u) {
        fprintf(stderr, "q6gemvcmp requires K=2048, NSG={1,2,4,8}, and full row tiles\n");
        return false;
    }
    const size_t wBytes = (size_t)M * ggmlRowBytes(t->type, K);
    printf("\n== q6gemvcmp output.weight W[%u,%u], NSG=%u (W %.1f MiB) ==\n",
           M, K, nsg, (double)wBytes / (1 << 20));

    std::mt19937 rng(0x51364756u);
    std::normal_distribution<float> nd(0.f, 0.35f);
    std::vector<float> x(K);
    for (float& v : x) v = nd(rng);

    id<MTLBuffer> bW = createBuf(c, wBytes, t->data);
    id<MTLBuffer> bX = createBuf(c, (size_t)K * 4u, x.data());
    id<MTLBuffer> bSafe = createBuf(c, (size_t)M * 4u);
    id<MTLBuffer> bFixed = createBuf(c, (size_t)M * 4u);
    id<MTLComputePipelineState> pSafe = getPipe(c, "gemv_q6_k", "gemv_q6_k", nsg);
    id<MTLComputePipelineState> pFixed =
        getPipe(c, "gemv_q6_k_fixed", "gemv_q6_k_k2048", nsg);
    const uint32_t mk[2] = {M, K};
    const uint32_t wgs = M / (2u * nsg);

    auto encode = [&](id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState> pso, id<MTLBuffer> y) {
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:y offset:0 atIndex:2];
        [enc setBytes:mk length:sizeof(mk) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(wgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32u * nsg, 1, 1)];
    };
    auto once = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode(enc, pso, y);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
    };
    once(pSafe, bSafe);
    once(pFixed, bFixed);
    const uint32_t* ys = (const uint32_t*)bSafe.contents;
    const uint32_t* yf = (const uint32_t*)bFixed.contents;
    const float* fs = (const float*)ys;
    const float* ff = (const float*)yf;
    size_t mismatches = 0;
    double rms = 0.0, maxAbs = 0.0;
    for (uint32_t i = 0; i < M; ++i) rms += (double)fs[i] * fs[i];
    rms = std::sqrt(rms / M);
    for (uint32_t i = 0; i < M; ++i) {
        mismatches += ys[i] != yf[i];
        maxAbs = std::max(maxAbs, std::fabs((double)fs[i] - ff[i]));
    }
    const double scaled = maxAbs / std::max(1e-7, rms);
    const bool close = scaled <= 2e-6;
    printf("fixed-vs-safe: %zu/%u bit mismatches, max_abs %.3g, max_abs/rms %.3g -> %s\n",
           mismatches, M, maxAbs, scaled, mismatches ? (close ? "CLOSE" : "FAIL") : "EXACT");

    auto run = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < iters; ++i) encode(enc, pso, y);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / iters;
        }
    };
    if (iters) {
        run(pSafe, bSafe); run(pFixed, bFixed);
        const double s0 = run(pSafe, bSafe), f0 = run(pFixed, bFixed);
        const double f1 = run(pFixed, bFixed), s1 = run(pSafe, bSafe);
        printf("gpu us/iter: safe %.1f / %.1f | fixed %.1f / %.1f\n", s0, s1, f0, f1);
    }
    return close;
}

// Real-weight IQ4_XS prefill GEMM A/B.  This avoids a synthetic quantizer and
// compares the full-tile specialization directly against the established safe
// kernel before timing a thermally bracketed control/candidate pair.
static bool caseIQ4Gemm(MtlCtx& c, const std::string& tensorName,
                        uint32_t N, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    const GgufTensor* t = g.find(tensorName);
    if (!t || t->type != GGML_IQ4_XS || t->ne[2] != 1) {
        fprintf(stderr, "iq4gemm requires a 2D IQ4_XS tensor\n");
        return false;
    }
    const uint32_t K = (uint32_t)t->ne[0];
    const uint32_t M = (uint32_t)t->ne[1];
    if (!N || K % 256u || M % 64u || N % 32u) {
        fprintf(stderr, "iq4gemm requires K%%256=M%%64=N%%32=0\n");
        return false;
    }
    const size_t wBytes = (size_t)M * ggmlRowBytes(t->type, K);
    const size_t xBytes = (size_t)N * K * 4;
    const size_t yBytes = (size_t)N * M * 4;
    printf("\n== iq4gemm %s W[%u,%u] X[%u,%u] (W %.1f MiB) ==\n",
           tensorName.c_str(), M, K, N, K, (double)wBytes / (1 << 20));

    std::vector<float> x((size_t)N * K);
    std::mt19937 rng(0x49344745u);
    std::normal_distribution<float> nd(0.f, 0.35f);
    for (float& v : x) v = nd(rng);

    id<MTLBuffer> bW = createBuf(c, wBytes, t->data);
    id<MTLBuffer> bX = createBuf(c, xBytes, x.data());
    id<MTLBuffer> bSafe = createBuf(c, yBytes);
    id<MTLBuffer> bAligned = createBuf(c, yBytes);
    id<MTLComputePipelineState> pSafe =
        getPipe(c, "gemm_iq4_xs", "gemm_iq4_xs_hp", 0);
    id<MTLComputePipelineState> pAligned =
        getPipe(c, "gemm_iq4_xs", "gemm_iq4_xs_hp_aligned", 0);
    struct { uint32_t M, K, N; } pc{M, K, N};

    auto encode = [&](id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState> pso, id<MTLBuffer> y) {
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:y offset:0 atIndex:2];
        [enc setBytes:&pc length:12 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M / 64u, 1, N / 32u)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };
    auto once = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode(enc, pso, y);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
    };
    once(pSafe, bSafe);
    once(pAligned, bAligned);
    const uint32_t* ys = (const uint32_t*)bSafe.contents;
    const uint32_t* ya = (const uint32_t*)bAligned.contents;
    size_t mismatches = 0;
    for (size_t i = 0; i < yBytes / 4; ++i) mismatches += ys[i] != ya[i];
    printf("aligned-vs-safe: %zu/%zu bit mismatches -> %s\n",
           mismatches, yBytes / 4, mismatches ? "FAIL" : "EXACT");

    auto run = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> y) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < iters; ++i) encode(enc, pso, y);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / iters;
        }
    };
    run(pSafe, bSafe); run(pAligned, bAligned);
    const double s0 = run(pSafe, bSafe), a0 = run(pAligned, bAligned);
    const double a1 = run(pAligned, bAligned), s1 = run(pSafe, bSafe);
    printf("gpu us/iter: safe %.1f / %.1f | aligned %.1f / %.1f\n", s0, s1, a0, a1);
    return mismatches == 0;
}

// Real Q6_K output-head comparison.  The shipping GEMV is the exact-logit
// reference and the tiled path targets verifier batches.
static bool caseHeadCmp(MtlCtx& c, uint32_t N, uint32_t iters) {
    Gguf g;
    if (!g.open(ggufPath())) return false;
    const GgufTensor* t = g.find("output.weight");
    if (!t || t->type != GGML_Q6_K || t->ne[2] != 1) {
        fprintf(stderr, "headcmp requires a 2D Q6_K output.weight\n");
        return false;
    }
    const uint32_t K = (uint32_t)t->ne[0];
    const uint32_t M = (uint32_t)t->ne[1];
    if (!N || K % 256u) {
        fprintf(stderr, "headcmp requires N > 0 and K divisible by 256\n");
        return false;
    }
    const size_t rowBytes = ggmlRowBytes(t->type, K);
    const size_t wBytes = (size_t)M * rowBytes;
    printf("\n== headcmp  Q6_K W[%u,%u], X[%u,%u] (W %.1f MiB) ==\n",
           M, K, N, K, (double)wBytes / (1 << 20));

    std::mt19937 rng(0x48454144u);
    std::normal_distribution<float> nd(0.f, 0.35f);
    std::vector<float> x((size_t)N * K);
    for (float& v : x) v = nd(rng);

    id<MTLComputePipelineState> pBase = getPipe(c, "gemv_q6_k", "gemv_q6_k", 2);
    id<MTLComputePipelineState> pGemm =
        getPipe(c, "head_q6", "head_q6_gemm_b8_f32", 0);
    const bool topF16 = getenv("QK_HEAD_F16") && atoi(getenv("QK_HEAD_F16")) != 0;
    const uint32_t topM = topF16 ? 64u : getenv("QK_HEAD_TOP_M64") ? 64u : 32u;
    const uint32_t topN = topF16 ? (N <= 8u ? 8u : N <= 16u ? 16u : 32u) : 8u;
    const char* topFn = topF16
        ? (topN == 8u ? "head_q6_gemm_b8_top1_f16" :
           topN == 16u ? "head_q6_gemm_b16_top1_f16" :
                         "head_q6_gemm_b32_top1_f16")
        : topM == 64u ? "head_q6_gemm_b8_top1_f32_m64"
                      : "head_q6_gemm_b8_top1_f32";
    id<MTLComputePipelineState> pTop = getPipe(c, "head_q6", topFn, 0);
    id<MTLComputePipelineState> pTopReduce =
        getPipe(c, "head_q6", "head_top1_reduce_batch", 0);
    id<MTLComputePipelineState> pAm1 = getPipe(c, "argmax", "argmax1", 0);
    id<MTLComputePipelineState> pAm2 = getPipe(c, "argmax", "argmax2", 0);

    id<MTLBuffer> bW = createBuf(c, wBytes, t->data);
    id<MTLBuffer> bX = createBuf(c, x.size() * sizeof(float), x.data());
    id<MTLBuffer> bRef = createBuf(c, (size_t)N * M * sizeof(float));
    id<MTLBuffer> bGemm = createBuf(c, (size_t)N * M * sizeof(float));
    id<MTLBuffer> bAV = createBuf(c, (size_t)N * 64 * sizeof(float));
    id<MTLBuffer> bAI = createBuf(c, (size_t)N * 64 * sizeof(uint32_t));
    id<MTLBuffer> bTok = createBuf(c, (size_t)N * sizeof(uint32_t));
    id<MTLBuffer> bRb = createBuf(c, sizeof(uint32_t));
    const uint32_t nTiles = (M + topM - 1u) / topM;
    const uint32_t topThreads = topM * 4u;
    id<MTLBuffer> bTV = createBuf(c, (size_t)N * nTiles * sizeof(float));
    id<MTLBuffer> bTI = createBuf(c, (size_t)N * nTiles * sizeof(uint32_t));
    id<MTLBuffer> bLast = createBuf(c, (size_t)M * sizeof(float));
    id<MTLBuffer> bTopTok = createBuf(c, (size_t)N * sizeof(uint32_t));

    struct HeadPC { uint32_t M, K, N, span; } pc{M, K, N, 0};
    struct Am1PC { uint32_t n, span; } pcAm1{M, 4096u};
    const uint32_t amWgs = (M + pcAm1.span - 1u) / pcAm1.span;
    struct Am2PC { uint32_t m, pos; } pcAm2{amWgs, 0u};
    const uint32_t mk[2] = {M, K};

    auto barrier = [](id<MTLComputeCommandEncoder> enc) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    auto encBase = [&](id<MTLComputeCommandEncoder> enc, bool argmax) {
        [enc setComputePipelineState:pBase];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:bRef offset:0 atIndex:2];
        [enc setBytes:mk length:sizeof(mk) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((M + 3u) / 4u, 1, N)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (!argmax) return;
        barrier(enc);
        [enc setComputePipelineState:pAm1];
        [enc setBuffer:bRef offset:0 atIndex:0];
        [enc setBuffer:bAV offset:0 atIndex:1];
        [enc setBuffer:bAI offset:0 atIndex:2];
        [enc setBytes:&pcAm1 length:sizeof(pcAm1) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(amWgs, 1, N)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier(enc);
        [enc setComputePipelineState:pAm2];
        [enc setBuffer:bAV offset:0 atIndex:0];
        [enc setBuffer:bAI offset:0 atIndex:1];
        [enc setBuffer:bTok offset:0 atIndex:2];
        [enc setBuffer:bRb offset:0 atIndex:3];
        [enc setBytes:&pcAm2 length:sizeof(pcAm2) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, N)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto encGemm = [&](id<MTLComputeCommandEncoder> enc, bool argmax) {
        [enc setComputePipelineState:pGemm];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:bGemm offset:0 atIndex:2];
        [enc setBytes:&pc length:sizeof(pc) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((M + 31u) / 32u, 1, (N + 7u) / 8u)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        if (!argmax) return;
        barrier(enc);
        [enc setComputePipelineState:pAm1];
        [enc setBuffer:bGemm offset:0 atIndex:0];
        [enc setBuffer:bAV offset:0 atIndex:1];
        [enc setBuffer:bAI offset:0 atIndex:2];
        [enc setBytes:&pcAm1 length:sizeof(pcAm1) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(amWgs, 1, N)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier(enc);
        [enc setComputePipelineState:pAm2];
        [enc setBuffer:bAV offset:0 atIndex:0];
        [enc setBuffer:bAI offset:0 atIndex:1];
        [enc setBuffer:bTok offset:0 atIndex:2];
        [enc setBuffer:bRb offset:0 atIndex:3];
        [enc setBytes:&pcAm2 length:sizeof(pcAm2) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, N)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto encTop = [&](id<MTLComputeCommandEncoder> enc) {
        struct { uint32_t M, K, N, tiles, materialize; } topPc{M, K, N, nTiles, 1u};
        [enc setComputePipelineState:pTop];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:bTV offset:0 atIndex:2];
        [enc setBuffer:bTI offset:0 atIndex:3];
        [enc setBuffer:bLast offset:0 atIndex:4];
        [enc setBytes:&topPc length:sizeof(topPc) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(nTiles, 1, (N + topN - 1u) / topN)
            threadsPerThreadgroup:MTLSizeMake(topThreads, 1, 1)];
        barrier(enc);
        struct { uint32_t tiles, N; } redPc{nTiles, N};
        [enc setComputePipelineState:pTopReduce];
        [enc setBuffer:bTV offset:0 atIndex:0];
        [enc setBuffer:bTI offset:0 atIndex:1];
        [enc setBuffer:bTopTok offset:0 atIndex:2];
        [enc setBytes:&redPc length:sizeof(redPc) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, N)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto submit = [&](auto&& encode, uint32_t reps) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < reps; ++i) {
                encode(enc);
                if (i + 1u < reps) barrier(enc);
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "headcmp Metal failure: %s\n",
                        cb.error.localizedDescription.UTF8String);
                return -1.0;
            }
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps;
        }
    };

    if (submit([&](auto enc) { encBase(enc, true); }, 1) < 0 ||
        submit([&](auto enc) { encGemm(enc, true); }, 1) < 0 ||
        submit([&](auto enc) { encTop(enc); }, 1) < 0) return false;
    const float* yr = (const float*)bRef.contents;
    const float* yg = (const float*)bGemm.contents;
    const uint32_t* gt = (const uint32_t*)bTok.contents;
    const uint32_t* tt = (const uint32_t*)bTopTok.contents;
    std::vector<uint32_t> refTok(N);
    double rms2 = 0.0, maxAbs = 0.0, maxRel = 0.0;
    for (size_t i = 0; i < (size_t)N * M; ++i) rms2 += (double)yr[i] * yr[i];
    const double rms = std::sqrt(rms2 / ((size_t)N * M));
    const double floor = std::max(1e-6, rms * 1e-4);
    uint32_t tokenBad = 0;
    for (uint32_t n = 0; n < N; ++n) {
        uint32_t ri = 0, gi = 0;
        for (uint32_t m = 1; m < M; ++m) {
            if (yr[(size_t)n * M + m] > yr[(size_t)n * M + ri]) ri = m;
            if (yg[(size_t)n * M + m] > yg[(size_t)n * M + gi]) gi = m;
        }
        refTok[n] = ri;
        const bool ok = ri == gi && ri == gt[n] && ri == tt[n];
        tokenBad += !ok;
        printf("  row %u: ref=%u gemm=%u gpu-argmax=%u fused=%u  %s\n",
               n, ri, gi, gt[n], tt[n], ok ? "EXACT" : "MISMATCH");
    }
    for (size_t i = 0; i < (size_t)N * M; ++i) {
        const double ae = std::fabs((double)yg[i] - yr[i]);
        maxAbs = std::max(maxAbs, ae);
        maxRel = std::max(maxRel, ae / std::max(floor, (double)std::fabs(yr[i])));
    }
    printf("  tiled logits: rms=%.6g max_abs=%.6g max_rel(floor %.3g)=%.6g\n",
           rms, maxAbs, floor, maxRel);
    double lastMax = 0.0;
    const float* last = (const float*)bLast.contents;
    for (uint32_t m = 0; m < M; ++m)
        lastMax = std::max(lastMax,
            std::fabs((double)last[m] - yg[(size_t)(N - 1u) * M + m]));
    const bool lastClose = topF16 ? lastMax <= 1e-3 : lastMax == 0.0;
    printf("  fused materialized last row max_abs=%.6g (%s precision)\n",
           lastMax, topF16 ? "f16-input" : "exact-f32");

    // Deliberate all-logit tie: both reductions must choose vocabulary id 0.
    memset(bX.contents, 0, x.size() * sizeof(float));
    if (submit([&](auto enc) { encBase(enc, true); }, 1) < 0) return false;
    std::vector<uint32_t> tieRef(N);
    memcpy(tieRef.data(), bTok.contents, (size_t)N * sizeof(uint32_t));
    if (submit([&](auto enc) { encGemm(enc, true); }, 1) < 0) return false;
    const uint32_t* tieGot = (const uint32_t*)bTok.contents;
    std::vector<uint32_t> tieGemm(N);
    memcpy(tieGemm.data(), tieGot, (size_t)N * sizeof(uint32_t));
    if (submit([&](auto enc) { encTop(enc); }, 1) < 0) return false;
    const uint32_t* tieTop = (const uint32_t*)bTopTok.contents;
    uint32_t tieBad = 0;
    for (uint32_t n = 0; n < N; ++n)
        tieBad += tieRef[n] != 0u || tieGemm[n] != 0u || tieTop[n] != 0u;
    memcpy(bX.contents, x.data(), x.size() * sizeof(float));
    printf("  all-zero tie: %u/%u chose lower id 0\n", N - tieBad, N);

    const bool logitsClose = maxAbs <= std::max(1e-5, rms * 1e-4);
    const bool pass = tokenBad == 0 && tieBad == 0 && logitsClose && lastClose;
    printf("correctness: tiled argmax %u/%u exact, logits %s -> %s\n",
           N - tokenBad, N, logitsClose ? "CLOSE" : "DRIFT", pass ? "PASS" : "FAIL");
    if (pass && iters) {
        auto bench = [&](const char* name, auto&& encode) {
            submit(encode, 1);
            const double us = submit(encode, iters);
            printf("  %-27s %8.2f us  (%7.1f tok/s aggregate)\n",
                   name, us, 1e6 * N / us);
        };
        bench("shipping GEMV", [&](auto enc) { encBase(enc, false); });
        bench("shipping GEMV + argmax", [&](auto enc) { encBase(enc, true); });
        bench("32x8 f32 GEMM", [&](auto enc) { encGemm(enc, false); });
        bench("32x8 f32 GEMM + argmax", [&](auto enc) { encGemm(enc, true); });
        char topName[48];
        snprintf(topName, sizeof(topName), "%ux%u %s candidates", topM, topN,
                 topF16 ? "f16" : "f32");
        bench(topName, [&](auto enc) { encTop(enc); });
    }
    return pass;
}

// Synthetic prefill flash-attention comparison. The shipping Q16/K64/S8
// kernel is the independent reference for the paired exact and SIMD-softmax
// twins retained after the full geometry sweep.
static bool caseFaCmp(MtlCtx& c, uint32_t N, uint32_t base, uint32_t iters) {
    constexpr uint32_t dh = 256, hQ = 16, hKV = 2;
    const bool kvQ8 = getenv("QK_KV_Q8") && atoi(getenv("QK_KV_Q8")) != 0;
    const bool kvF16 = !kvQ8 && getenv("QK_KV_F16") &&
                       atoi(getenv("QK_KV_F16")) != 0;
    const uint32_t kvBytes = kvQ8 ? 1u : kvF16 ? 2u : 4u;
    if (!N || N > 1024 || base + N > 8192) {
        fprintf(stderr, "facmp requires 1 <= N <= 1024 and base+N <= 8192\n");
        return false;
    }
    const uint32_t tmax = std::max(2048u, base + N + 64u);
    std::mt19937 rng(0x4641544eu + N * 17u + base);
    std::normal_distribution<float> qd(0.f, 0.20f), vd(0.f, 0.35f);
    const uint32_t qcap = (N + 15u) / 16u * 16u;
    std::vector<float> qhat((size_t)qcap * hQ * dh);
    std::vector<float> qfull((size_t)qcap * hQ * 2u * dh);
    std::vector<float> kc((size_t)hKV * tmax * dh);
    std::vector<float> vc((size_t)hKV * tmax * dh);
    for (float& v : qhat) v = qd(rng);
    for (float& v : qfull) v = qd(rng);
    for (float& v : kc) v = qd(rng);
    for (float& v : vc) v = vd(rng);

    std::vector<uint16_t> kc16, vc16;
    std::vector<uint8_t> kc8, vc8;
    if (kvF16) {
        kc16.resize(kc.size()); vc16.resize(vc.size());
        for (size_t i = 0; i < kc.size(); ++i) kc16[i] = qk_f32_to_f16(kc[i]);
        for (size_t i = 0; i < vc.size(); ++i) vc16[i] = qk_f32_to_f16(vc[i]);
    }
    auto quantQ8Rows = [&](const std::vector<float>& src, std::vector<uint8_t>& dst) {
        constexpr size_t rowBytes = 272u;
        const size_t rows = src.size() / dh;
        dst.assign(rows * rowBytes, 0u);
        for (size_t r = 0; r < rows; ++r) {
            float amax = 0.0f;
            for (uint32_t j = 0; j < dh; ++j)
                amax = std::max(amax, std::fabs(src[r * dh + j]));
            const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
            memcpy(dst.data() + r * rowBytes, &d, sizeof d);
            int8_t* q = (int8_t*)(dst.data() + r * rowBytes + 16u);
            for (uint32_t j = 0; j < dh; ++j)
                q[j] = d > 0.0f ? (int8_t)std::clamp(
                    (int)std::lrint(src[r * dh + j] / d), -127, 127) : 0;
        }
    };
    if (kvQ8) { quantQ8Rows(kc, kc8); quantQ8Rows(vc, vc8); }

    id<MTLBuffer> bQ = createBuf(c, qhat.size() * sizeof(float), qhat.data());
    const size_t kBytes = kvQ8 ? kc8.size() : kc.size() * kvBytes;
    const size_t vBytes = kvQ8 ? vc8.size() : vc.size() * kvBytes;
    const void* kData = kvQ8 ? (const void*)kc8.data()
        : kvF16 ? (const void*)kc16.data() : (const void*)kc.data();
    const void* vData = kvQ8 ? (const void*)vc8.data()
        : kvF16 ? (const void*)vc16.data() : (const void*)vc.data();
    id<MTLBuffer> bK = createBuf(c, kBytes, kData);
    id<MTLBuffer> bV = createBuf(c, vBytes, vData);
    id<MTLBuffer> bKF32 = kvQ8
        ? createBuf(c, kc.size() * sizeof(float), kc.data()) : bK;
    id<MTLBuffer> bVF32 = kvQ8
        ? createBuf(c, vc.size() * sizeof(float), vc.data()) : bV;
    id<MTLBuffer> bQf = createBuf(c, qfull.size() * sizeof(float), qfull.data());
    const size_t outN = (size_t)N * hQ * dh;
    id<MTLBuffer> bRef = createBuf(c, outN * sizeof(float));
    id<MTLBuffer> bF32Ref = kvQ8 ? createBuf(c, outN * sizeof(float)) : nil;
    id<MTLBuffer> bOut = createBuf(c, outN * sizeof(float));
    struct FaPC {
        uint32_t tmax, dh_, nRot, hQ_, hKV_;
        float eps, freqBase;
        uint32_t base_, Tn, qbase;
    } pc{tmax, dh, 64u, hQ, hKV, 1e-6f, 1000000.f, base, N, 0u};

    struct Variant {
        const char* name; uint32_t qt, kb, ns, hg;
        id<MTLComputePipelineState> p;
    };
    std::vector<Variant> vars;
    if (kvQ8) {
        vars.push_back({"gqa2_q8_k64_s16_q8", 8, 64, 16, 2,
            getKvPipe(c, "fa_gqa", "fa_attn_batch_gqa2_q8", kvBytes)});
    } else if (kvF16) {
        vars.push_back({"gqa2_q8_k64_s16_f16", 8, 64, 16, 2,
            getKvPipe(c, "fa_gqa", "fa_attn_batch_gqa2_f16", kvBytes)});
    } else {
        vars.push_back({"gqa2_q8_k64_s16_exact", 8, 64, 16, 2,
            getKvPipe(c, "fa_gqa", "fa_attn_batch_gqa2_exact", kvBytes)});
        vars.push_back({"gqa2_q8_k64_s16_simd", 8, 64, 16, 2,
            getKvPipe(c, "fa_gqa", "fa_attn_batch_gqa2", kvBytes)});
    }
    id<MTLComputePipelineState> pRef = (kvF16 || kvQ8)
        ? getKvPipe(c, "fa_batch", "fa_attn_batch", kvBytes)
        : getKvPipe(c, "fa_batch", "fa_attn_batch_mma", kvBytes);
    id<MTLComputePipelineState> pF32Ref = kvQ8
        ? getKvPipe(c, "fa_batch", "fa_attn_batch", 4u) : pRef;
    const uint32_t refQt = (kvF16 || kvQ8) ? 1u : 16u;

    auto encode = [&](id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState> p, id<MTLBuffer> out,
                      uint32_t qt, uint32_t ns, uint32_t hg) {
        [enc setComputePipelineState:p];
        [enc setBuffer:bQ offset:0 atIndex:0];
        [enc setBuffer:bK offset:0 atIndex:1];
        [enc setBuffer:bV offset:0 atIndex:2];
        [enc setBuffer:bQf offset:0 atIndex:3];
        [enc setBuffer:out offset:0 atIndex:4];
        [enc setBytes:&pc length:sizeof(pc) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(hQ / hg, 1, (N + qt - 1u) / qt)
            threadsPerThreadgroup:MTLSizeMake(ns * 32u, 1, 1)];
    };
    auto submit = [&](id<MTLComputePipelineState> p, id<MTLBuffer> out,
                      uint32_t qt, uint32_t ns, uint32_t hg, uint32_t reps) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < reps; ++i) {
                encode(enc, p, out, qt, ns, hg);
                if (i + 1u < reps)
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "facmp Metal failure: %s\n",
                        cb.error.localizedDescription.UTF8String);
                return -1.0;
            }
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps;
        }
    };

    if (kvQ8) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pF32Ref];
            [enc setBuffer:bQ offset:0 atIndex:0];
            [enc setBuffer:bKF32 offset:0 atIndex:1];
            [enc setBuffer:bVF32 offset:0 atIndex:2];
            [enc setBuffer:bQf offset:0 atIndex:3];
            [enc setBuffer:bF32Ref offset:0 atIndex:4];
            [enc setBytes:&pc length:sizeof(pc) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(hQ, 1, N)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) return false;
        }
    }
    if (submit(pRef, bRef, refQt, 8u, 1u, 1u) < 0) return false;
    const float* ref = (const float*)bRef.contents;
    double rms2 = 0.0;
    for (size_t i = 0; i < outN; ++i) rms2 += (double)ref[i] * ref[i];
    const double rms = std::sqrt(rms2 / outN);
    const double floor = std::max(1e-6, rms * 1e-4);
    printf("\n== facmp N=%u base=%u  Q[%u,%u,%u] KV[%u,%u,%u] %s ==\n",
           N, base, N, hQ, dh, hKV, tmax, dh,
           kvQ8 ? "q8-row" : kvF16 ? "f16" : "f32");
    if (kvQ8) {
        const float* f32ref = (const float*)bF32Ref.contents;
        double qMaxAbs = 0.0, qD2 = 0.0, qR2 = 0.0;
        for (size_t i = 0; i < outN; ++i) {
            const double d = (double)ref[i] - f32ref[i];
            qMaxAbs = std::max(qMaxAbs, std::fabs(d));
            qD2 += d * d; qR2 += (double)f32ref[i] * f32ref[i];
        }
        printf("  q8-vs-f32: max_abs %.6g | rel_rms %.6g\n",
               qMaxAbs, std::sqrt(qD2 / qR2));
    }
    printf("  %-24s %10s %11s %11s %8s\n",
           "geometry", "gpu_us", "max_abs", "max_rel", "result");
    const double refUs = iters ? submit(pRef, bRef, refQt, 8u, 1u, iters) : 0.0;
    printf("  %-24s %10.2f %11s %11s %8s\n",
           kvQ8 ? "scalar_q8 (ref)" : kvF16 ? "scalar_f16 (ref)" : "q16_k64_s8 (ref)",
           refUs, "-", "-", "REF");
    bool pass = true;
    for (const Variant& v : vars) {
        if (submit(v.p, bOut, v.qt, v.ns, v.hg, 1u) < 0) return false;
        const float* got = (const float*)bOut.contents;
        double maxAbs = 0.0, maxRel = 0.0;
        size_t nDiff = 0;
        for (size_t i = 0; i < outN; ++i) {
            const double ae = std::fabs((double)got[i] - ref[i]);
            maxAbs = std::max(maxAbs, ae);
            maxRel = std::max(maxRel, ae / std::max(floor, (double)std::fabs(ref[i])));
            nDiff += got[i] != ref[i];
        }
        const bool ok = maxAbs <= std::max(2e-5, rms * 2e-4);
        pass &= ok;
        const double us = iters ? submit(v.p, bOut, v.qt, v.ns, v.hg, iters) : 0.0;
        char label[40];
        snprintf(label, sizeof label, "%s%s", v.name, nDiff ? "" : " bit-exact");
        printf("  %-24s %10.2f %11.4g %11.4g %8s\n",
               label, us, maxAbs, maxRel, ok ? "PASS" : "FAIL");
    }
    return pass;
}

// Synthetic serving/decode attention comparison.  Positions are distributed
// linearly from minPos through maxPos so one invocation covers both homogeneous
// (min=max) and maximally heterogeneous slot batches.  This deliberately
// exercises the real per-slot KV strides and the split partial layout.
static bool caseFaSrvCmp(MtlCtx& c, uint32_t slots, uint32_t maxPos,
                         uint32_t minPos, uint32_t iters) {
    constexpr uint32_t dh = 256, hQ = 16, hKV = 2, CK = 256;
    const bool kvQ8 = getenv("QK_KV_Q8") && atoi(getenv("QK_KV_Q8")) != 0;
    const bool kvF16 = !kvQ8 && getenv("QK_KV_F16") &&
                       atoi(getenv("QK_KV_F16")) != 0;
    const uint32_t kvBytes = kvQ8 ? 1u : kvF16 ? 2u : 4u;
    if (!slots || slots > 16 || minPos > maxPos || maxPos >= 49152) {
        fprintf(stderr,
                "fasrvcmp requires 1<=slots<=16 and 0<=min-pos<=max-pos<49152\n");
        return false;
    }
    const uint32_t tmax = ((maxPos + 1u + CK - 1u) / CK) * CK;
    const uint32_t maxChunks = (tmax + CK - 1u) / CK;
    std::vector<uint32_t> pos(slots);
    for (uint32_t s = 0; s < slots; ++s)
        pos[s] = slots == 1 ? maxPos
            : minPos + (uint64_t)(maxPos - minPos) * s / (slots - 1u);
    if (slots > 1 && getenv("QK_FA_SKEW")) {
        uint32_t nLong = std::min(
            slots, (uint32_t)std::max(1, atoi(getenv("QK_FA_SKEW"))));
        std::fill(pos.begin(), pos.end(), minPos);
        std::fill(pos.end() - nLong, pos.end(), maxPos);
    }

    std::mt19937 rng(0x46535256u + slots * 17u + maxPos * 3u + minPos);
    std::normal_distribution<float> qd(0.f, 0.20f), vd(0.f, 0.35f);
    std::vector<float> qhat((size_t)slots * hQ * dh);
    std::vector<float> qfull((size_t)slots * hQ * 2u * dh);
    std::vector<float> kc((size_t)slots * hKV * tmax * dh);
    std::vector<float> vc((size_t)slots * hKV * tmax * dh);
    for (float& v : qhat) v = qd(rng);
    for (float& v : qfull) v = qd(rng);
    for (float& v : kc) v = qd(rng);
    for (float& v : vc) v = vd(rng);

    std::vector<uint16_t> kc16, vc16;
    std::vector<uint8_t> kc8, vc8;
    if (kvF16) {
        kc16.resize(kc.size()); vc16.resize(vc.size());
        for (size_t i = 0; i < kc.size(); ++i) kc16[i] = qk_f32_to_f16(kc[i]);
        for (size_t i = 0; i < vc.size(); ++i) vc16[i] = qk_f32_to_f16(vc[i]);
    }
    auto quantQ8Rows = [&](const std::vector<float>& src, std::vector<uint8_t>& dst) {
        constexpr size_t rowBytes = 272u;
        const size_t rows = src.size() / dh;
        dst.assign(rows * rowBytes, 0u);
        for (size_t r = 0; r < rows; ++r) {
            float amax = 0.0f;
            for (uint32_t j = 0; j < dh; ++j)
                amax = std::max(amax, std::fabs(src[r * dh + j]));
            const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
            memcpy(dst.data() + r * rowBytes, &d, sizeof d);
            int8_t* q = (int8_t*)(dst.data() + r * rowBytes + 16u);
            for (uint32_t j = 0; j < dh; ++j)
                q[j] = d > 0.0f ? (int8_t)std::clamp(
                    (int)std::lrint(src[r * dh + j] / d), -127, 127) : 0;
        }
    };
    if (kvQ8) {
        quantQ8Rows(kc, kc8);
        quantQ8Rows(vc, vc8);
    }

    id<MTLBuffer> bQ = createBuf(c, qhat.size() * sizeof(float), qhat.data());
    const size_t kBytes = kvQ8 ? kc8.size() : kc.size() * kvBytes;
    const size_t vBytes = kvQ8 ? vc8.size() : vc.size() * kvBytes;
    const void* kData = kvQ8 ? (const void*)kc8.data()
        : kvF16 ? (const void*)kc16.data() : (const void*)kc.data();
    const void* vData = kvQ8 ? (const void*)vc8.data()
        : kvF16 ? (const void*)vc16.data() : (const void*)vc.data();
    id<MTLBuffer> bK = createBuf(c, kBytes, kData);
    id<MTLBuffer> bV = createBuf(c, vBytes, vData);
    id<MTLBuffer> bKF32 = kvQ8
        ? createBuf(c, kc.size() * sizeof(float), kc.data()) : bK;
    id<MTLBuffer> bVF32 = kvQ8
        ? createBuf(c, vc.size() * sizeof(float), vc.data()) : bV;
    id<MTLBuffer> bQf = createBuf(c, qfull.size() * sizeof(float), qfull.data());
    id<MTLBuffer> bPos = createBuf(c, pos.size() * sizeof(uint32_t), pos.data());
    const size_t outN = (size_t)slots * hQ * dh;
    id<MTLBuffer> bRef = createBuf(c, outN * sizeof(float));
    id<MTLBuffer> bF32Ref = kvQ8 ? createBuf(c, outN * sizeof(float)) : nil;
    id<MTLBuffer> bOut = createBuf(c, outN * sizeof(float));
    id<MTLBuffer> bPartial = createBuf(
        c, (size_t)slots * hQ * maxChunks * (dh + 2u) * sizeof(float));

    struct FaPC {
        uint32_t pos, tmax, dh_, nRot, hQ_, hKV_;
        float eps, freqBase;
    } pc{0u, tmax, dh, 64u, hQ, hKV, 1e-6f, 1000000.f};
    struct FaSplitPC {
        uint32_t pos, tmax, dh_, nRot, hQ_, hKV_;
        float eps, freqBase;
        uint32_t maxChunks;
    } spc{0u, tmax, dh, 64u, hQ, hKV, 1e-6f, 1000000.f, maxChunks};

    id<MTLComputePipelineState> pScalar =
        getKvPipe(c, "fa_srv", "fa_attn_srv", kvBytes);
    id<MTLComputePipelineState> pScalarF32 = kvQ8
        ? getKvPipe(c, "fa_srv", "fa_attn_srv", 4u) : pScalar;
    id<MTLComputePipelineState> pSplit =
        getKvPipe(c, "fa_srv", kvQ8 ? "fa_attn_srv_split_q8" : "fa_attn_srv_split",
                kvBytes);
    id<MTLComputePipelineState> pReduce =
        getPipe(c, "fa_srv", "fa_attn_srv_reduce", 0);
    id<MTLComputePipelineState> pCompact =
        getKvPipe(c, "fa_srv", "fa_attn_srv_split_compact", kvBytes);
    id<MTLComputePipelineState> pMulti =
        getKvPipe(c, "fa_srv", "fa_attn_srv_split_multi", kvBytes);
    struct SplitVariant {
        std::string name;
        id<MTLComputePipelineState> split;
        id<MTLBuffer> work;
        uint32_t workN;
        uint32_t headGroup;
    };
    std::vector<SplitVariant> splitVars{
        {"split256", pSplit, nil, 0u, 1u}};
    if (kvQ8)
        splitVars.push_back({"q8-gqa2",
            getKvPipe(c, "fa_srv", "fa_attn_srv_split_q8_gqa2", kvBytes),
            nil, 0u, 2u});
    if (kvQ8)
        splitVars.push_back({"q8-gqa4",
            getKvPipe(c, "fa_srv", "fa_attn_srv_split_q8_gqa4", kvBytes),
            nil, 0u, 4u});
    if (kvQ8)
        splitVars.push_back({"q8-gqa8",
            getKvPipe(c, "fa_srv", "fa_attn_srv_split_q8_gqa8", kvBytes),
            nil, 0u, 8u});
    if (kvF16)
        splitVars.push_back({"f16-gqa2",
            getKvPipe(c, "fa_srv", "fa_attn_srv_split_f16_gqa2", kvBytes),
            nil, 0u, 2u});
    if (kvF16)
        splitVars.push_back({"f16-gqa4",
            getKvPipe(c, "fa_srv", "fa_attn_srv_split_f16_gqa4", kvBytes),
            nil, 0u, 4u});
    if (kvF16)
        splitVars.push_back({"f16-gqa8",
            getKvPipe(c, "fa_srv", "fa_attn_srv_split_f16_gqa8", kvBytes),
            nil, 0u, 8u});
    if (!kvF16 && !kvQ8)
        splitVars.push_back({"f32-gqa2",
            getKvPipe(c, "fa_srv", "fa_attn_srv_split_f32_gqa2", kvBytes),
            nil, 0u, 2u});
    if (!kvF16 && !kvQ8)
        splitVars.push_back({"f32-gqa4",
            getKvPipe(c, "fa_srv", "fa_attn_srv_split_f32_gqa4", kvBytes),
            nil, 0u, 4u});
    if (!kvF16 && !kvQ8)
        splitVars.push_back({"f32-gqa8",
            getKvPipe(c, "fa_srv", "fa_attn_srv_split_f32_gqa8", kvBytes),
            nil, 0u, 8u});
    std::vector<uint32_t> compactWork;
    for (uint32_t s = 0; s < slots; ++s) {
        const uint32_t chunks = (pos[s] + 1u + CK - 1u) / CK;
        for (uint32_t c = 0; c < chunks; ++c)
            compactWork.push_back((s << 16u) | c);
    }
    splitVars.push_back({"compact", pCompact,
        createBuf(c, compactWork.size() * sizeof(uint32_t), compactWork.data()),
        (uint32_t)compactWork.size(), 1u});
    for (uint32_t fan : {1u, 4u, 8u, 16u, 32u, 64u}) {
        std::vector<uint32_t> balancedWork;
        uint32_t nBalanced = 0;
        for (uint32_t s = 0; s < slots; ++s) {
            const uint32_t chunks = (pos[s] + 1u + CK - 1u) / CK;
            const uint32_t ng = fan == 1u ? chunks
                : std::max(1u, (chunks + fan / 2u) / fan);
            for (uint32_t g = 0; g < ng; ++g) {
                const uint32_t first = (uint64_t)g * chunks / ng;
                const uint32_t end = (uint64_t)(g + 1u) * chunks / ng;
                balancedWork.push_back((s << 16u) | g);
                balancedWork.push_back((first << 16u) | (end - first));
                ++nBalanced;
            }
        }
        char name[24];
        snprintf(name, sizeof(name), "multi%u", fan);
        id<MTLBuffer> bBalancedWork = createBuf(
            c, balancedWork.size() * sizeof(uint32_t), balancedWork.data());
        splitVars.push_back({name, pMulti, bBalancedWork, nBalanced, 1u});
    }

    auto bind = [](id<MTLComputeCommandEncoder> enc,
                   id<MTLComputePipelineState> p,
                   std::initializer_list<id<MTLBuffer>> bufs,
                   const void* bytes, size_t size) {
        [enc setComputePipelineState:p];
        uint32_t i = 0;
        for (id<MTLBuffer> b : bufs) [enc setBuffer:b offset:0 atIndex:i++];
        [enc setBytes:bytes length:size atIndex:i];
    };
    auto scalar = [&](id<MTLComputeCommandEncoder> enc, id<MTLBuffer> out) {
        bind(enc, pScalar, {bQ, bK, bV, bQf, out, bPos}, &pc, sizeof(pc));
        [enc dispatchThreadgroups:MTLSizeMake(hQ, 1, slots)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto scalarF32 = [&](id<MTLComputeCommandEncoder> enc, id<MTLBuffer> out) {
        bind(enc, pScalarF32, {bQ, bKF32, bVF32, bQf, out, bPos}, &pc, sizeof(pc));
        [enc dispatchThreadgroups:MTLSizeMake(hQ, 1, slots)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto split = [&](id<MTLComputeCommandEncoder> enc, id<MTLBuffer> out,
                     const SplitVariant& v) {
        if (v.work) {
            bind(enc, v.split, {bQ, bK, bV, bPartial, bPos, v.work},
                 &spc, sizeof(spc));
            [enc dispatchThreadgroups:MTLSizeMake(hQ / v.headGroup, v.workN, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            bind(enc, v.split, {bQ, bK, bV, bPartial, bPos}, &spc, sizeof(spc));
            [enc dispatchThreadgroups:MTLSizeMake(hQ / v.headGroup, maxChunks, slots)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        bind(enc, pReduce, {bPartial, bQf, out, bPos}, &spc, sizeof(spc));
        [enc dispatchThreadgroups:MTLSizeMake(hQ, 1, slots)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto submit = [&](auto&& encode, uint32_t reps) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = getenv("QK_FA_CONCURRENT")
                ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                : [cb computeCommandEncoder];
            for (uint32_t i = 0; i < reps; ++i) {
                encode(enc);
                if (i + 1u < reps)
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "fasrvcmp Metal failure: %s\n",
                        cb.error.localizedDescription.UTF8String);
                return -1.0;
            }
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps;
        }
    };

    if (kvQ8 && submit([&](auto enc) { scalarF32(enc, bF32Ref); }, 1) < 0)
        return false;
    if (submit([&](auto enc) { scalar(enc, bRef); }, 1) < 0) return false;
    const float* ref = (const float*)bRef.contents;
    double rms2 = 0.0;
    for (size_t i = 0; i < outN; ++i) rms2 += (double)ref[i] * ref[i];
    const double rms = std::sqrt(rms2 / outN);
    const double floor = std::max(1e-6, rms * 1e-4);
    uint64_t liveChunks = 0;
    for (uint32_t p : pos) liveChunks += (p + 1u + CK - 1u) / CK;
    const uint64_t launchedChunks = (uint64_t)slots * maxChunks;
    const double scalarUs = iters
        ? submit([&](auto enc) { scalar(enc, bRef); }, iters) : 0.0;
    if (submit([&](auto enc) { split(enc, bOut, splitVars[0]); }, 1) < 0)
        return false;
    std::vector<float> splitRef(outN);
    memcpy(splitRef.data(), bOut.contents, outN * sizeof(float));
    printf("\n== fasrvcmp slots=%u pos=%u..%u tmax=%u %s KV "
           "chunks=%llu/%llu live ==\n",
           slots, minPos, maxPos, tmax, kvQ8 ? "q8-row" : kvF16 ? "f16" : "f32",
           (unsigned long long)liveChunks, (unsigned long long)launchedChunks);
    if (kvQ8) {
        const float* f32ref = (const float*)bF32Ref.contents;
        double qMaxAbs = 0.0, qRelRmsNum = 0.0, qRelRmsDen = 0.0;
        for (size_t i = 0; i < outN; ++i) {
            const double d = (double)ref[i] - f32ref[i];
            qMaxAbs = std::max(qMaxAbs, std::fabs(d));
            qRelRmsNum += d * d;
            qRelRmsDen += (double)f32ref[i] * f32ref[i];
        }
        printf("  q8-vs-f32: max_abs %.6g | rel_rms %.6g | storage %.4f B/value\n",
               qMaxAbs, std::sqrt(qRelRmsNum / qRelRmsDen), 272.0 / 256.0);
    }
    printf("  %-13s %10s %9s %10s %10s %11s %11s %8s\n",
           "schedule", "gpu_us", "speedup", "diff", "split_diff",
           "max_abs", "max_rel", "result");
    printf("  %-13s %10.2f %9s %10s %10s %11s %11s %8s\n",
           "scalar", scalarUs, "1.000x", "-", "-", "-", "-", "REF");
    bool pass = true;
    for (const SplitVariant& v : splitVars) {
        if (submit([&](auto enc) { split(enc, bOut, v); }, 1) < 0) return false;
        const double us = iters
            ? submit([&](auto enc) { split(enc, bOut, v); }, iters) : 0.0;
        const float* got = (const float*)bOut.contents;
        double maxAbs = 0.0, maxRel = 0.0;
        uint64_t nDiff = 0, nSplitDiff = 0;
        for (size_t i = 0; i < outN; ++i) {
            const double ae = std::fabs((double)got[i] - ref[i]);
            maxAbs = std::max(maxAbs, ae);
            maxRel = std::max(maxRel,
                ae / std::max(floor, (double)std::fabs(ref[i])));
            nDiff += got[i] != ref[i];
            nSplitDiff += got[i] != splitRef[i];
        }
        const bool ok = maxAbs <= std::max(2e-5, rms * 2e-4);
        pass &= ok;
        printf("  %-13s %10.2f %8.3fx %10llu %10llu %11.4g %11.4g %8s\n",
               v.name.c_str(), us, us > 0 ? scalarUs / us : 0.0,
               (unsigned long long)nDiff, (unsigned long long)nSplitDiff,
               maxAbs, maxRel, ok ? "PASS" : "FAIL");
    }
    printf("  positions:");
    for (uint32_t p : pos) printf(" %u", p);
    printf("\n");
    return pass;
}

// Decode DeltaNet megakernel schedule comparison.  This uses nonzero conv and
// matrix state, checks every mutated/output buffer bit-for-bit, and exercises
// both head mappings used by the 35B (modulo) and 80B (consecutive) models.
static bool caseDnStep(MtlCtx& c, uint32_t slots, uint32_t iters) {
    constexpr uint32_t dS = 128, hK = 16, hV = 32;
    constexpr uint32_t chQkv = (2 * hK + hV) * dS;
    if (!slots || slots > 16) return false;
    std::mt19937 rng(0x44535445u + slots);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> qkv((size_t)slots * chQkv), conv0((size_t)slots * chQkv * 3u),
        ker((size_t)chQkv * 4u), gb((size_t)slots * 2u * hV),
        state0((size_t)slots * hV * dS * dS), z((size_t)slots * hV * dS), sn(dS);
    for (float& v : qkv) v = 0.2f * nd(rng);
    for (float& v : conv0) v = 0.1f * nd(rng);
    for (float& v : ker) v = 0.2f * nd(rng);
    for (float& v : state0) v = 0.03f * nd(rng);
    for (float& v : z) v = 0.3f * nd(rng);
    for (float& v : sn) v = 1.0f + 0.1f * nd(rng);
    for (uint32_t r = 0; r < slots; ++r)
        for (uint32_t h = 0; h < hV; ++h) {
            gb[(size_t)r * 2u * hV + h] = -0.1f - 0.7f * std::fabs(nd(rng));
            const float x = nd(rng);
            gb[(size_t)r * 2u * hV + hV + h] = 1.0f / (1.0f + std::exp(-x));
        }

    id<MTLBuffer> bQ = createBuf(c, qkv.size() * 4u, qkv.data());
    id<MTLBuffer> bQc = createBuf(c, qkv.size() * 4u, qkv.data());
    id<MTLBuffer> bK = createBuf(c, ker.size() * 4u, ker.data());
    id<MTLBuffer> bG = createBuf(c, gb.size() * 4u, gb.data());
    id<MTLBuffer> bZ = createBuf(c, z.size() * 4u, z.data());
    id<MTLBuffer> bN = createBuf(c, sn.size() * 4u, sn.data());
    id<MTLBuffer> bCr = createBuf(c, conv0.size() * 4u);
    id<MTLBuffer> bCc = createBuf(c, conv0.size() * 4u);
    id<MTLBuffer> bSr = createBuf(c, state0.size() * 4u);
    id<MTLBuffer> bSc = createBuf(c, state0.size() * 4u);
    id<MTLBuffer> bOr = createBuf(c, z.size() * 4u);
    id<MTLBuffer> bOc = createBuf(c, z.size() * 4u);
    std::vector<float> refConv(conv0.size()), refState(state0.size()), refOut(z.size());
    id<MTLComputePipelineState> pRef = getPipe(c, "dn_step", "dn_step", 0);
    struct Variant { const char* name; id<MTLComputePipelineState> p; };
    const Variant vars[] = {
        {"cache8",  getPipe(c, "dn_step", "dn_step_res8", 0)},
    };
    id<MTLComputePipelineState> pPrep = getPipe(c, "dn_step", "dn_step_prep", 0);
    const Variant splitVars[] = {
        {"split0",  getPipe(c, "dn_step", "dn_step_state", 0)},
        {"split8",  getPipe(c, "dn_step", "dn_step_state_res8", 0)},
        {"rowSIMD", getPipe(c, "dn_step", "dn_step_state_rowsimd", 0)},
    };
    struct { uint32_t d, hk, hv; float e; uint32_t kd; } pc{dS, hK, hV, 1e-6f, 0};

    auto reset = [&](id<MTLBuffer> bc, id<MTLBuffer> bs) {
        memcpy(bc.contents, conv0.data(), conv0.size() * 4u);
        memcpy(bs.contents, state0.data(), state0.size() * 4u);
    };
    auto submit = [&](id<MTLComputePipelineState> p, id<MTLBuffer> bc,
                      id<MTLBuffer> bs, id<MTLBuffer> bo, uint32_t reps) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < reps; ++i) {
                [enc setComputePipelineState:p];
                id<MTLBuffer> bufs[] = {bQ, bc, bK, bG, bs, bZ, bN, bo};
                for (uint32_t j = 0; j < 8; ++j) [enc setBuffer:bufs[j] offset:0 atIndex:j];
                [enc setBytes:&pc length:20 atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(hK, 1, slots)
                    threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
                if (i + 1u < reps)
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "dnstepcmp Metal failure: %s\n",
                        cb.error.localizedDescription.UTF8String);
                return -1.0;
            }
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps;
        }
    };
    auto submitSplit = [&](id<MTLComputePipelineState> state, id<MTLBuffer> bc,
                           id<MTLBuffer> bs, id<MTLBuffer> bo, uint32_t reps) -> double {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < reps; ++i) {
                [enc setComputePipelineState:pPrep];
                [enc setBuffer:bQc offset:0 atIndex:0];
                [enc setBuffer:bc offset:0 atIndex:1];
                [enc setBuffer:bK offset:0 atIndex:2];
                [enc setBytes:&pc length:20 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(hK, 1, slots)
                    threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setComputePipelineState:state];
                id<MTLBuffer> bufs[] = {bQc, bG, bs, bZ, bN, bo};
                for (uint32_t j = 0; j < 6; ++j) [enc setBuffer:bufs[j] offset:0 atIndex:j];
                [enc setBytes:&pc length:20 atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(hV, 1, slots)
                    threadsPerThreadgroup:MTLSizeMake(dS, 1, 1)];
                if (i + 1u < reps)
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) return -1.0;
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps;
        }
    };
    auto countDiff = [](const float* a, const float* b, size_t n, double& maxAbs) {
        size_t d = 0; maxAbs = 0.0;
        for (size_t i = 0; i < n; ++i) {
            d += a[i] != b[i];
            maxAbs = std::max(maxAbs, std::fabs((double)a[i] - b[i]));
        }
        return d;
    };

    printf("\n== dnstepcmp decode megakernel, slots=%u ==\n", slots);
    printf("  %-7s %-9s %11s %11s %11s %10s %10s\n",
           "mapping", "variant", "conv_diff", "state_diff", "out_diff", "gpu_us", "speedup");
    bool pass = true;
    for (uint32_t kd : {0u, 2u}) {
        pc.kd = kd;
        reset(bCr, bSr);
        if (submit(pRef, bCr, bSr, bOr, 1) < 0) return false;
        memcpy(refConv.data(), bCr.contents, refConv.size() * 4u);
        memcpy(refState.data(), bSr.contents, refState.size() * 4u);
        memcpy(refOut.data(), bOr.contents, refOut.size() * 4u);
        double usRef = 0.0;
        if (iters) {
            reset(bCr, bSr); submit(pRef, bCr, bSr, bOr, std::min(100u, iters));
            reset(bCr, bSr); usRef = submit(pRef, bCr, bSr, bOr, iters);
        }
        printf("  kDiv=%-2u %-9s %11s %11s %11s %10.2f %10s\n",
               kd, "stream", "-", "-", "-", usRef, "1.000x");
        for (const Variant& v : vars) {
            reset(bCc, bSc);
            if (submit(v.p, bCc, bSc, bOc, 1) < 0) return false;
            double ac, as, ao;
            const size_t dc = countDiff(refConv.data(),
                                        (const float*)bCc.contents, conv0.size(), ac);
            const size_t ds = countDiff(refState.data(),
                                        (const float*)bSc.contents, state0.size(), as);
            const size_t dout = countDiff(refOut.data(),
                                          (const float*)bOc.contents, z.size(), ao);
            const bool exact = dc == 0 && ds == 0 && dout == 0;
            pass &= exact;
            double us = 0.0;
            if (iters) {
                reset(bCc, bSc); submit(v.p, bCc, bSc, bOc, std::min(100u, iters));
                reset(bCc, bSc); us = submit(v.p, bCc, bSc, bOc, iters);
            }
            printf("  kDiv=%-2u %-9s %11zu %11zu %11zu %10.2f %9.3fx  %s (max %.3g)\n",
                   kd, v.name, dc, ds, dout, us, us ? usRef / us : 0.0,
                   exact ? "EXACT" : "FAIL", std::max({ac, as, ao}));
        }
        for (const Variant& v : splitVars) {
            memcpy(bQc.contents, qkv.data(), qkv.size() * 4u);
            reset(bCc, bSc);
            if (submitSplit(v.p, bCc, bSc, bOc, 1) < 0) return false;
            double ac, as, ao;
            const size_t dc = countDiff(refConv.data(), (const float*)bCc.contents,
                                        conv0.size(), ac);
            const size_t ds = countDiff(refState.data(), (const float*)bSc.contents,
                                        state0.size(), as);
            const size_t dout = countDiff(refOut.data(), (const float*)bOc.contents,
                                          z.size(), ao);
            const bool exact = dc == 0 && ds == 0 && dout == 0;
            pass &= exact;
            double us = 0.0;
            if (iters) {
                memcpy(bQc.contents, qkv.data(), qkv.size() * 4u);
                reset(bCc, bSc);
                submitSplit(v.p, bCc, bSc, bOc, std::min(100u, iters));
                memcpy(bQc.contents, qkv.data(), qkv.size() * 4u);
                reset(bCc, bSc);
                us = submitSplit(v.p, bCc, bSc, bOc, iters);
            }
            printf("  kDiv=%-2u %-9s %11zu %11zu %11zu %10.2f %9.3fx  %s (max %.3g)\n",
                   kd, v.name, dc, ds, dout, us, us ? usRef / us : 0.0,
                   exact ? "EXACT" : "FAIL", std::max({ac, as, ao}));
        }
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
    uint32_t stepPW = 8;
    if (const char* v = getenv("QK_DN_STEP_RES")) {
        const uint32_t x = (uint32_t)atoi(v);
        if (x == 0u || x == 8u) stepPW = x;
    }
    const uint32_t stepNsg = stepPW ? 8u : 4u;
    const uint32_t stepPanels = stepPW ? dS / stepPW : 2u;
    const uint32_t chQkv = (2 * hK + hV) * dS;
    const uint32_t TnCap = std::max(TnMain, 512u);
    printf("\n== dncmp  chunked delta rule vs dn_step_batch (dS=%u hK=%u hV=%u, step=%s, nsg=%u) ==\n",
           dS, hK, hV, stepPW ? "resident-8" : "streamed-64", stepNsg);

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
    id<MTLBuffer> bSc = createBuf(c, st0.size() * 4);
    id<MTLBuffer> bOa = createBuf(c, (size_t)TnCap * hV * dS * 4);
    id<MTLBuffer> bOb = createBuf(c, (size_t)TnCap * hV * dS * 4);
    id<MTLBuffer> bOc = createBuf(c, (size_t)TnCap * hV * dS * 4);
    const size_t nChMax = (TnCap + 63) / 64;
    id<MTLBuffer> bKQ = createBuf(c, nChMax * hK * 2 * 64 * 64 * 4);
    id<MTLBuffer> bUW = createBuf(c, nChMax * hV * 4 * 64 * (size_t)dS * 4);
    id<MTLBuffer> bAtt = createBuf(c, nChMax * hV * 64 * 64 * 4);
    id<MTLBuffer> bEl = createBuf(c, nChMax * hV * 4);

    id<MTLComputePipelineState> pSeq = getPipe(c, "dn_batch", "dn_step_batch", 0);
    id<MTLComputePipelineState> pKq = getPipe(c, "dn_chunk", "dn_chunk_kq", 0);
    id<MTLComputePipelineState> pSolve = getPipe(c, "dn_chunk", "dn_chunk_solve", 0);
    id<MTLComputePipelineState> pStepBase = getPipe(c, "dn_chunk", "dn_chunk_step", dS);
    id<MTLComputePipelineState> pStepC = getPipe(c, "dn_chunk",
        stepPW ? "dn_chunk_step_res8_s8" : "dn_chunk_step", dS);

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
    auto encChainTo = [&](id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> stepPipe,
                          id<MTLBuffer> out, id<MTLBuffer> state,
                          uint32_t panels, uint32_t threads) {
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
        [enc setComputePipelineState:stepPipe];
        [enc setBuffer:bUW offset:0 atIndex:0];
        [enc setBuffer:bAtt offset:0 atIndex:1];
        [enc setBuffer:bEl offset:0 atIndex:2];
        [enc setBuffer:out offset:0 atIndex:3];
        [enc setBuffer:state offset:0 atIndex:4];
        [enc setBytes:&pc length:20 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(hV, 1, panels)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    };
    auto encChain = [&](id<MTLComputeCommandEncoder> enc) {
        encChainTo(enc, pStepC, bOb, bSb, stepPanels, stepNsg * 32u);
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

    if (stepPW && TnMain <= TnCap) {
        pc.Tn = TnMain;
        memcpy(bSb.contents, st0.data(), st0.size() * 4);
        memcpy(bSc.contents, st0.data(), st0.size() * 4);
        runOnce(true);
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encChainTo(enc, pStepBase, bOc, bSc, 2u, dS);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        size_t diffO = 0, diffS = 0;
        const float* ob = (const float*)bOb.contents;
        const float* oc = (const float*)bOc.contents;
        const float* sb = (const float*)bSb.contents;
        const float* sc = (const float*)bSc.contents;
        for (size_t i = 0; i < (size_t)TnMain * hV * dS; ++i)
            if (ob[i] != oc[i]) ++diffO;
        for (size_t i = 0; i < st0.size(); ++i)
            if (sb[i] != sc[i]) ++diffS;
        const bool exact = diffO == 0 && diffS == 0;
        pass &= exact;
        printf("  resident vs streamed @Tn=%u: o %zu differ, state %zu differ -> %s\n",
               TnMain, diffO, diffS, exact ? "BIT-EXACT" : "FAIL");
    }

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
                        [enc dispatchThreadgroups:MTLSizeMake(hV, 1, stepPanels)
                            threadsPerThreadgroup:MTLSizeMake(stepNsg * 32u, 1, 1)];
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

        if (mode == "slotcmp") {
            if (argc < 4) {
                fprintf(stderr, "usage: qk slotcmp <ids-file> <nGen> "
                                "[nSlots] [tmax] [prompt-repeat]\n");
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
            const uint32_t ns = argc > 4 ? (uint32_t)atoi(argv[4]) : 8;
            const uint32_t tmax = argc > 5 ? (uint32_t)atoi(argv[5]) : 512;
            const uint32_t repeat = argc > 6 ? (uint32_t)atoi(argv[6]) : 1;
            if (repeat > 1) {
                const std::vector<uint32_t> one = prompt;
                prompt.reserve(one.size() * repeat);
                for (uint32_t r = 1; r < repeat; ++r)
                    prompt.insert(prompt.end(), one.begin(), one.end());
            }
            if (nGen < 8 || ns < 2 || ns > 16 || prompt.empty() ||
                prompt.size() + nGen > tmax) {
                fprintf(stderr, "slotcmp: require nGen>=8, 2<=nSlots<=16, prompt+nGen<=tmax\n");
                return 1;
            }
            qk_config cfg{ns, tmax, 1};
            char err[256] = {0};
            qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
            if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
            std::vector<std::vector<uint32_t>> prompts(ns, prompt);
            for (uint32_t s = 1; s < ns; ++s) {
                prompts[s][0] = (prompts[s][0] + 7919u * s) % e->vocab;
                if (prompts[s].size() > 256u)
                    prompts[s].resize(prompts[s].size() - (17u * s) % 64u);
            }
            std::vector<uint32_t> outTok(ns), outCnt(ns);
            uint32_t fin = 0;
            auto stepInto = [&](std::vector<std::vector<uint32_t>>& gen) {
                int active = qk_step_chunk(e, outTok.data(), outCnt.data(), &fin);
                if (active < 0) return active;
                for (uint32_t s = 0; s < ns; ++s)
                    for (uint32_t i = 0; i < outCnt[s]; ++i) gen[s].push_back(outTok[s]);
                return active;
            };

            // Establish one golden trajectory per heterogeneous prompt while
            // every public index is independently mapped to physical stripe 0.
            std::vector<std::vector<uint32_t>> golden(ns);
            for (uint32_t target = 0; target < ns; ++target) {
                std::vector<std::vector<uint32_t>> gen(ns);
                if (qk_slot_start(e, target, prompts[target].data(),
                                  (uint32_t)prompts[target].size(),
                                  nGen, 0)) {
                    fprintf(stderr, "slotcmp: start %u failed\n", target);
                    qk_close(e); return 1;
                }
                while (stepInto(gen) > 0) {}
                golden[target] = std::move(gen[target]);
            }

            // Exercise a hole at every physical index, then one simultaneous
            // irregular mask. A survivor step forces state moves; restarting
            // removed clients makes subsequent dispatches heterogeneous in
            // position while logical output ownership stays stable.
            for (uint32_t churn = 0; churn <= ns; ++churn) {
                std::vector<std::vector<uint32_t>> gen(ns);
                for (uint32_t s = 0; s < ns; ++s) {
                    if (qk_slot_start(e, s, prompts[s].data(),
                                      (uint32_t)prompts[s].size(), nGen, 0)) {
                        fprintf(stderr, "slotcmp: churn %u start %u failed\n", churn, s);
                        qk_close(e); return 1;
                    }
                }
                for (uint32_t i = 0; i < 3; ++i)
                    if (stepInto(gen) <= 0) {
                        fprintf(stderr, "slotcmp: early finish in churn %u\n", churn);
                        qk_close(e); return 1;
                    }
                std::vector<uint32_t> removed;
                for (uint32_t s = 0; s < ns; ++s) {
                    const bool remove = churn < ns ? s == churn
                        : ((s % 3) == 1 || (s + 1 == ns && s != 0));
                    if (!remove) continue;
                    qk_slot_cancel(e, s);
                    gen[s].clear();
                    removed.push_back(s);
                }
                if (stepInto(gen) <= 0) {
                    fprintf(stderr, "slotcmp: no survivors in churn %u\n", churn);
                    qk_close(e); return 1;
                }
                for (uint32_t s : removed) {
                    if (qk_slot_start(e, s, prompts[s].data(),
                                      (uint32_t)prompts[s].size(), nGen, 0)) {
                        fprintf(stderr, "slotcmp: churn %u restart %u failed\n", churn, s);
                        qk_close(e); return 1;
                    }
                }
                while (stepInto(gen) > 0) {}
                for (uint32_t s = 0; s < ns; ++s) {
                    if (gen[s] == golden[s]) continue;
                    size_t at = 0;
                    while (at < gen[s].size() && at < golden[s].size() &&
                           gen[s][at] == golden[s][at]) ++at;
                    fprintf(stderr, "slotcmp: churn %u slot %u differs at %zu "
                                    "(got %zu, want %zu)\n",
                            churn, s, at, gen[s].size(), golden[s].size());
                    qk_close(e); return 1;
                }
            }
            printf("slotcmp: PASS %u/%u lone slots + every hole + mask churn, "
                   "%zu tokens exact\n", ns, ns, golden[0].size());
            qk_close(e);
            return 0;
        }

        if (mode == "prefillmulticmp") {
            const uint32_t N = argc > 2 ? (uint32_t)atoi(argv[2]) : 64;
            const uint32_t B = argc > 3 ? (uint32_t)atoi(argv[3]) : 2;
            const uint32_t iters = argc > 4 ? (uint32_t)atoi(argv[4]) : 3;
            const uint32_t base0 = argc > 5 ? (uint32_t)atoi(argv[5]) : 0;
            if (!N || B < 2 || B > 16 || (size_t)N * B > 1024) {
                fprintf(stderr, "usage: qk prefillmulticmp [N] [B=2..16] [iters]\n");
                return 1;
            }
            qk_config cfg{B, std::max(128u, base0 + 5u * B + N + 16u), 1};
            char err[256] = {0};
            qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
            if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
            if ((size_t)N * B > e->maxB ||
                (size_t)B * ((N + 63u) / 64u) > (e->maxB + 63u) / 64u) {
                fprintf(stderr, "prefillmulticmp: B*N/chunk scratch exceeds QK_MAXB=%u\n",
                        e->maxB);
                qk_close(e); return 1;
            }
            std::mt19937 rng(0x504d554cu + N * 17u + B);
            std::vector<uint32_t> toks((size_t)N * B), pending(B), slots(B), bases(B);
            std::vector<std::vector<uint32_t>> prefix(B);
            for (uint32_t q = 0; q < B; ++q) {
                slots[q] = q;
                bases[q] = base0 ? base0 + 5u * q : 0u;
                prefix[q].resize(bases[q]);
                for (uint32_t& t : prefix[q]) t = rng() % e->vocab;
                pending[q] = rng() % e->vocab;
                for (uint32_t i = 0; i < N; ++i) toks[(size_t)q * N + i] = rng() % e->vocab;
            }
            auto seedPrefixes = [&]() {
                std::vector<float> unused;
                for (uint32_t q = 0; q < B; ++q) {
                    if (prefix[q].empty()) { e->resetStripe(q); continue; }
                    e->prefillBatchLast(prefix[q].data(), bases[q], q, unused,
                                        /*wantLogits=*/false);
                }
            };
            std::vector<std::vector<uint8_t>> ref(B, std::vector<uint8_t>(e->snapSize));
            std::vector<uint32_t> refTok(B), gotTok(B);
            std::vector<float> logits;
            double serialGpu = 0.0;
            for (uint32_t q = 0; q < B; ++q) {
                if (!prefix[q].empty())
                    e->prefillBatchLast(prefix[q].data(), bases[q], q, logits,
                                        /*wantLogits=*/false);
                e->prefillBatchLast(toks.data() + (size_t)q * N, N, q, logits,
                                    /*wantLogits=*/false, bases[q]);
                serialGpu += e->lastCbGpu;
                e->copyStripes(q, ref[q].data(), /*save=*/true, bases[q] + N);
                e->prefillBatchLast(&pending[q], 1, q, logits, /*wantLogits=*/true,
                                    bases[q] + N);
                refTok[q] = (uint32_t)std::distance(
                    logits.begin(), std::max_element(logits.begin(), logits.end()));
            }
            seedPrefixes();
            e->prefillBatchLast(toks.data(), N * B, 0, logits, /*wantLogits=*/false, 0,
                                nullptr, nullptr, nullptr, false, B,
                                slots.data(), bases.data());
            const double multiGpu = e->lastCbGpu;
            double maxAbs = 0.0, sumD2 = 0.0, sumR2 = 0.0;
            uint64_t nValue = 0, bitDiff = 0;
            std::vector<uint8_t> got(e->snapSize);
            auto compareRegion = [&](const uint8_t* ap, const uint8_t* bp,
                                     size_t bytes, uint32_t elemBytes,
                                     uint64_t& nd, double& ma) {
                const size_t nElem = bytes / elemBytes;
                for (size_t i = 0; i < nElem; ++i) {
                    float av, bv;
                    if (elemBytes == 1u) {
                        av = (float)ap[i]; bv = (float)bp[i];
                        nd += ap[i] != bp[i];
                    } else if (elemBytes == 2u) {
                        const uint16_t ah = ((const uint16_t*)ap)[i];
                        const uint16_t bh = ((const uint16_t*)bp)[i];
                        av = qk_f16_to_f32(ah); bv = qk_f16_to_f32(bh);
                        nd += ah != bh;
                    } else {
                        const uint32_t ab = ((const uint32_t*)ap)[i];
                        const uint32_t bb = ((const uint32_t*)bp)[i];
                        memcpy(&av, &ab, sizeof av); memcpy(&bv, &bb, sizeof bv);
                        nd += ab != bb;
                    }
                    const double d = (double)bv - av;
                    ma = std::max(ma, std::abs(d));
                    sumD2 += d * d;
                    sumR2 += (double)av * av;
                }
                nValue += nElem;
            };
            for (uint32_t q = 0; q < B; ++q) {
                e->copyStripes(q, got.data(), /*save=*/true, bases[q] + N);
                for (uint32_t il = 0; il < e->nLayer; ++il) {
                    const qk_engine::Layer& L = e->layers[il];
                    uint64_t nd = 0;
                    double ma = 0.0;
                    const uint32_t eb = L.rec ? 4u : e->kvBytes;
                    compareRegion(ref[q].data() + e->snapOff1[il],
                                  got.data() + e->snapOff1[il], L.ps1, eb, nd, ma);
                    compareRegion(ref[q].data() + e->snapOff2[il],
                                  got.data() + e->snapOff2[il], L.ps2, eb, nd, ma);
                    bitDiff += nd; maxAbs = std::max(maxAbs, ma);
                    if (nd && getenv("QK_PMULTI_DETAIL"))
                        printf("    detail slot=%u layer=%u %s diff=%llu max=%.3g\n",
                               q, il, L.rec ? "dn" : "attn",
                               (unsigned long long)nd, ma);
                }
                e->prefillBatchLast(&pending[q], 1, q, logits, /*wantLogits=*/true,
                                    bases[q] + N);
                gotTok[q] = (uint32_t)std::distance(
                    logits.begin(), std::max_element(logits.begin(), logits.end()));
            }
            bool exactTok = refTok == gotTok;
            printf("prefillmulticmp: B=%u N=%u state bitdiff=%llu/%llu max_abs=%.3g "
                   "rel_rms=%.3g handoff=%s\n", B, N,
                   (unsigned long long)bitDiff, (unsigned long long)nValue, maxAbs,
                   sumR2 > 0 ? sqrt(sumD2 / sumR2) : 0.0,
                   exactTok ? "EXACT" : "MISMATCH");
            printf("  serial %.3f ms | coalesced %.3f ms | %.2fx\n",
                   serialGpu * 1e3, multiGpu * 1e3,
                   multiGpu > 0 ? serialGpu / multiGpu : 0.0);
            for (uint32_t q = 0; q < B; ++q)
                printf("  slot %u token %u/%u%s\n", q, refTok[q], gotTok[q],
                       refTok[q] == gotTok[q] ? "" : " **DIFF**");
            if (iters) {
                double s = 0.0, m = 0.0;
                for (uint32_t it = 0; it < iters; ++it) {
                    seedPrefixes();
                    for (uint32_t q = 0; q < B; ++q) {
                        e->prefillBatchLast(toks.data() + (size_t)q * N, N, q, logits,
                                            /*wantLogits=*/false, bases[q]);
                        s += e->lastCbGpu;
                    }
                    seedPrefixes();
                    e->prefillBatchLast(toks.data(), N * B, 0, logits,
                                        /*wantLogits=*/false, 0, nullptr, nullptr, nullptr,
                                        false, B, slots.data(), bases.data());
                    m += e->lastCbGpu;
                }
                printf("  avg serial %.3f ms | coalesced %.3f ms | %.2fx (%u iters)\n",
                       s * 1e3 / iters, m * 1e3 / iters, m > 0 ? s / m : 0.0, iters);
            }
            qk_close(e);
            return exactTok ? 0 : 1;
        }

        if (mode == "fadecode") {
            const uint32_t B = argc > 2 ? (uint32_t)atoi(argv[2]) : 1u;
            const uint32_t pos = argc > 3 ? (uint32_t)atoi(argv[3]) : 8192u;
            const uint32_t steps = argc > 4 ? (uint32_t)atoi(argv[4]) : 64u;
            constexpr uint32_t warm = 4u;
            const uint32_t tmax = argc > 5 ? (uint32_t)atoi(argv[5])
                : pos + steps + warm + 64u;
            const uint32_t minPos = argc > 6 ? (uint32_t)atoi(argv[6]) : pos;
            if (!B || B > 16u || !pos || !steps || minPos > pos ||
                pos + steps + warm > tmax) {
                fprintf(stderr, "usage: qk fadecode [slots=1..16] [pos] [steps] "
                                "[tmax] [min-pos]\n");
                return 1;
            }
            qk_config cfg{B, tmax, 1u};
            char err[256] = {0};
            qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
            if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }

            static constexpr uint32_t seed[] = {
                12162u, 5028u, 264u, 854u, 11u, 303u, 264u,
                11012u, 13721u, 85624u, 1881u, 1330u, 22767u, 11u,
                1017u, 11815u, 449u, 2235u, 8598u, 24946u, 852u};
            std::vector<uint32_t> prefix(pos);
            for (uint32_t i = 0; i < pos; ++i) prefix[i] = seed[i % std::size(seed)];
            std::vector<float> unused;
            e->resetStripe(0);
            for (uint32_t off = 0; off < pos; off += e->maxB) {
                const uint32_t n = std::min(e->maxB, pos - off);
                e->prefillBatchLast(prefix.data() + off, n, 0, unused,
                                    /*wantLogits=*/false, off);
            }
            for (uint32_t s = 1; s < B; ++s) e->moveSlotState(0, s, pos);

            uint32_t* slotIn = (uint32_t*)e->bSlotIn.contents;
            uint32_t* slotPos = (uint32_t*)e->bSlotPos.contents;
            const uint32_t* sampled = (const uint32_t*)e->bTok.contents;
            std::vector<uint32_t> basePos(B, pos);
            if (B > 1u && minPos < pos) {
                for (uint32_t s = 0; s < B; ++s)
                    basePos[s] = minPos + (uint64_t)(pos - minPos) * s / (B - 1u);
                if (const char* skew = getenv("QK_FA_SKEW")) {
                    const uint32_t nLong = std::min(
                        B, (uint32_t)std::max(1, atoi(skew)));
                    std::fill(basePos.begin(), basePos.end(), minPos);
                    std::fill(basePos.end() - nLong, basePos.end(), pos);
                }
            }
            const bool homogeneous = minPos == pos;
            for (uint32_t s = 0; s < B; ++s) slotIn[s] = prefix.back();
            double gpu = 0.0, wall = 0.0;
            bool identical = true;
            std::vector<uint32_t> gen;
            std::vector<uint64_t> hash(B, 1469598103934665603ull);
            for (uint32_t it = 0; it < warm + steps; ++it) {
                for (uint32_t s = 0; s < B; ++s) slotPos[s] = basePos[s] + it;
                auto t0 = std::chrono::steady_clock::now();
                @autoreleasepool {
                    id<MTLCommandBuffer> cb = [e->c.queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc =
                        [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    e->encodeStep(enc, B, homogeneous && B > 1u);
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    if (cb.status != MTLCommandBufferStatusCompleted) {
                        fprintf(stderr, "fadecode Metal failure: %s\n",
                                cb.error.localizedDescription.UTF8String);
                        qk_close(e); return 1;
                    }
                    if (it >= warm) gpu += cb.GPUEndTime - cb.GPUStartTime;
                }
                const double w = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                if (it >= warm) wall += w;
                for (uint32_t s = 1; s < B; ++s) identical &= sampled[s] == sampled[0];
                for (uint32_t s = 0; s < B; ++s) slotIn[s] = sampled[s];
                if (it >= warm) {
                    gen.push_back(sampled[0]);
                    for (uint32_t s = 0; s < B; ++s) {
                        hash[s] ^= sampled[s];
                        hash[s] *= 1099511628211ull;
                    }
                }
            }
            printf("fadecode: B=%u pos=%u..%u steps=%u | gpu %.3f ms/step | "
                   "wall %.3f ms/step | slots identical: %s\n",
                   B, minPos, pos, steps, gpu * 1e3 / steps, wall * 1e3 / steps,
                   identical ? "YES" : "NO (heterogeneous expected)");
            printf("HASH:");
            for (uint64_t h : hash) printf(" %016llx", (unsigned long long)h);
            printf("\n");
            printf("GEN:");
            for (uint32_t t : gen) printf(" %u", t);
            printf("\n");
            qk_close(e);
            return homogeneous && !identical ? 1 : 0;
        }

        if (mode == "serve-test") {
            if (argc < 4) {
                fprintf(stderr, "usage: qk serve-test <ids-file> <nGen> "
                                "[nSlots] [tmax] [prompt-repeat]\n");
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
            uint32_t repeat = argc > 6 ? (uint32_t)atoi(argv[6]) : 1;
            if (repeat > 1) {
                const std::vector<uint32_t> one = prompt;
                prompt.reserve(one.size() * repeat);
                for (uint32_t r = 1; r < repeat; ++r)
                    prompt.insert(prompt.end(), one.begin(), one.end());
            }
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

        if (mode == "cachecmp") {
            const uint32_t tmax = argc > 2 ? (uint32_t)atoi(argv[2]) : 512u;
            if (tmax < 400u || tmax > 65536u) {
                fprintf(stderr, "usage: qk cachecmp [tmax=512, >=400]\n");
                return 1;
            }
            const char* oldPc = getenv("QK_PCACHE");
            const bool hadPc = oldPc != nullptr;
            const std::string oldPcValue = oldPc ? oldPc : "";
            setenv("QK_PCACHE", "6", 1);
            qk_config cfg{2, tmax, 1};
            char err[256] = {0};
            qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
            if (hadPc) setenv("QK_PCACHE", oldPcValue.c_str(), 1);
            else unsetenv("QK_PCACHE");
            if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }

            static constexpr uint32_t seed[] = {
                12162u, 5028u, 264u, 854u, 11u, 303u, 264u,
                11012u, 13721u, 85624u, 1881u, 1330u, 22767u, 11u};
            auto makePrompt = [&](uint32_t tag, uint32_t n) {
                std::vector<uint32_t> p(n);
                for (uint32_t i = 0; i < n; ++i)
                    p[i] = (seed[i % std::size(seed)] + tag * 7919u + i * 17u) % e->vocab;
                return p;
            };
            std::vector<float> unused;
            auto seedState = [&](uint32_t stripe, const std::vector<uint32_t>& p) {
                e->resetStripe(stripe);
                std::vector<float> hidden;
                if (!e->firstStage()) {
                    hidden.resize((size_t)p.size() * qk_engine::nEmbd);
                    for (size_t i = 0; i < hidden.size(); ++i)
                        hidden[i] = (float)((int)((i * 17u + p[i / qk_engine::nEmbd]) % 257u) -
                                            128) * (1.0f / 128.0f);
                }
                for (uint32_t off = 0; off < p.size();) {
                    const uint32_t n = std::min(e->maxB, (uint32_t)p.size() - off);
                    e->prefillBatchLast(e->firstStage() ? p.data() + off : nullptr,
                                        n, stripe, unused, /*wantLogits=*/false, off,
                                        /*argmaxOut=*/nullptr,
                                        e->firstStage() ? nullptr
                                            : hidden.data() + (size_t)off * qk_engine::nEmbd);
                    off += n;
                }
            };

            std::vector<uint8_t> ref(e->snapSize), got(e->snapSize);
            uint64_t diffBytes = 0;
            auto compareEntry = [&](uint32_t idx, uint32_t stateTok,
                                    uint32_t saveTok, uint32_t loadTok,
                                    uint32_t tag) {
                const std::vector<uint32_t> p = makePrompt(tag, stateTok);
                seedState(0, p);
                std::fill(ref.begin(), ref.end(), 0);
                e->copyStripes(0, ref.data(), /*save=*/true, loadTok);
                auto ts = std::chrono::steady_clock::now();
                if (qk_state_save(e, 0, idx, saveTok)) return false;
                const double saveMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - ts).count();
                const size_t expected = e->pcacheCompact ? e->packedSnapSize(saveTok) : e->snapSize;
                if (e->pcache[idx].snapBytes != expected) return false;
                e->resetStripe(1);
                auto tl = std::chrono::steady_clock::now();
                if (qk_state_load(e, 1, idx, loadTok)) return false;
                const double loadMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tl).count();
                std::fill(got.begin(), got.end(), 0);
                e->copyStripes(1, got.data(), /*save=*/true, loadTok);
                uint64_t nd = 0;
                if (memcmp(ref.data(), got.data(), ref.size()) != 0)
                    for (size_t i = 0; i < ref.size(); ++i) nd += ref[i] != got[i];
                diffBytes += nd;
                printf("  idx=%u state=%u save=%u load=%u payload=%.1f MiB "
                       "copy=%.1f/%.1f ms diff=%llu\n",
                       idx, stateTok, saveTok, loadTok,
                       e->pcache[idx].snapBytes / 1048576.0, saveMs, loadMs,
                       (unsigned long long)nd);
                return nd == 0;
            };

            bool ok = true;
            // Reuse one entry in both growth directions; this catches stale
            // head-stride assumptions when a long snapshot is overwritten by
            // a short one and vice versa.
            for (uint32_t n : {384u, 8u, 255u, 17u, 127u, 64u})
                ok &= compareEntry(0, n, n, n, 10u + n);
            // A smaller-prefix restore from a longer packed entry must skip
            // each stored head's unused suffix while restoring full DN state.
            ok &= compareEntry(1, 384u, 384u, 65u, 777u);
            ok &= qk_state_load(e, 1, 1, 385u) == -2;
            // n_tok=0 retains the public ABI's full-stripe behavior.
            ok &= compareEntry(0, 384u, 0u, 0u, 888u);
            ok &= compareEntry(0, 8u, 8u, 8u, 999u);  // full -> short shrink

            // Populate all six automatic entries, touch entry 0, then insert
            // a seventh distinct prefix. The least-recently-used entry 1 must
            // be replaced while entry 0 and entries 2..5 survive unchanged.
            std::array<uint32_t, 7> sig{};
            for (uint32_t j = 0; j < 6; ++j) {
                std::vector<uint32_t> p = makePrompt(200u + j, 9u + j);
                sig[j] = p[0];
                seedState(0, p);
                qk_engine::Slot& sl = e->slots[0];
                sl.prompt = p; sl.genTokens.clear(); sl.pos = (uint32_t)p.size();
                e->snapshotSlot(0);
            }
            e->restoreInto(0, 0);  // entry 0 becomes newest
            {
                std::vector<uint32_t> p = makePrompt(206u, 15u);
                sig[6] = p[0];
                seedState(0, p);
                qk_engine::Slot& sl = e->slots[0];
                sl.prompt = p; sl.genTokens.clear(); sl.pos = (uint32_t)p.size();
                e->snapshotSlot(0);
            }
            ok &= e->pcache.size() == 6 && e->pcache[0].valid && e->pcache[1].valid;
            ok &= !e->pcache[0].tokens.empty() && e->pcache[0].tokens[0] == sig[0];
            ok &= !e->pcache[1].tokens.empty() && e->pcache[1].tokens[0] == sig[6];
            for (uint32_t j = 2; j < 6; ++j)
                ok &= e->pcache[j].valid && !e->pcache[j].tokens.empty() &&
                      e->pcache[j].tokens[0] == sig[j];
            std::vector<uint32_t> probe = e->pcache[0].tokens;
            probe.push_back(42u);
            ok &= e->matchPrefix(probe.data(), (uint32_t)probe.size()) == 0;
            size_t payload = 0;
            for (const auto& ce : e->pcache) payload += ce.snapBytes;
            printf("cachecmp: %s layout, diff=%llu, six-entry payload %.1f MiB, LRU=%s -> %s\n",
                   e->pcacheCompact ? "packed" : "full",
                   (unsigned long long)diffBytes, payload / 1048576.0,
                   ok ? "PASS" : "FAIL", ok ? "PASS" : "FAIL");
            qk_close(e);
            return ok ? 0 : 1;
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
                auto t0 = std::chrono::steady_clock::now();
                qk_slot_start(e, 0, ids.data(), (uint32_t)ids.size(), gN, 0);
                uint32_t ch = qk_chunk(e);
                std::vector<uint32_t> ot((size_t)ch), oc(1);
                uint32_t fin = 0;
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

        if (mode == "stagecmp") {
            if (argc < 3) {
                fprintf(stderr, "usage: qk stagecmp <ids-file> [topk] [tmax]\n");
                return 1;
            }
            std::vector<uint32_t> toks;
            FILE* f = fopen(argv[2], "r");
            if (!f) { perror(argv[2]); return 1; }
            int v;
            while (fscanf(f, "%d%*[, \n]", &v) == 1) toks.push_back((uint32_t)v);
            fclose(f);
            const uint32_t k = argc > 3 ? (uint32_t)atoi(argv[3]) : 16u;
            const uint32_t tmax = argc > 4 ? (uint32_t)atoi(argv[4]) : 2048u;
            qk_config cfg{1, tmax, 8};
            char err[256] = {0};
            qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
            if (!e) { fprintf(stderr, "qk_open failed: %s\n", err); return 1; }
            if (toks.empty() || toks.size() > tmax || k < 1 || k > 256) {
                fprintf(stderr, "stagecmp: bad token count/topk\n");
                qk_close(e);
                return 1;
            }
            std::vector<uint32_t> ids(toks.size()), topIds(k);
            std::vector<float> topVals(k);
            const int rc = e->stageRun(0, toks.data(), nullptr, (uint32_t)toks.size(),
                                       0, nullptr, ids.data());
            const int trc = rc ? rc : e->stageTopK(k, topIds.data(), topVals.data());
            bool ordered = true;
            for (uint32_t i = 1; i < k; ++i) ordered &= topVals[i - 1] >= topVals[i];
            const bool ok = trc == 0 && topIds[0] == ids.back() && ordered;
            printf("stagecmp: final argmax=%u top1=%u top-%u ordered=%s -> %s\n",
                   ids.back(), topIds[0], k, ordered ? "YES" : "NO", ok ? "PASS" : "FAIL");
            qk_close(e);
            return ok ? 0 : 1;
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
            // The production engine allocates the 62.8-MiB recurrent rollback
            // stripe only for enabled speculation.  This harness calls the
            // private rollback path directly, so request that scratch without
            // turning on n-gram speculation in the serial reference run.
            setenv("QK_SPEC_SCRATCH", "1", 1);
            qk_engine* e = qk_open(ggufPath(), &cfg, err, sizeof err);
            unsetenv("QK_SPEC_SCRATCH");
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
                    std::vector<uint32_t> argmaxB(sz);
                    uint32_t tokS = e->serialPrefillLogits(toks.data(), sz, 1, logitsS);
                    e->prefillBatchLast(toks.data(), sz, 0, logitsB, /*wantLogits=*/true,
                                        /*base=*/0, argmaxB.data());
                    uint32_t tokB = argmaxB.back();

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

        if (mode == "counters") {
            printCounterInfo(c);
            return 0;
        }

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
        } else if (mode == "normroutecmp") {
            ok = caseNormRouteCmp(c, argU(2, 256), argU(3, 1), argU(4, 500));
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
        } else if (mode == "moegucmp") {
            ok = caseMoeGuCmp(c, argU(2, 0), argU(3, 100));
        } else if (mode == "block") {
            ok = caseBlock(c, argU(2, 0), argU(3, 3), argU(4, 200));
        } else if (mode == "ablock") {
            ok = caseABlock(c, argU(2, 3), argU(3, 3), argU(4, 200));
        } else if (mode == "bgemm") {
            ok = caseBGemm(c, argU(2, 8192), argU(3, 2048), argU(4, 128), argU(5, 50));
        } else if (mode == "iq4gemm") {
            if (argc < 3) {
                fprintf(stderr, "usage: qk iq4gemm <tensor> [N] [iters]\n");
                return 1;
            }
            ok = caseIQ4Gemm(c, argv[2], argU(3, 512), argU(4, 50));
        } else if (mode == "q8gemvcmp") {
            if (argc < 3) {
                fprintf(stderr, "usage: qk q8gemvcmp <tensor> [tpr] [iters]\n");
                return 1;
            }
            ok = caseQ8GemvCmp(c, argv[2], argU(3, 16), argU(4, 500));
        } else if (mode == "iq4gemvcmp") {
            if (argc < 3) {
                fprintf(stderr, "usage: qk iq4gemvcmp <tensor> [iters]\n");
                return 1;
            }
            ok = caseIQ4GemvCmp(c, argv[2], argU(3, 500));
        } else if (mode == "q6gemvcmp") {
            ok = caseQ6GemvCmp(c, argU(2, 2), argU(3, 10));
        } else if (mode == "headcmp") {
            ok = caseHeadCmp(c, argU(2, 8), argU(3, 20));
        } else if (mode == "facmp") {
            ok = caseFaCmp(c, argU(2, 512), argU(3, 0), argU(4, 20));
        } else if (mode == "fasrvcmp") {
            ok = caseFaSrvCmp(c, argU(2, 1), argU(3, 8191), argU(4, 0),
                              argU(5, 100));
        } else if (mode == "dnstepcmp") {
            ok = caseDnStep(c, argU(2, 1), argU(3, 400));
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
