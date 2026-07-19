#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Semantic contract shared by the Stage 2--4 harness and the future Stage 5
// serial graph.  Keep architecture constants and buffer layouts here rather
// than scattering Gemma-specific values through the Qwen engine.
namespace gemma4 {

constexpr uint32_t kEmbedding = 2816;
constexpr uint32_t kLayers = 30;
constexpr uint32_t kExperts = 128;
constexpr uint32_t kExpertsUsed = 8;
constexpr uint32_t kSharedFf = 2112;
constexpr uint32_t kExpertFf = 704;
constexpr uint32_t kSlidingWindow = 1024;
constexpr float kRmsEpsilon = 1.0e-6f;
constexpr float kSoftcap = 30.0f;

struct AttentionConfig {
    uint32_t headDim;
    uint32_t queryHeads;
    uint32_t kvHeads;
    uint32_t ropeDim;
    uint32_t cacheLength;
    float ropeBase;
    bool sliding;
};

constexpr AttentionConfig kSlidingAttention{
    256, 16, 8, 256, kSlidingWindow, 10000.0f, true};
constexpr AttentionConfig kGlobalAttention{
    512, 16, 2, 512, 262144, 1000000.0f, false};

constexpr bool isGlobalLayer(uint32_t layer) {
    return layer == 5 || layer == 11 || layer == 17 || layer == 23 || layer == 29;
}

constexpr AttentionConfig attentionConfig(uint32_t layer) {
    return isGlobalLayer(layer) ? kGlobalAttention : kSlidingAttention;
}

// Exactly 64 bytes in both C++ and std430 GLSL.  Expert ranks stay in
// descending router order; ties use the lower physical expert id.
struct alignas(16) Selection {
    uint32_t ids[kExpertsUsed];
    float weights[kExpertsUsed];
};
static_assert(sizeof(Selection) == 64, "Gemma 4 Selection ABI must be 64 bytes");
static_assert(alignof(Selection) == 16, "Gemma 4 Selection ABI must be 16-byte aligned");

inline float gelu(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoef = 0.044715f;
    return 0.5f*x*(1.0f + std::tanh(kSqrt2OverPi*x*(1.0f + kCoef*x*x)));
}

inline void rmsNorm(const float* x, const float* weight, float* out,
                    uint32_t n, float eps = kRmsEpsilon, float postScale = 1.0f) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) sum += x[i]*x[i];
    const float scale = postScale/std::sqrt(sum/(float)n + eps);
    if (weight) {
        for (uint32_t i = 0; i < n; ++i) out[i] = x[i]*scale*weight[i];
    } else {
        for (uint32_t i = 0; i < n; ++i) out[i] = x[i]*scale;
    }
}

inline Selection stableTop8(const float* logits) {
    std::array<uint32_t, kExperts> order{};
    for (uint32_t i = 0; i < kExperts; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        const float av = std::isfinite(logits[a]) ? logits[a]
                                                  : -std::numeric_limits<float>::infinity();
        const float bv = std::isfinite(logits[b]) ? logits[b]
                                                  : -std::numeric_limits<float>::infinity();
        return av > bv || (av == bv && a < b);
    });
    Selection result{};
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t rank = 0; rank < kExpertsUsed; ++rank) {
        result.ids[rank] = order[rank];
        const float value = logits[order[rank]];
        if (rank == 0) maximum = value;
        result.weights[rank] = std::exp(value - maximum);
    }
    float sum = 0.0f;
    for (float value : result.weights) sum += value;
    for (float& value : result.weights) value /= sum;
    return result;
}

inline float ropeFrequency(uint32_t pair, uint32_t ropeDim, float base,
                           const float* factors) {
    const float theta = std::pow(base, -2.0f*(float)pair/(float)ropeDim);
    return theta/(factors ? factors[pair] : 1.0f);
}

inline uint32_t cacheSlot(const AttentionConfig& cfg, uint32_t absolutePosition) {
    return cfg.sliding ? absolutePosition % kSlidingWindow : absolutePosition;
}

inline uint32_t attentionStart(const AttentionConfig& cfg, uint32_t position) {
    return cfg.sliding && position + 1 > kSlidingWindow
        ? position + 1 - kSlidingWindow : 0;
}

}  // namespace gemma4
