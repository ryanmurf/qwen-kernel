// Stage-1-only Gemma 4 tensor-role map.  This intentionally owns no graph,
// cache, descriptors, or pipelines: later stages can consume the verified
// pointers without making the Qwen execution path guess Gemma semantics.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gguf.h"

struct Gemma4LayerWeights {
    bool sliding = true;
    const GgufTensor* attnK = nullptr;
    const GgufTensor* attnKNorm = nullptr;
    const GgufTensor* attnNorm = nullptr;
    const GgufTensor* attnOut = nullptr;
    const GgufTensor* attnQ = nullptr;
    const GgufTensor* attnQNorm = nullptr;
    const GgufTensor* attnV = nullptr;  // deliberately null on global layers
    const GgufTensor* ffnDown = nullptr;
    const GgufTensor* expertDownScale = nullptr;
    const GgufTensor* expertDown = nullptr;
    const GgufTensor* ffnGate = nullptr;
    const GgufTensor* routerScale = nullptr;
    const GgufTensor* router = nullptr;
    const GgufTensor* expertGateUp = nullptr;
    const GgufTensor* ffnNorm = nullptr;
    const GgufTensor* ffnUp = nullptr;
    const GgufTensor* layerOutputScale = nullptr;
    const GgufTensor* postAttentionNorm = nullptr;
    const GgufTensor* postFfwNorm = nullptr;
    const GgufTensor* postFfwNorm1 = nullptr;
    const GgufTensor* postFfwNorm2 = nullptr;
    const GgufTensor* preFfwNorm2 = nullptr;
};

class Gemma4Stage1Weights {
  public:
    static constexpr uint32_t kLayers = 30;
    static constexpr uint32_t kEmbedding = 2816;
    static constexpr uint32_t kVocabulary = 262144;

    Gguf gguf;
    const GgufTensor* tokenEmbedding = nullptr;  // tied Q6_K embedding / LM head
    const GgufTensor* outputNorm = nullptr;
    const GgufTensor* ropeFreqs = nullptr;
    std::array<Gemma4LayerWeights, kLayers> layers{};
    size_t mappedTextTensors = 0;
    size_t skippedVisionTensors = 0;
    uint64_t q4Bytes = 0;
    uint64_t q6Bytes = 0;
    uint64_t otherBytes = 0;

    bool open(const std::string& path, std::string& error) {
        if (!gguf.open(path)) return fail(error, "GGUF open failed");
        if (gguf.kvStr("general.architecture", "") != "gemma4")
            return fail(error, "general.architecture is not gemma4");
        if (gguf.kvInt("gemma4.block_count", 0) != kLayers)
            return fail(error, "gemma4.block_count is not 30");

        tokenEmbedding = require("token_embd.weight", GGML_Q6_K,
                                 {kEmbedding, kVocabulary}, error);
        outputNorm = require("output_norm.weight", GGML_F32, {kEmbedding}, error);
        ropeFreqs = require("rope_freqs.weight", GGML_F32, {256}, error);
        if (!tokenEmbedding || !outputNorm || !ropeFreqs) return false;

        for (uint32_t il = 0; il < kLayers; ++il) {
            auto& layer = layers[il];
            layer.sliding = !isGlobalLayer(il);
            uint64_t qWidth = layer.sliding ? 4096 : 8192;
            uint64_t kvWidth = layer.sliding ? 2048 : 1024;
            uint64_t headDim = layer.sliding ? 256 : 512;
            auto r = [&](const char* suffix, uint32_t type,
                         std::initializer_list<uint64_t> shape) {
                return require(layerName(il, suffix), type, shape, error);
            };
            layer.attnK = r("attn_k.weight", GGML_Q4_0, {kEmbedding, kvWidth});
            layer.attnKNorm = r("attn_k_norm.weight", GGML_F32, {headDim});
            layer.attnNorm = r("attn_norm.weight", GGML_F32, {kEmbedding});
            layer.attnOut = r("attn_output.weight", GGML_Q4_0, {qWidth, kEmbedding});
            layer.attnQ = r("attn_q.weight", GGML_Q4_0, {kEmbedding, qWidth});
            layer.attnQNorm = r("attn_q_norm.weight", GGML_F32, {headDim});
            if (layer.sliding)
                layer.attnV = r("attn_v.weight", GGML_Q4_0, {kEmbedding, kvWidth});
            else if (gguf.find(layerName(il, "attn_v.weight")))
                return fail(error, layerName(il, "attn_v.weight") +
                                   " must be absent on a global layer");
            layer.ffnDown = r("ffn_down.weight", GGML_Q4_0, {2112, kEmbedding});
            layer.expertDownScale = r("ffn_down_exps.scale", GGML_F32, {128});
            layer.expertDown = r("ffn_down_exps.weight", GGML_Q4_0,
                                 {704, kEmbedding, 128});
            layer.ffnGate = r("ffn_gate.weight", GGML_Q4_0, {kEmbedding, 2112});
            layer.routerScale = r("ffn_gate_inp.scale", GGML_F32, {kEmbedding});
            layer.router = r("ffn_gate_inp.weight", GGML_F32, {kEmbedding, 128});
            layer.expertGateUp = r("ffn_gate_up_exps.weight", GGML_Q4_0,
                                   {kEmbedding, 1408, 128});
            layer.ffnNorm = r("ffn_norm.weight", GGML_F32, {kEmbedding});
            layer.ffnUp = r("ffn_up.weight", GGML_Q4_0, {kEmbedding, 2112});
            layer.layerOutputScale = r("layer_output_scale.weight", GGML_F32, {1});
            layer.postAttentionNorm = r("post_attention_norm.weight", GGML_F32, {kEmbedding});
            layer.postFfwNorm = r("post_ffw_norm.weight", GGML_F32, {kEmbedding});
            layer.postFfwNorm1 = r("post_ffw_norm_1.weight", GGML_F32, {kEmbedding});
            layer.postFfwNorm2 = r("post_ffw_norm_2.weight", GGML_F32, {kEmbedding});
            layer.preFfwNorm2 = r("pre_ffw_norm_2.weight", GGML_F32, {kEmbedding});
            if (!error.empty()) return false;
        }

        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (const auto& [name, tensor] : gguf.tensors()) {
            if (claimed_.count(name) == 0) {
                if (isVisionTensor(name)) {
                    ++skippedVisionTensors;  // never mapped or uploaded by the text loader
                    continue;
                }
                return fail(error, "unmapped Gemma 4 text tensor: " + name);
            }
            if (!tensor.nbytes)
                return fail(error, "unrecognized encoded range for tensor: " + name);
            if (tensor.nDims == 0 || tensor.nDims > 4)
                return fail(error, "unsupported dimension count for tensor: " + name);
            if (tensor.type == GGML_Q4_0) q4Bytes += tensor.nbytes;
            else if (tensor.type == GGML_Q6_K) q6Bytes += tensor.nbytes;
            else otherBytes += tensor.nbytes;
            ranges.push_back({tensor.dataOffset, tensor.dataOffset + tensor.nbytes});
        }
        mappedTextTensors = claimed_.size();
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i)
            if (ranges[i].first < ranges[i - 1].second)
                return fail(error, "GGUF tensor payload ranges overlap");

        // Full-file encoded totals (distinct from the Stage-0 active/token
        // traffic, which counts only eight of 128 experts). These make Q4_0
        // block recognition and exact per-tensor range consumption executable.
        if (mappedTextTensors != 658)
            return fail(error, "target text tensor count is not the frozen 658");
        if (q4Bytes != 13771929600ull || q6Bytes != 605552640ull || otherBytes != 46056568ull)
            return fail(error, "encoded format byte totals differ from the Stage-0 ledger: Q4_0=" +
                               std::to_string(q4Bytes) + " Q6_K=" + std::to_string(q6Bytes) +
                               " other=" + std::to_string(otherBytes));
        return true;
    }

  private:
    std::set<std::string> claimed_;

    static bool fail(std::string& error, const std::string& message) {
        error = message;
        return false;
    }
    static bool isGlobalLayer(uint32_t il) {
        return il == 5 || il == 11 || il == 17 || il == 23 || il == 29;
    }
    static std::string layerName(uint32_t il, const char* suffix) {
        return "blk." + std::to_string(il) + "." + suffix;
    }
    static bool isVisionTensor(const std::string& name) {
        return name.rfind("v.", 0) == 0 || name.rfind("vision.", 0) == 0 ||
               name.rfind("mm.", 0) == 0 || name.rfind("mmproj.", 0) == 0 ||
               name.rfind("multi_modal_projector.", 0) == 0;
    }
    const GgufTensor* require(const std::string& name, uint32_t type,
                              std::initializer_list<uint64_t> shape,
                              std::string& error) {
        if (!error.empty()) return nullptr;
        const GgufTensor* tensor = gguf.find(name);
        if (!tensor) {
            fail(error, "missing tensor: " + name);
            return nullptr;
        }
        if (tensor->type != type || tensor->nDims != shape.size()) {
            fail(error, "type/dimension mismatch for tensor: " + name);
            return nullptr;
        }
        size_t dim = 0;
        for (uint64_t expected : shape) {
            if (tensor->ne[dim++] != expected) {
                fail(error, "shape mismatch for tensor: " + name);
                return nullptr;
            }
        }
        size_t rowBytes = ggmlRowBytes(type, tensor->ne[0]);
        uint64_t expectedBytes = rowBytes;
        for (uint32_t d = 1; d < tensor->nDims; ++d) expectedBytes *= tensor->ne[d];
        if (!rowBytes || tensor->nbytes != expectedBytes) {
            fail(error, "encoded byte-range mismatch for tensor: " + name);
            return nullptr;
        }
        claimed_.insert(name);
        return tensor;
    }
};
