// Optional correctness oracle, linked ONLY to a separately built reference
// llama.cpp fork. This is not a qk runtime backend or a native performance test.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct Capture {
    std::string target, output;
    bool found = false, written = false;
    bool dump = false;
    unsigned step = 0;
    std::map<std::string, unsigned> occurrences;
};
static bool capture(ggml_tensor* tensor, bool ask, void* opaque) {
    auto& c = *static_cast<Capture*>(opaque);
    const std::string name = ggml_get_name(tensor);
    const bool target = name == c.target;
    const bool inspect = c.dump && (name == "model.input_embed" ||
        (name.size() > 2 && name.substr(name.size()-2) == "-0"));
    if (!target && !inspect) return ask ? false : true;
    if (ask) return true;
    if (target) c.found = true;
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) return !target;
    std::vector<float> data(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    const auto path = target ? c.output : c.output + "." + name + "." + std::to_string(c.occurrences[name]++);
    std::ofstream out(path, std::ios::binary | (target && c.step ? std::ios::app : std::ios::trunc));
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    out.close(); if (target) c.written = bool(out);
    std::printf("reference captured %s: [%lld,%lld,%lld,%lld], %zu bytes\n", name.c_str(),
        (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2],
        (long long)tensor->ne[3], data.size() * sizeof(float));
    return !target;  // Intentional early cancellation after the requested layer.
}
int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::fprintf(stderr, "usage: flash-reference MODEL.gguf OUT.f32 [token=198] [tensor=l_last-0] [steps=1]\n");
        return 2;
    }
    Capture state{argc > 4 ? argv[4] : "l_last-0", argv[2]};
    state.dump = std::getenv("QK_REFERENCE_DUMP") != nullptr;
    llama_log_set([](ggml_log_level level, const char* message, void*) {
        if (level == GGML_LOG_LEVEL_ERROR) std::fputs(message, stderr);
    }, nullptr);
    ggml_backend_load_all(); llama_backend_init();
    ggml_backend_dev_t selected[] = {nullptr, nullptr};
    const char* prefix = std::getenv("QK_REFERENCE_GPU_PREFIX");
    const unsigned lastLayer = prefix ? std::atoi(prefix) : 0;
    if (lastLayer > 3) { std::fputs("reference prefix limited to the first four layers\n",stderr); return 2; }
    const bool gpu = prefix || std::getenv("QK_REFERENCE_GPU_LAYER0");
    if (gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (!std::strstr(ggml_backend_dev_description(dev), "STRIX_HALO")) continue;
            if (selected[0]) { std::fputs("ambiguous Halo device\n", stderr); return 2; }
            selected[0] = dev;
        }
        if (!selected[0]) { std::fputs("Halo device not found\n", stderr); return 2; }
        std::printf("reference: ONLY layers 0:%u weights on %s; all other weights CPU-mapped\n",
                    lastLayer+1,ggml_backend_dev_description(selected[0]));
    }
    const std::string layerPattern = "^blk\\.[0-"+std::to_string(lastLayer)+"]\\.";
    llama_model_tensor_buft_override overrides[] = {
        {layerPattern.c_str(), gpu ? ggml_backend_dev_buffer_type(selected[0]) : nullptr},
        {".*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}
    };
    auto mp = llama_model_default_params();
    mp.devices = selected; mp.n_gpu_layers = gpu ? -1 : 0;
    mp.tensor_buft_overrides = gpu ? overrides : nullptr;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.tensor_read_lazy = LLAMA_TENSOR_READ_LAZY_ON;
    mp.use_extra_bufts = false; mp.no_host = false; mp.load_mtp = false;
    mp.progress_callback = [](float, void*) { return true; };
    auto* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    llama_token token = argc > 3 ? std::atoi(argv[3]) : 198;
    const unsigned steps = argc > 5 ? std::atoi(argv[5]) : 1;
    if (!steps || steps > 32 || token < 0 || token + steps > unsigned(llama_vocab_n_tokens(llama_model_get_vocab(model)))) {
        llama_model_free(model); return 2;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = cp.n_ubatch = cp.n_seq_max = 1;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.offload_kqv = gpu; cp.op_offload = gpu;
    // Accuracy oracle: flash-attention may internally round Q/K even with
    // F32 cache storage. Compare the explicit F32 attention graph instead.
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.cb_eval = capture; cp.cb_eval_user_data = &state;
    auto* context = llama_init_from_model(model, cp);
    if (!context) { llama_model_free(model); return 1; }
    auto batch = llama_batch_init(1,0,1);
    bool ok = true;
    for (state.step = 0; state.step < steps; ++state.step) {
        state.found = state.written = false;
        batch.n_tokens = 1; batch.token[0] = token + state.step;
        batch.pos[0] = state.step; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
        int rc = llama_decode(context, batch);
        std::printf("reference step=%u decode_rc=%d (early cancellation expected), captured=%s\n",
                    state.step, rc, state.written ? "yes" : "no");
        if (!state.found || !state.written) { ok = false; break; }
    }
    llama_batch_free(batch);
    llama_free(context); llama_model_free(model); llama_backend_free();
    return ok ? 0 : 1;
}
