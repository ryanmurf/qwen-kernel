// Minimal read-only GGUF v3 reader: mmaps the file and exposes tensor
// name -> {ggml type, shape, data pointer}. Only what the harness needs.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

enum GgmlType : uint32_t {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8,
    GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13,
    GGML_Q6_K = 14, GGML_IQ3_XXS = 18, GGML_IQ4_XS = 23, GGML_BF16 = 30,
};

static inline const char* ggmlTypeName(uint32_t t) {
    switch (t) {
        case GGML_F32: return "F32";
        case GGML_F16: return "F16";
        case GGML_Q8_0: return "Q8_0";
        case GGML_Q6_K: return "Q6_K";
        case GGML_IQ3_XXS: return "IQ3_XXS";
        case GGML_IQ4_XS: return "IQ4_XS";
        default: return "other";
    }
}

// bytes per row of ne0 elements, or 0 if unsupported for GEMV here
static inline size_t ggmlRowBytes(uint32_t type, uint64_t ne0) {
    switch (type) {
        case GGML_F32:  return ne0 * 4;
        case GGML_F16:  return ne0 * 2;
        case GGML_Q8_0: return ne0 / 32 * 34;
        case GGML_Q6_K: return ne0 / 256 * 210;
        case GGML_IQ4_XS: return ne0 / 256 * 136;
        case GGML_IQ3_XXS: return ne0 / 256 * 98;
        default:        return 0;
    }
}

struct GgufTensor {
    uint32_t type = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    const uint8_t* data = nullptr;  // points into the mmap
};

class Gguf {
  public:
    bool open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) { perror(path.c_str()); return false; }
        struct stat st;
        fstat(fd_, &st);
        size_ = (size_t)st.st_size;
        base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (base_ == MAP_FAILED) { perror("mmap"); return false; }
        return parse();
    }

    const GgufTensor* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

    const std::map<std::string, GgufTensor>& tensors() const { return tensors_; }
    const uint8_t* base() const { return base_; }
    size_t size() const { return size_; }

  private:
    const uint8_t* base_ = nullptr;
    size_t size_ = 0, pos_ = 0;
    int fd_ = -1;
    uint64_t alignment_ = 32;
    std::map<std::string, GgufTensor> tensors_;

    template <typename T> T rd() {
        T v;
        memcpy(&v, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string rdStr() {
        uint64_t n = rd<uint64_t>();
        std::string s((const char*)base_ + pos_, n);
        pos_ += n;
        return s;
    }
    // returns u64 value for scalar ints (used for general.alignment), else 0
    uint64_t skipValue(uint32_t t) {
        switch (t) {
            case 0: case 1: case 7: { uint8_t v = rd<uint8_t>(); return v; }
            case 2: case 3: { uint16_t v = rd<uint16_t>(); return v; }
            case 4: case 5: case 6: { uint32_t v = rd<uint32_t>(); return v; }
            case 8: rdStr(); return 0;
            case 9: {
                uint32_t et = rd<uint32_t>();
                uint64_t n = rd<uint64_t>();
                for (uint64_t i = 0; i < n; i++) skipValue(et);
                return 0;
            }
            case 10: case 11: case 12: return rd<uint64_t>();
            default: fprintf(stderr, "gguf: bad kv type %u\n", t); exit(1);
        }
    }

    bool parse() {
        if (size_ < 24 || memcmp(base_, "GGUF", 4) != 0) {
            fprintf(stderr, "not a GGUF file\n");
            return false;
        }
        pos_ = 4;
        uint32_t ver = rd<uint32_t>();
        if (ver < 2) { fprintf(stderr, "gguf v%u unsupported\n", ver); return false; }
        uint64_t nTensors = rd<uint64_t>();
        uint64_t nKv = rd<uint64_t>();
        for (uint64_t i = 0; i < nKv; i++) {
            std::string key = rdStr();
            uint32_t t = rd<uint32_t>();
            uint64_t v = skipValue(t);
            if (key == "general.alignment" && v) alignment_ = v;
        }
        struct Info { std::string name; GgufTensor t; uint64_t off; };
        std::vector<Info> infos(nTensors);
        for (auto& inf : infos) {
            inf.name = rdStr();
            uint32_t nd = rd<uint32_t>();
            for (uint32_t d = 0; d < nd && d < 4; d++) inf.t.ne[d] = rd<uint64_t>();
            inf.t.type = rd<uint32_t>();
            inf.off = rd<uint64_t>();
        }
        uint64_t dataStart = (pos_ + alignment_ - 1) / alignment_ * alignment_;
        for (auto& inf : infos) {
            inf.t.data = base_ + dataStart + inf.off;
            tensors_[inf.name] = inf.t;
        }
        return true;
    }
};
