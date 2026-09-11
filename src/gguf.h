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
#include <stdexcept>
#include <string>
#include <vector>

enum GgmlType : uint32_t {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q5_1 = 7, GGML_Q8_0 = 8,
    GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13,
    GGML_Q6_K = 14, GGML_IQ3_XXS = 18, GGML_IQ4_XS = 23, GGML_BF16 = 30,
};

static inline const char* ggmlTypeName(uint32_t t) {
    switch (t) {
        case GGML_F32: return "F32";
        case GGML_F16: return "F16";
        case GGML_BF16: return "BF16";
        case GGML_Q4_0: return "Q4_0";
        case GGML_Q5_1: return "Q5_1";
        case GGML_Q5_K: return "Q5_K";
        case GGML_Q8_0: return "Q8_0";
        case GGML_Q6_K: return "Q6_K";
        case GGML_IQ3_XXS: return "IQ3_XXS";
        case GGML_IQ4_XS: return "IQ4_XS";
        default: return "other";
    }
}

// Encoded bytes per row, or 0 if the layout is unknown/misaligned.
// Recognizing a layout does not imply a GPU GEMV/serving implementation.
static inline size_t ggmlRowBytes(uint32_t type, uint64_t ne0) {
    switch (type) {
        case GGML_F32:  return ne0 * 4;
        case GGML_F16:  return ne0 * 2;
        case GGML_BF16: return ne0 * 2;
        case GGML_Q4_0: return ne0 / 32 * 18;
        case GGML_Q5_1: return ne0 % 32 == 0 ? ne0 / 32 * 24 : 0;
        case GGML_Q5_K: return ne0 % 256 == 0 ? ne0 / 256 * 176 : 0;
        case GGML_Q8_0: return ne0 / 32 * 34;
        case GGML_Q6_K: return ne0 / 256 * 210;
        case GGML_IQ4_XS: return ne0 / 256 * 136;
        case GGML_IQ3_XXS: return ne0 / 256 * 98;
        default:        return 0;
    }
}

struct GgufTensor {
    uint32_t type = 0;
    uint32_t nDims = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    uint64_t dataOffset = 0;       // absolute byte offset in the owning GGUF shard
    size_t nbytes = 0;             // exact encoded payload size when the type is recognized
    const uint8_t* data = nullptr;  // points into the mmap
};

class Gguf {
  public:
    Gguf() = default;
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;
    ~Gguf() { clear(); }
    // Accepts a single-file GGUF or the FIRST shard of an llama.cpp-style split
    // model ("<prefix>-00001-of-000NN.gguf", written by llama-gguf-split). For a
    // split model the shard count comes from the split.count KV; sibling shards
    // are derived from the filename and merged into one tensor directory. All
    // non-split KV metadata (arch, tokenizer, ...) lives in shard 1, so callers
    // that parse KVs from the given path keep working unchanged.
    bool open(const std::string& path) {
        clear();
        try {
        uint16_t splitCount = 0;
        if (!openOne(path, &splitCount)) return false;
        if (splitCount > 1) {
            const char* marker = "-00001-of-";
            size_t tp = path.rfind(marker);
            bool okName = tp != std::string::npos &&
                          path.size() == tp + 10 + 5 + 5 &&  // marker + "NNNNN" + ".gguf"
                          path.compare(path.size() - 5, 5, ".gguf") == 0;
            if (!okName) {
                fprintf(stderr, "gguf: split.count=%u but path is not a -00001-of-NNNNN.gguf first shard: %s\n",
                        splitCount, path.c_str());
                return false;
            }
            std::string prefix = path.substr(0, tp);
            for (uint16_t i = 2; i <= splitCount; i++) {
                char name[64];
                snprintf(name, sizeof name, "-%05u-of-%05u.gguf", i, splitCount);
                if (!openOne(prefix + name, nullptr)) return false;
            }
            if (splitTensorsTotal_ && tensors_.size() != (size_t)splitTensorsTotal_)
                fprintf(stderr, "gguf: split.tensors.count=%llu but merged %zu tensors\n",
                        (unsigned long long)splitTensorsTotal_, tensors_.size());
        }
        return true;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "gguf: %s\n", error.what());
            clear();
            return false;
        }
    }

    const GgufTensor* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

    const std::map<std::string, GgufTensor>& tensors() const { return tensors_; }
    const uint8_t* base() const { return base_; }
    size_t size() const { return size_; }

    // Scalar-int and string KVs captured during parse (shard 1 holds them all).
    uint64_t kvInt(const std::string& key, uint64_t def) const {
        auto it = kvInt_.find(key);
        return it == kvInt_.end() ? def : it->second;
    }
    std::string kvStr(const std::string& key, const std::string& def) const {
        auto it = kvStr_.find(key);
        return it == kvStr_.end() ? def : it->second;
    }
    double kvFloat(const std::string& key, double def) const {
        auto it = kvFloat_.find(key);
        return it == kvFloat_.end() ? def : it->second;
    }
    const std::vector<uint64_t>& kvInts(const std::string& key) const {
        static const std::vector<uint64_t> empty;
        auto it = kvInts_.find(key);
        return it == kvInts_.end() ? empty : it->second;
    }

  private:
    const uint8_t* base_ = nullptr;
    size_t size_ = 0, pos_ = 0;
    uint64_t alignment_ = 32;
    uint64_t splitTensorsTotal_ = 0;
    std::map<std::string, GgufTensor> tensors_;
    std::map<std::string, uint64_t> kvInt_;
    std::map<std::string, std::string> kvStr_;
    std::map<std::string, double> kvFloat_;
    std::map<std::string, std::vector<uint64_t>> kvInts_;
    std::vector<std::pair<const uint8_t*, size_t>> maps_;  // keeps every shard mapped

    void clear() {
        for (auto [ptr, length] : maps_) munmap((void*)ptr, length);
        maps_.clear(); tensors_.clear(); kvInt_.clear(); kvStr_.clear();
        kvFloat_.clear(); kvInts_.clear();
        base_ = nullptr; size_ = pos_ = splitTensorsTotal_ = 0;
    }

    void require(uint64_t bytes) const {
        if (pos_ > size_ || bytes > size_ - pos_)
            throw std::runtime_error("truncated GGUF metadata");
    }

    bool openOne(const std::string& path, uint16_t* splitCountOut) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { perror(path.c_str()); return false; }
        struct stat st;
        if (fstat(fd, &st) != 0) { ::close(fd); return false; }
        if (st.st_size < 24) { ::close(fd); return false; }
        size_ = (size_t)st.st_size;
        base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);  // the mapping holds its own reference
        if (base_ == MAP_FAILED) { perror("mmap"); return false; }
        maps_.emplace_back(base_, size_);
        return parse(splitCountOut);
    }

    template <typename T> T rd() {
        require(sizeof(T));
        T v;
        memcpy(&v, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string rdStr() {
        uint64_t n = rd<uint64_t>();
        require(n);
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
                require(n);
                if (et == 9) throw std::runtime_error("nested GGUF arrays unsupported");
                for (uint64_t i = 0; i < n; i++) skipValue(et);
                return 0;
            }
            case 10: case 11: case 12: return rd<uint64_t>();
            default: throw std::runtime_error("invalid GGUF metadata type");
        }
    }

    bool parse(uint16_t* splitCountOut) {
        if (size_ < 24 || memcmp(base_, "GGUF", 4) != 0) {
            fprintf(stderr, "not a GGUF file\n");
            return false;
        }
        pos_ = 4;
        alignment_ = 32;  // per-shard default; each shard may set its own
        uint32_t ver = rd<uint32_t>();
        if (ver < 2) { fprintf(stderr, "gguf v%u unsupported\n", ver); return false; }
        uint64_t nTensors = rd<uint64_t>();
        uint64_t nKv = rd<uint64_t>();
        if (nTensors > size_ / 24 || nKv > size_ / 13)
            throw std::runtime_error("invalid GGUF entry counts");
        for (uint64_t i = 0; i < nKv; i++) {
            std::string key = rdStr();
            uint32_t t = rd<uint32_t>();
            if (t == 8) { kvStr_[key] = rdStr(); continue; }
            if (t == 6) { kvFloat_[key] = rd<float>(); continue; }
            if (t == 12) { kvFloat_[key] = rd<double>(); continue; }
            if (t == 9 && key.compare(0, 10, "tokenizer.") != 0) {
                uint32_t et = rd<uint32_t>();
                uint64_t n = rd<uint64_t>();
                require(n);
                if (et == 9) throw std::runtime_error("nested GGUF arrays unsupported");
                const bool ints = et <= 5 || et == 7 || et == 10 || et == 11;
                auto& values = kvInts_[key];
                if (ints) values.reserve(n);
                for (uint64_t j = 0; j < n; ++j) {
                    uint64_t v = skipValue(et);
                    if (ints) values.push_back(v);
                }
                continue;
            }
            uint64_t v = skipValue(t);
            if (t <= 5 || t == 7 || t == 10 || t == 11) kvInt_[key] = v;  // scalar ints/bool, not f32/f64
            if (key == "general.alignment" && v) alignment_ = v;
            if (key == "split.count" && splitCountOut) *splitCountOut = (uint16_t)v;
            if (key == "split.tensors.count") splitTensorsTotal_ = v;
        }
        if (!alignment_ || alignment_ > (1u << 20) || (alignment_ & (alignment_ - 1)))
            throw std::runtime_error("invalid GGUF tensor alignment");
        struct Info { std::string name; GgufTensor t; uint64_t off; };
        std::vector<Info> infos(nTensors);
        for (auto& inf : infos) {
            inf.name = rdStr();
            uint32_t nd = rd<uint32_t>();
            if (nd < 1 || nd > 4) throw std::runtime_error("invalid GGUF tensor rank");
            inf.t.nDims = nd;
            for (uint32_t d = 0; d < nd; d++) {
                uint64_t ne = rd<uint64_t>();
                if (!ne || ne > UINT64_MAX / 4) throw std::runtime_error("invalid GGUF tensor dimension");
                if (d < 4) inf.t.ne[d] = ne;
            }
            inf.t.type = rd<uint32_t>();
            inf.off = rd<uint64_t>();
        }
        uint64_t dataStart = (pos_ + alignment_ - 1) / alignment_ * alignment_;
        for (auto& inf : infos) {
            uint64_t absolute = dataStart + inf.off;
            if (absolute < dataStart || absolute > size_) {
                fprintf(stderr, "gguf: tensor %s offset is outside the file\n", inf.name.c_str());
                return false;
            }
            // Only publish an exact encoded range when every dimension is
            // represented in GgufTensor::ne.  Higher-dimensional tensors are
            // still parsed, but callers must reject them before range use.
            size_t rowBytes = inf.t.nDims <= 4
                            ? ggmlRowBytes(inf.t.type, inf.t.ne[0]) : 0;
            if (rowBytes) {
                uint64_t total = rowBytes;
                for (uint32_t d = 1; d < inf.t.nDims && d < 4; d++) {
                    if (inf.t.ne[d] && total > UINT64_MAX / inf.t.ne[d]) {
                        fprintf(stderr, "gguf: tensor %s byte size overflows\n", inf.name.c_str());
                        return false;
                    }
                    total *= inf.t.ne[d];
                }
                if (total > size_ - absolute) {
                    fprintf(stderr, "gguf: tensor %s payload extends past EOF\n", inf.name.c_str());
                    return false;
                }
                inf.t.nbytes = (size_t)total;
            }
            inf.t.dataOffset = absolute;
            inf.t.data = base_ + absolute;
            tensors_[inf.name] = inf.t;
        }
        return true;
    }
};
