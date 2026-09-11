#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/gguf.h"
#include <cassert>
#include <vector>

template<class T> static void put(std::vector<uint8_t>& out, T value) {
    auto* p = reinterpret_cast<const uint8_t*>(&value); out.insert(out.end(), p, p + sizeof(T));
}
static void str(std::vector<uint8_t>& out, const std::string& s) {
    put<uint64_t>(out, s.size()); out.insert(out.end(), s.begin(), s.end());
}
static bool readFixture(Gguf& g, const std::vector<uint8_t>& bytes) {
    char path[] = "/tmp/qk-gguf-test-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    assert(write(fd, bytes.data(), bytes.size()) == (ssize_t)bytes.size());
    close(fd);
    bool ok = g.open(path);
    unlink(path);
    return ok;
}
int main() {
    std::vector<uint8_t> bytes{'G','G','U','F'};
    put<uint32_t>(bytes, 3); put<uint64_t>(bytes, 0); put<uint64_t>(bytes, 4);
    str(bytes, "general.architecture"); put<uint32_t>(bytes, 8); str(bytes, "qwen4exp");
    str(bytes, "qwen4exp.epsilon"); put<uint32_t>(bytes, 6); put<float>(bytes, 0.25f);
    str(bytes, "qwen4exp.array"); put<uint32_t>(bytes, 9); put<uint32_t>(bytes, 10);
    put<uint64_t>(bytes, 3);
    for (uint64_t v : std::vector<uint64_t>{1, 23703573157769ull, UINT64_MAX}) put(bytes, v);
    str(bytes, "qwen4exp.count"); put<uint32_t>(bytes, 4); put<uint32_t>(bytes, 48);
    Gguf g;
    assert(readFixture(g, bytes));
    assert(g.kvStr("general.architecture", "") == "qwen4exp");
    assert(g.kvFloat("qwen4exp.epsilon", 0) == 0.25);
    assert(g.kvInt("qwen4exp.count", 0) == 48);
    assert(g.kvInts("qwen4exp.array") == std::vector<uint64_t>({1,23703573157769ull,UINT64_MAX}));
    for (size_t n = 0; n < bytes.size(); ++n)
        assert(!readFixture(g, std::vector<uint8_t>(bytes.begin(), bytes.begin()+n)));
    assert(readFixture(g, bytes));  // reuse after a failed/truncated open
    std::puts("GGUF float/uint64-array metadata, truncation checks and reader reuse: PASS");
}
