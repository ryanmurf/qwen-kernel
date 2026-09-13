// CPU-only guard-page and bit-exact validation for the experimental SIMD path.
// Build with -ffp-contract=off, optionally -fsanitize=address,undefined.
#include "halo_cpu_prep_simd.h"
#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>
#include <vector>
#include <cstdio>

struct Guarded {
    void* mapping;
    size_t total;
    uint8_t* data;
    explicit Guarded(size_t bytes) {
        const size_t page=size_t(sysconf(_SC_PAGESIZE));
        const size_t middle=((bytes+page-1)/page)*page;
        total=middle+2*page;
        mapping=mmap(nullptr,total,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(mapping==MAP_FAILED) throw std::runtime_error("mmap failed");
        auto* start=static_cast<uint8_t*>(mapping)+page;
        if(mprotect(start,middle,PROT_READ|PROT_WRITE)!=0) throw std::runtime_error("mprotect failed");
        data=start+middle-bytes;
    }
    ~Guarded() { munmap(mapping,total); }
    Guarded(const Guarded&)=delete;
};
template<class Block,class Ref,class Simd>
void check(size_t width,Ref ref,Simd simd) {
    simd(nullptr,nullptr,0);
    for(size_t count:{1u,2u,5u,10u}) {
        Guarded in(count*sizeof(Block)),out(count*width*sizeof(float));
        auto* blocks=reinterpret_cast<Block*>(in.data);
        for(size_t i=0;i<count*sizeof(Block);++i) in.data[i]=uint8_t((i*37+19)&255);
        for(size_t i=0;i<count;++i) {
            const uint16_t values[]={0x3555,0x8011};
            memcpy(in.data+i*sizeof(Block),values,4);
        }
        std::vector<float> reference(count*width);
        ref(blocks,reference.data(),count*width);
        simd(blocks,reinterpret_cast<float*>(out.data),count*width);
        if(memcmp(reference.data(),out.data,count*width*4)) throw std::runtime_error("guarded SIMD mismatch");
    }
}
int main() {
    if(!haloPrepAvx512Available()) return 77;
    check<block_q5_1>(32,dequant_row_q5_1,haloPrepQ51);
    check<block_q5_K>(256,dequant_row_q5_K,haloPrepQ5K);
    puts("Q5_1/Q5_K protected-page tails, zero length and exact outputs: PASS");
}
