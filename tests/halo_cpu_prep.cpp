// Standalone exactness + preparation microbenchmark; no GPU/backend changes.
// g++ -O3 -std=c++17 -ffp-contract=off -pthread tests/halo_cpu_prep.cpp -o OUT
#include "halo_cpu_prep_simd.h"
#include "../src/qwen4_ple.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>

using Clock = std::chrono::steady_clock;
static void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
static uint64_t digest(const float* p,size_t n) {
    uint64_t h=14695981039346656037ull;
    for(size_t i=0;i<n;++i) {
        uint32_t bits; memcpy(&bits,p+i,4);
        for(int b=0;b<4;++b) { h^=(bits>>(8*b))&255; h*=1099511628211ull; }
    }
    return h;
}
template<class Block,class Ref,class Simd>
static void exact(const char* name,size_t blockWidth,Ref ref,Simd simd) {
    std::mt19937 rng(59270);
    size_t values=0;
    for(size_t count : {0u,1u,2u,5u,8u,10u,16u,33u}) {
        // Two-byte aligned input, deliberately not 16/64-byte aligned;
        // output starts one float after a canary.
        std::vector<uint16_t> storage((count*sizeof(Block)+2)/2+32);
        auto* blocks=reinterpret_cast<Block*>(storage.data()+1);
        for(int repeat=0;repeat<128;++repeat) {
            for(size_t b=0;b<count;++b) {
                auto* bytes=reinterpret_cast<uint8_t*>(blocks+b);
                for(size_t j=0;j<sizeof(Block);++j) bytes[j]=uint8_t(rng());
                // All finite half patterns, including subnormals/signed zero.
                for(size_t j=0;j<2;++j) {
                    uint16_t v=uint16_t(rng());
                    if((v&0x7c00)==0x7c00) v^=0x0400;
                    if(repeat<4) v=uint16_t(repeat==0 ? 0 : repeat==1 ? 0x8000 : repeat==2 ? 1 : 0x7bff);
                    memcpy(bytes+2*j,&v,2);
                }
            }
            const size_t n=count*blockWidth;
            std::vector<float> a(n+2,1234567.0f),b=a;
            ref(blocks,a.data()+1,n); simd(blocks,b.data()+1,n);
            require(memcmp(a.data(),b.data(),a.size()*4)==0,"SIMD differs or changed output guard");
            for(size_t i=1;i<=n;++i) require(std::isfinite(a[i]),"nonfinite exact-test output");
            values+=n;
        }
    }
    printf("{\"type\":\"exact\",\"format\":\"%s\",\"values\":%zu,\"result\":\"PASS\"}\n",name,values);
}

static void modelBench(const char* modelPath,const char* tokenPath) {
    Gguf model;
    require(model.open(modelPath),"model open failed");
    const auto* embedding=model.find("token_embd.weight");
    const auto* table=model.find("per_layer_token_embd.weight");
    require(embedding && embedding->type==GGML_Q5_K && embedding->ne[0]==2560,"unexpected embedding");
    require(table && table->type==GGML_Q5_1,"unexpected PLE table");
    auto config=Qwen4PleConfig::read(model); config.validate(table->ne[1]);
    require(config.width==160 && config.offsets.size()==16,"unexpected PLE width/heads");
    std::ifstream input(tokenPath); require(bool(input),"cannot read tokens");
    std::vector<uint32_t> tokens;
    uint64_t token;
    while(input>>token) {
        require(token<embedding->ne[1] && tokens.size()<2048,"token invalid or fixture too long");
        tokens.push_back(uint32_t(token));
    }
    require(input.eof() && tokens.size()==2048,"requires exactly 2048 token IDs");
    // This class configures bounded request-row prefetch, never whole-table warming.
    Qwen4PleLookup lookup(*table,config);
    const size_t eb=ggmlRowBytes(embedding->type,2560), pb=ggmlRowBytes(table->type,160);
    for(size_t T : {128u,512u,2048u}) {
        auto start=Clock::now();
        std::vector<uint32_t> history,rows;
        for(size_t t=0;t<T;++t) {
            auto current=config.rows(tokens[t],history); rows.insert(rows.end(),current.begin(),current.end());
            history.push_back(tokens[t]); if(history.size()>2) history.erase(history.begin());
        }
        double index=seconds(start);
        std::vector<float> gathered(T*2560),hidden(T*10240);
        start=Clock::now(); lookup.prefetch(rows); double prefetch=seconds(start);
        start=Clock::now(); lookup.gather(rows,gathered.data()); double gather=seconds(start);
        // Copy only the selected compressed rows; isolate arithmetic/layout from
        // page faults and the mutable cache. This is NOT full prefill throughput.
        std::vector<block_q5_1> ple(rows.size()*5);
        std::vector<block_q5_K> embeddings(T*10);
        for(size_t r=0;r<rows.size();++r) memcpy(ple.data()+r*5,table->data+size_t(rows[r])*pb,pb);
        for(size_t t=0;t<T;++t) memcpy(embeddings.data()+t*10,embedding->data+size_t(tokens[t])*eb,eb);
        printf("{\"type\":\"lookup\",\"tokens\":%zu,\"index_ms\":%.6f,\"prefetch_ms\":%.6f,\"gather_ms\":%.6f,\"hits\":%llu,\"misses\":%llu,\"cache_state\":\"uncontrolled file cache, cumulative lookup cache\"}\n",
               T,index*1000,prefetch*1000,gather*1000,(unsigned long long)lookup.hits,(unsigned long long)lookup.misses);
        uint64_t reference=0;
        std::vector<float> referenceHidden,referenceGathered;
        for(int iteration=0;iteration<8;++iteration) {
            const bool simd=iteration%4==1 || iteration%4==2;
            const int repeats=30;
            start=Clock::now();
            for(int rep=0;rep<repeats;++rep) {
                for(size_t t=0;t<T;++t) {
                    float* out=hidden.data()+t*10240;
                    if(simd) haloPrepQ5K(embeddings.data()+t*10,out,2560);
                    else dequant_row_q5_K(embeddings.data()+t*10,out,2560);
                    for(int h=1;h<4;++h) memcpy(out+h*2560,out,2560*4);
                }
                if(simd) haloPrepQ51(ple.data(),gathered.data(),T*2560);
                else dequant_row_q5_1(ple.data(),gathered.data(),T*2560);
            }
            const double ms=seconds(start)*1000/repeats;
            uint64_t hash=digest(hidden.data(),hidden.size())^digest(gathered.data(),gathered.size());
            if(iteration==0) { reference=hash; referenceHidden=hidden; referenceGathered=gathered; }
            require(hash==reference && memcmp(referenceHidden.data(),hidden.data(),hidden.size()*4)==0 &&
                    memcmp(referenceGathered.data(),gathered.data(),gathered.size()*4)==0,
                    "actual-model preparation differs");
            printf("{\"type\":\"prep\",\"tokens\":%zu,\"iteration\":%d,\"mode\":\"%s\",\"repeats\":%d,\"ms\":%.6f,\"output_fnv64\":\"%016llx\",\"result\":\"PASS\"}\n",
                   T,iteration,simd?"avx512":"portable",repeats,ms,(unsigned long long)hash);
            fflush(stdout);
        }
    }
}
int main(int argc,char** argv) {
    try {
        if(!haloPrepAvx512Available()) { fprintf(stderr,"AVX-512 unavailable; not run\n"); return 77; }
        require(argc==1 || argc==3,"usage: halo_cpu_prep [MODEL TOKENS]");
        exact<block_q5_1>("Q5_1",32,dequant_row_q5_1,haloPrepQ51);
        exact<block_q5_K>("Q5_K",256,dequant_row_q5_K,haloPrepQ5K);
        if(argc==3) modelBench(argv[1],argv[2]);
        puts("{\"type\":\"result\",\"result\":\"PASS\",\"gpu_used\":false,\"production_changed\":false}");
    } catch(const std::exception& e) { fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
