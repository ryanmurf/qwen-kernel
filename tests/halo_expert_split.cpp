// Isolated real-weight Q5_K gate/up layout experiment. No serving integration.
// c++ -O2 -std=c++17 -pthread tests/halo_expert_split.cpp -lvulkan -o OUT
#define QK_LIBRARY
#include "../src/main.cpp"
#include <limits>

namespace {
struct GemmPC { uint32_t n_embd,n_ff,n_expert,n_used; };
#include "halo_gemm_operator_common.h"
constexpr uint32_t N=2560,F=640,E=32,U=10,B=N/256;
void layoutRequire(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
void putWord(float* p,size_t index,uint32_t value) { memcpy(p+index,&value,4); }
uint64_t weightHash(const std::vector<block_q5_K>& data) {
    uint64_t h=14695981039346656037ull;
    const auto* p=reinterpret_cast<const uint8_t*>(data.data());
    for(size_t i=0;i<data.size()*sizeof(block_q5_K);++i) { h^=p[i]; h*=1099511628211ull; }
    return h;
}
void pack(const std::vector<block_q5_K>& source,MappedBuffer& destination) {
    for(uint32_t tile=0;tile<E*F/128;++tile) for(uint32_t k=0;k<B;++k)
        for(uint32_t word=0;word<44;++word) for(uint32_t lane=0;lane<128;++lane) {
            const size_t from=(size_t(tile*128+lane)*B+k)*44+word;
            const size_t to=((size_t(tile)*B+k)*44+word)*128+lane;
            uint32_t value;
            memcpy(&value,reinterpret_cast<const uint8_t*>(source.data())+from*4,4);
            putWord(destination.ptr,to,value);
        }
    // Reverse mapping independently walks original block rows/words.
    for(uint32_t row=0;row<E*F;++row) for(uint32_t block=0;block<B;++block)
        for(uint32_t word=0;word<44;++word) {
            const size_t to=size_t(row/128)*B*44*128+block*44*128+word*128+row%128;
            const auto* from=reinterpret_cast<const uint8_t*>(source.data())+(size_t(row)*B*44+block*44+word)*4;
            layoutRequire(memcmp(from,destination.ptr+to,4)==0,"packed weight roundtrip failed");
        }
}
struct Case { uint32_t active,count; float scale; };
struct Gold { size_t index; double value; };
std::vector<Gold> gold(const Case& test,const std::vector<block_q5_K>& gate,
                       const std::vector<block_q5_K>& up,const float* x) {
    std::vector<Gold> result;
    std::vector<float> g(N),v(N);
    for(uint32_t e : {0u,test.active/2,test.active-1})
        for(uint32_t which : {0u,test.count-1}) for(uint32_t row : {0u,127u,128u,639u}) {
            uint32_t token=e*test.count+which,slot=token%U;
            dequant_row_q5_K(gate.data()+(e*F+row)*B,g.data(),N);
            dequant_row_q5_K(up.data()+(e*F+row)*B,v.data(),N);
            double a=0,b=0;
            for(uint32_t k=0;k<N;++k) { a+=double(g[k])*x[size_t(token)*N+k]; b+=double(v[k])*x[size_t(token)*N+k]; }
            result.push_back({(size_t(token)*(U+1)+slot)*F+row,(a/(1+std::exp(-a)))*b});
        }
    return result;
}
}

int main(int argc,char** argv) {
    try {
        layoutRequire(argc==2,"one existing GGUF model path required");
        Gguf model; layoutRequire(model.open(argv[1]),"model open failed");
        unsetenv("QK_DEVICE"); unsetenv("QK_DEVICE_NAME");
        setenv("QK_DEVICE_PCI","0000:c1:00.0",1);
        setenv("QK_OPERATOR_DEVICE_LOCAL","1",1);
        VkCtx c; initVk(c,argv[0]);
        layoutRequire(c.props.vendorID==0x1002 && c.props.deviceID==0x1586 &&
                      pciBdf(c.phys)=="0000:c1:00.0","Halo required");
        size_t cells=0;
        for(uint32_t layer : {0u,47u}) {
            std::vector<block_q5_K> gate(size_t(E)*F*B),up(gate.size());
            for(auto item : {std::make_pair("gate",&gate),std::make_pair("up",&up)}) {
                auto* tensor=model.find("blk."+std::to_string(layer)+".ffn_"+item.first+"_exps.weight");
                layoutRequire(tensor && tensor->type==GGML_Q5_K && tensor->ne[0]==N &&
                              tensor->ne[1]==F && tensor->ne[2]==512,"unexpected expert shape");
                for(uint32_t e=0;e<E;++e) {
                    const uint32_t sourceExpert=e*16+(layer?15:0);
                    memcpy(item.second->data()+size_t(e)*F*B,tensor->data+size_t(sourceExpert)*F*B*sizeof(block_q5_K),
                           size_t(F)*B*sizeof(block_q5_K));
                }
            }
            const size_t weightWords=gate.size()*sizeof(block_q5_K)/4;
            MappedBuffer g(c,weightWords),u(c,weightWords),gs(c,weightWords),us(c,weightWords);
            memcpy(g.ptr,gate.data(),weightWords*4); memcpy(u.ptr,up.data(),weightWords*4);
            pack(gate,gs); pack(up,us);
            printf("{\"type\":\"weights\",\"layer\":%u,\"experts\":%u,\"source_expert_stride\":16,\"source_expert_offset\":%u,\"gate_fnv64\":\"%016llx\",\"up_fnv64\":\"%016llx\",\"packed_bytes_per_matrix\":%zu,\"same_size\":true,\"roundtrip\":\"PASS\"}\n",
                   layer,E,layer?15:0,(unsigned long long)weightHash(gate),(unsigned long long)weightHash(up),weightWords*4);
            for(const Case test : {Case{32,1,1},Case{32,8,1},Case{32,16,1},Case{32,17,1},Case{1,64,8},Case{16,3,.25f}}) {
                const uint32_t T=test.active*test.count;
                const size_t outputFloats=size_t(T)*(U+1)*F+64;
                MappedBuffer x(c,size_t(T)*N),offsets(c,E+1),pairs(c,T),out(c,outputFloats,true);
                std::mt19937 rng(59270); std::uniform_real_distribution<float> uniform(-.1f,.1f);
                for(size_t i=0;i<size_t(T)*N;++i) x.ptr[i]=uniform(rng)*test.scale;
                for(uint32_t e=0;e<=E;++e) putWord(offsets.ptr,e,std::min(e,test.active)*test.count);
                for(uint32_t t=0;t<T;++t) putWord(pairs.ptr,t,(t<<4)|(t%U));
                const auto expected=gold(test,gate,up,x.ptr);
                Kernel baseline(c,"baseline.spv",{&g,&u,&x,&offsets,&pairs,&out},16);
                Kernel words(c,"soa.spv",{&gs,&us,&x,&offsets,&pairs,&out},16);
                Kernel soa(c,"split.spv",{&gs,&us,&x,&offsets,&pairs,&out},16);
                GemmPC pc{N,F,E,U};
                std::vector<float> reference;
                uint32_t order=0;
                for(uint32_t mode : {0u,1u,2u,2u,1u,0u}) {
                    Kernel& kernel=mode==0?baseline:mode==1?words:soa;
                    const char* name=mode==0?"baseline":mode==1?"soa":"split";
                    const float poison=std::numeric_limits<float>::quiet_NaN();
                    std::fill(out.ptr,out.ptr+outputFloats,poison);
                    timed(c,1,[&]{kernel.dispatch(pc,F/128,E,mode==2 ? (test.count+15)/16 : 1);});
                    const double usTime=timed(c,4,[&]{kernel.dispatch(pc,F/128,E,mode==2 ? (test.count+15)/16 : 1);});
                    size_t mismatch=0,nonfinite=0,padding=0,miss=0;
                    for(size_t i=0;i<outputFloats;++i) {
                        const size_t token=i/((U+1)*F),slot=(i/F)%(U+1);
                        const bool written=token<T && slot==token%U;
                        if(written) nonfinite+=!std::isfinite(out.ptr[i]);
                        else padding+=memcmp(out.ptr+i,&poison,4)!=0;
                        if(!reference.empty()) mismatch+=memcmp(out.ptr+i,reference.data()+i,4)!=0;
                    }
                    double worst=0;
                    for(auto item:expected) {
                        const double error=std::abs(double(out.ptr[item.index])-item.value);
                        worst=std::max(worst,error);
                        miss+=!std::isfinite(out.ptr[item.index]) || error>2e-4*(1+std::abs(item.value));
                    }
                    if(reference.empty()) reference.assign(out.ptr,out.ptr+outputFloats);
                    bool ok=!(mismatch||nonfinite||padding||miss);
                    printf("{\"type\":\"cell\",\"layer\":%u,\"active_experts\":%u,\"pairs_per_expert\":%u,\"q_scale\":%.2f,\"tokens\":%u,\"mode\":\"%s\",\"order\":%u,\"gpu_us\":%.6f,\"bit_mismatches\":%zu,\"nonfinite\":%zu,\"padding_changes\":%zu,\"fp64_misses\":%zu,\"fp64_max_abs\":%.9g,\"result\":\"%s\"}\n",
                           layer,test.active,test.count,test.scale,T,name,order++,usTime,mismatch,nonfinite,padding,miss,worst,ok?"PASS":"FAIL");
                    fflush(stdout); layoutRequire(ok,"expert layout correctness failure"); ++cells;
                }
            }
        }
        VK_CHECK(vkDeviceWaitIdle(c.dev));
        vkDestroyCommandPool(c.dev,c.pool,nullptr);
        vkDestroyDevice(c.dev,nullptr); vkDestroyInstance(c.inst,nullptr);
        printf("{\"type\":\"result\",\"result\":\"PASS\",\"cells\":%zu,\"full_model\":false}\n",cells);
    } catch(const std::exception& e) { fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
