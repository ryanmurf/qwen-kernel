// Model-free same-order load scheduling experiment; not serving throughput.
// Compile: c++ -O2 -std=c++17 -pthread tests/halo_attn_loads_operator.cpp -lvulkan -o OUT
// Four modes ABCDDCBA; explicit immutable shader set and exclusive Halo required.
// Run only after full-model A/B has stopped and the Halo has drained.
#define QK_LIBRARY
#include "/home/ryan/IdeaProjects/qwen-kernel-strix-halo/src/main.cpp"
#include <limits>

namespace {
struct GemmPC {
    uint32_t pos, tmax, dh, nRot, hQ, hKV;
    float eps, freqBase;
};
static_assert(sizeof(GemmPC) == 32, "decode attention push constants");
#include "/home/ryan/IdeaProjects/qwen-kernel-strix-halo/tests/halo_gemm_operator_common.h"
constexpr uint32_t capacity=32768, heads=24, kvHeads=2, dim=256, slots=2;
constexpr size_t rowFloats=heads*dim, cacheFloats=size_t(kvHeads)*capacity*dim;
constexpr size_t outFloats=slots*rowFloats+64;
struct Case { uint32_t n0, n1=0; float qScale=1; };
struct Gold { uint32_t rq, head; std::vector<double> values; };

std::vector<Gold> reference(const Case& test, const float* q, const float* k,
                            const float* v, const float* gates) {
    std::vector<Gold> result;
    for (uint32_t rq=0; rq<(test.n1?2u:1u); ++rq) {
        const uint32_t n=rq?test.n1:test.n0;
        for(uint32_t h:{0u,11u,12u,23u}) {
            const uint32_t kv=h/(heads/kvHeads);
            std::vector<double> scores(n), values(dim,0);
            double maximum=-std::numeric_limits<double>::infinity();
            for(uint32_t t=0;t<n;++t) {
                double score=0;
                for(uint32_t d=0;d<dim;++d)
                    score+=double(q[rq*rowFloats+h*dim+d])*k[rq*cacheFloats+(size_t(kv)*capacity+t)*dim+d];
                scores[t]=score/std::sqrt(double(dim));
                maximum=std::max(maximum,scores[t]);
            }
            double sum=0;
            for(auto& score:scores) { score=std::exp(score-maximum); sum+=score; }
            for(uint32_t t=0;t<n;++t) for(uint32_t d=0;d<dim;++d)
                values[d]+=scores[t]*v[rq*cacheFloats+(size_t(kv)*capacity+t)*dim+d];
            for(uint32_t d=0;d<dim;++d)
                values[d]/=sum*(1+std::exp(-double(gates[rq*rowFloats*2+h*dim*2+dim+d])));
            result.push_back({rq,h,std::move(values)});
        }
    }
    return result;
}

bool check(const Case& test, const float* output, const std::vector<float>& baseline,
           const std::vector<Gold>& expected, float poison, const char* mode,
           uint32_t order, double us) {
    size_t nonfinite=0,padding=0,mismatch=0,miss=0;
    double worst=0;
    const size_t live=(test.n1?2u:1u)*rowFloats;
    for(size_t i=0;i<outFloats;++i) {
        if(i<live) nonfinite+=!std::isfinite(output[i]);
        else padding+=memcmp(output+i,&poison,4)!=0;
        if(!baseline.empty()) mismatch+=memcmp(output+i,baseline.data()+i,4)!=0;
    }
    for(const auto& gold:expected) for(uint32_t d=0;d<dim;++d) {
        double value=output[gold.rq*rowFloats+gold.head*dim+d];
        double error=std::abs(value-gold.values[d]);
        worst=std::max(worst,error);
        miss+=!std::isfinite(value)||error>2e-5*(1+std::abs(gold.values[d]));
    }
    bool pass=nonfinite==0&&padding==0&&mismatch==0&&miss==0;
    printf("{\"mode\":\"%s\",\"order\":%u,\"n0\":%u,\"n1\":%u,\"q_scale\":%.1f,"
           "\"gpu_us\":%.6f,\"nonfinite\":%zu,\"padding_changes\":%zu,"
           "\"baseline_bit_mismatches\":%zu,\"fp64_misses\":%zu,\"fp64_max_abs\":%.9g,\"result\":\"%s\"}\n",
           mode,order,test.n0,test.n1,test.qScale,us,nonfinite,padding,mismatch,miss,worst,pass?"PASS":"FAIL");
    fflush(stdout);
    return pass;
}
}

int main(int argc,char** argv) {
    uint32_t repeats=4;
    if(argc>2) return 2;
    if(argc==2) {
        char* end=nullptr; long value=strtol(argv[1],&end,10);
        if(end==argv[1]||*end||value<1||value>16) return 2;
        repeats=uint32_t(value);
    }
    unsetenv("QK_DEVICE"); unsetenv("QK_DEVICE_PCI");
    setenv("QK_DEVICE_NAME","STRIX_HALO",1);
    setenv("QK_OPERATOR_DEVICE_LOCAL","1",1);
    VkCtx c; initVk(c,argv[0]);
    if(c.props.vendorID!=0x1002||c.props.deviceID!=0x1586||pciBdf(c.phys)!="0000:c1:00.0") return 2;
    bool pass=true;
    {
        MappedBuffer q(c,slots*rowFloats),k(c,slots*cacheFloats),v(c,slots*cacheFloats),
                     gates(c,slots*rowFloats*2),position(c,slots),output(c,outFloats,true);
        printf("decode operator memory: %s; staged=%s; repetitions=%u; synthetic only\n",
               memTypeDesc(c,k.b.memType).c_str(),k.staged?"yes":"no",repeats); fflush(stdout);
        Kernel base(c,"baseline.spv",{&q,&k,&v,&gates,&output,&position},32);
        Kernel kv8(c,"kv8.spv",{&q,&k,&v,&gates,&output,&position},32);
        Kernel kv16(c,"kv16.spv",{&q,&k,&v,&gates,&output,&position},32);
        Kernel kv32(c,"kv32.spv",{&q,&k,&v,&gates,&output,&position},32);
        std::mt19937 rng(1729); std::uniform_real_distribution<float> uniform(-1,1);
        std::vector<float> goodQ(q.b.size/4),goodK(k.b.size/4),goodV(v.b.size/4),goodG(gates.b.size/4);
        for(auto* data:{&goodQ,&goodK,&goodV}) for(float& x:*data) x=uniform(rng);
        for(float& x:goodG) x=20*uniform(rng);
        const float poison=std::numeric_limits<float>::quiet_NaN();
        for(const Case test:std::vector<Case>{{1},{2},{63},{64},{65},{255},{256},{257},
                {1025},{2048},{8192},{16384},{16433},{32768},{257,16433},{1025,0,16},{257,0,128},{127}}) {
            std::fill(q.ptr,q.ptr+goodQ.size(),poison);
            std::fill(gates.ptr,gates.ptr+goodG.size(),poison);
            std::fill(k.ptr,k.ptr+goodK.size(),poison);
            std::fill(v.ptr,v.ptr+goodV.size(),poison);
            const uint32_t active=test.n1?2:1;
            for(uint32_t rq=0;rq<slots;++rq) {
                const uint32_t n=rq?test.n1:test.n0;
                const uint32_t pos=n?n-1:capacity+123;
                memcpy(position.ptr+rq,&pos,4);
                if(rq>=active) continue;
                for(size_t i=rq*rowFloats;i<(rq+1)*rowFloats;++i) q.ptr[i]=goodQ[i]*test.qScale;
                memcpy(gates.ptr+rq*rowFloats*2,goodG.data()+rq*rowFloats*2,rowFloats*8);
                for(uint32_t kv=0;kv<kvHeads;++kv) {
                    const size_t off=rq*cacheFloats+size_t(kv)*capacity*dim;
                    memcpy(k.ptr+off,goodK.data()+off,size_t(n)*dim*4);
                    memcpy(v.ptr+off,goodV.data()+off,size_t(n)*dim*4);
                }
            }
            auto gold=reference(test,q.ptr,k.ptr,v.ptr,gates.ptr);
            GemmPC pc{0,capacity,dim,64,heads,kvHeads,1e-6f,1e7f};
            std::vector<float> original;
            uint32_t order=0;
            for(uint32_t mode:{0u,1u,2u,3u,3u,2u,1u,0u}) {
                Kernel& kernel=mode==0?base:(mode==1?kv8:(mode==2?kv16:kv32));
                auto emit=[&]{kernel.dispatch(pc,heads,1,active);};
                std::fill(output.ptr,output.ptr+outFloats,poison);
                timed(c,1,emit);
                std::fill(output.ptr,output.ptr+outFloats,poison);
                double us=timed(c,repeats,emit);
                pass&=check(test,output.ptr,original,gold,poison,
                           mode==0?"baseline":(mode==1?"kv8":(mode==2?"kv16":"kv32")),order++,us);
                if(original.empty()) original.assign(output.ptr,output.ptr+outFloats);
            }
        }
    }
    VK_CHECK(vkDeviceWaitIdle(c.dev));
    vkDestroyCommandPool(c.dev,c.pool,nullptr);
    vkDestroyDevice(c.dev,nullptr); vkDestroyInstance(c.inst,nullptr);
    return pass?0:1;
}
