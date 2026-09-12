// Standalone model-free batch-attention experiment; an exclusive Halo window
// is required. See bench/RESULTS-halo-batch-attn-vec4.md for reproduction.
#define QK_LIBRARY
#include "../src/main.cpp"
#include <limits>
#include <set>

namespace {
struct GemmPC {
    uint32_t tmax, dh, nRot, hQ, hKV;
    float eps, freqBase;
    uint32_t base, Tn, qbase;
};
static_assert(sizeof(GemmPC) == 40, "batch attention push constants");
#include "halo_gemm_operator_common.h"

constexpr uint32_t capacity=32768, heads=24, kvHeads=2, dim=256, maxRows=512;
constexpr size_t rowFloats=size_t(heads)*dim, outFloats=maxRows*rowFloats+64;
struct Case { uint32_t base, rows, first, tile; float qScale=1; };
struct ReferenceRow { uint32_t query, head; std::vector<double> values; };

std::vector<ReferenceRow> reference(const Case& test, const float* q, const float* k,
                                   const float* v, const float* gates) {
    std::vector<ReferenceRow> out;
    const uint32_t last=std::min(test.rows, test.first+test.tile)-1;
    for (uint32_t query : std::set<uint32_t>{test.first, last}) {
        const uint32_t n=test.base+query+1;
        for (uint32_t h : {0u,11u,12u,23u}) {
            const uint32_t kv=h/(heads/kvHeads);
            std::vector<double> scores(n), values(dim,0);
            double maxScore=-std::numeric_limits<double>::infinity();
            for (uint32_t t=0; t<n; ++t) {
                double dot=0;
                for (uint32_t d=0; d<dim; ++d)
                    dot+=double(q[size_t(query)*rowFloats+h*dim+d])*k[(size_t(kv)*capacity+t)*dim+d];
                scores[t]=dot/std::sqrt(double(dim));
                maxScore=std::max(maxScore,scores[t]);
            }
            double denominator=0;
            for (double& s:scores) { s=std::exp(s-maxScore); denominator+=s; }
            for (uint32_t t=0; t<n; ++t)
                for (uint32_t d=0; d<dim; ++d)
                    values[d]+=scores[t]*v[(size_t(kv)*capacity+t)*dim+d];
            for (uint32_t d=0; d<dim; ++d)
                values[d]/=denominator*(1+std::exp(-double(gates[size_t(query)*rowFloats*2+h*dim*2+dim+d])));
            out.push_back({query,h,std::move(values)});
        }
    }
    return out;
}

bool check(const Case& test, const float* actual, const std::vector<float>& baseline,
           const std::vector<ReferenceRow>& expected, float poison,
           const char* mode, uint32_t order, double us) {
    size_t nonfinite=0, padChanges=0, mismatches=0, fp64miss=0;
    double worst=0, sq=0, norm=0;
    const size_t start=size_t(test.first)*rowFloats;
    const size_t end=size_t(std::min(test.rows,test.first+test.tile))*rowFloats;
    for (size_t i=0; i<outFloats; ++i) {
        if (i>=start && i<end) nonfinite+=!std::isfinite(actual[i]);
        else padChanges+=memcmp(actual+i,&poison,4)!=0;
        if (!baseline.empty()) mismatches+=memcmp(actual+i,baseline.data()+i,4)!=0;
    }
    for (const auto& ref:expected) for (uint32_t d=0; d<dim; ++d) {
        double value=actual[size_t(ref.query)*rowFloats+ref.head*dim+d];
        double error=std::abs(value-ref.values[d]);
        worst=std::max(worst,error); sq+=error*error; norm+=ref.values[d]*ref.values[d];
        fp64miss+=!std::isfinite(value)||error>2e-5*(1+std::abs(ref.values[d]));
    }
    bool pass=nonfinite==0&&padChanges==0&&mismatches==0&&fp64miss==0;
    printf("{\"mode\":\"%s\",\"order\":%u,\"base\":%u,\"rows\":%u,\"qbase\":%u,\"tile\":%u,\"q_scale\":%.1f,"
           "\"gpu_us\":%.6f,\"nonfinite\":%zu,\"padding_changes\":%zu,\"baseline_bit_mismatches\":%zu,"
           "\"fp64_misses\":%zu,\"fp64_max_abs\":%.9g,\"fp64_relative_rms\":%.9g,\"result\":\"%s\"}\n",
           mode,order,test.base,test.rows,test.first,test.tile,test.qScale,us,nonfinite,padChanges,mismatches,
           fp64miss,worst,std::sqrt(sq/std::max(norm,1e-30)),pass?"PASS":"FAIL");
    fflush(stdout);
    return pass;
}
}

int main(int argc, char** argv) {
    uint32_t repeats=4;
    if (argc>2) return 2;
    if (argc==2) {
        char* end=nullptr; long n=strtol(argv[1],&end,10);
        if (end==argv[1]||*end||n<1||n>16) return 2;
        repeats=uint32_t(n);
    }
    unsetenv("QK_DEVICE"); unsetenv("QK_DEVICE_PCI");
    setenv("QK_DEVICE_NAME","STRIX_HALO",1);
    setenv("QK_OPERATOR_DEVICE_LOCAL","1",1);
    VkCtx c; initVk(c,argv[0]);
    if(c.props.vendorID!=0x1002||c.props.deviceID!=0x1586||pciBdf(c.phys)!="0000:c1:00.0") return 2;
    bool pass=true;
    {
        MappedBuffer q(c,maxRows*rowFloats), k(c,size_t(kvHeads)*capacity*dim),
                     v(c,size_t(kvHeads)*capacity*dim), gates(c,maxRows*rowFloats*2), output(c,outFloats,true);
        printf("batch operator memory: %s; staged=%s; repetitions=%u; synthetic only\n",
               memTypeDesc(c,k.b.memType).c_str(),k.staged?"yes":"no",repeats); fflush(stdout);
        Kernel baseline(c,"fa_attn_batch.spv",{&q,&k,&v,&gates,&output},40);
        Kernel vec16(c,"fa_attn_batch_vec4_q16.spv",{&q,&k,&v,&gates,&output},40);
        Kernel vec8(c,"fa_attn_batch_vec4.spv",{&q,&k,&v,&gates,&output},40);
        std::mt19937 rng(314159); std::uniform_real_distribution<float> uniform(-1,1);
        std::vector<float> goodQ(q.b.size/4),goodK(k.b.size/4),goodV(v.b.size/4),goodG(gates.b.size/4);
        for(auto* data:{&goodQ,&goodK,&goodV}) for(float& x:*data) x=uniform(rng);
        for(float& x:goodG) x=20*uniform(rng);
        float poison=std::numeric_limits<float>::quiet_NaN();
        // Tail and qbase edges, partial tiles, long contexts, then a shorter
        // case to catch stale KV and output reuse. Every unused cache row is NaN.
        for(const Case test:std::vector<Case>{{0,1,0,16},{0,15,0,16},{0,16,0,16},
                {0,17,16,16},{0,31,0,32},{0,64,16,32},{255,127,0,128},
                {0,512,0,512},{15872,512,0,128},{15872,512,384,128},
                {32256,512,0,64},{32256,512,448,64},
                {16400,33,16,32},{257,33,16,32,16},{1024,17,0,32,128},{17,7,0,16}}) {
            std::fill(q.ptr,q.ptr+goodQ.size(),poison);
            std::fill(gates.ptr,gates.ptr+goodG.size(),poison);
            memcpy(q.ptr,goodQ.data(),size_t(test.rows)*rowFloats*4);
            for(size_t i=0;i<size_t(test.rows)*rowFloats;++i) q.ptr[i]*=test.qScale;
            memcpy(gates.ptr,goodG.data(),size_t(test.rows)*rowFloats*8);
            std::fill(k.ptr,k.ptr+goodK.size(),poison);
            std::fill(v.ptr,v.ptr+goodV.size(),poison);
            const uint32_t keys=test.base+std::min(test.rows,test.first+test.tile);
            for(uint32_t kv=0;kv<kvHeads;++kv) {
                const size_t off=size_t(kv)*capacity*dim;
                memcpy(k.ptr+off,goodK.data()+off,size_t(keys)*dim*4);
                memcpy(v.ptr+off,goodV.data()+off,size_t(keys)*dim*4);
            }
            const auto gold=reference(test,q.ptr,k.ptr,v.ptr,gates.ptr);
            GemmPC pc{capacity,dim,64,heads,kvHeads,1e-6f,1e7f,test.base,test.rows,test.first};
            std::vector<float> original;
            uint32_t order=0;
            for(uint32_t mode:{0u,1u,2u,2u,1u,0u}) {
                Kernel& kernel=mode==0?baseline:(mode==1?vec16:vec8);
                const uint32_t qb=mode==2?8:16;
                auto emit=[&]{kernel.dispatch(pc,heads,1,(test.tile+qb-1)/qb);};
                std::fill(output.ptr,output.ptr+outFloats,poison);
                timed(c,1,emit);
                const double us=timed(c,repeats,emit);
                pass&=check(test,output.ptr,original,gold,poison,
                            mode==0?"baseline":(mode==1?"vec4-q16":"vec4-q8"),order++,us);
                if(original.empty()) original.assign(output.ptr,output.ptr+outFloats);
            }
        }
    }
    VK_CHECK(vkDeviceWaitIdle(c.dev));
    vkDestroyCommandPool(c.dev,c.pool,nullptr);
    vkDestroyDevice(c.dev,nullptr); vkDestroyInstance(c.inst,nullptr);
    return pass?0:1;
}
