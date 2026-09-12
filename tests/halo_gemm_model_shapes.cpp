// Model-shape scalar tuning. CPU F32 blocked-FMA + full GPU baseline exact gate.
// Original FP64 tolerance misses remain recorded separately, never relabeled.
#define QK_LIBRARY
#include "../src/main.cpp"
#include <limits>
#include <type_traits>
namespace {
struct GemmPC { uint32_t M,K,N,xStride,xOff,yStride,yOff; };
static_assert(sizeof(GemmPC)==28);
#include "halo_gemm_operator_common.h"

template<class Block> void weights(std::vector<Block>& w, std::mt19937& rng) {
    for (auto& b:w) {
        for (size_t i=0;i<sizeof b;++i) reinterpret_cast<uint8_t*>(&b)[i]=uint8_t(rng());
        b.d=qk_f32_to_f16(0.001f+0.002f*(rng()&0xffff)/65536.f);
        if constexpr(std::is_same_v<Block,block_q5_K>) b.dmin=qk_f32_to_f16(0.003f);
        if constexpr(std::is_same_v<Block,block_q5_1>) b.m=qk_f32_to_f16(-0.04f);
    }
}

template<class Block> bool test(VkCtx& c,uint32_t fmt,uint32_t M,uint32_t K,uint32_t N,uint32_t reps) {
    constexpr uint32_t width = std::is_same_v<Block,block_q5_K> || std::is_same_v<Block,block_q6_K> ? 256 : 32;
    std::mt19937 rng(4421+fmt+M+K+N);
    std::vector<Block> w(size_t(M)*K/width); weights(w,rng);
    GemmPC pc{M,K,N,K+7,13,M+11,17};
    const size_t nx=pc.xOff+size_t(N)*pc.xStride+16, ny=pc.yOff+size_t(N)*pc.yStride+16;
    MappedBuffer bw(c,(w.size()*sizeof(Block)+3)/4),bx(c,nx),by(c,ny,true);
    memcpy(bw.ptr,w.data(),w.size()*sizeof(Block));
    std::fill(bx.ptr,bx.ptr+nx,std::numeric_limits<float>::quiet_NaN());
    for(uint32_t n=0;n<N;++n) for(uint32_t k=0;k<K;++k)
        bx.ptr[pc.xOff+size_t(n)*pc.xStride+k]=(float(rng()&0xffff)/65536.f-0.5f)*4.f;
    // FP64 checks all small-shape outputs and 257 samples of larger shapes.
    // Candidates ALSO must match every baseline output byte and guard byte.
    std::vector<size_t> indices;
    if(size_t(M)*N<=10000) {
        for(size_t i=0;i<size_t(M)*N;++i) indices.push_back(i);
    } else {
        indices={0,size_t(M)*N-1};
        for(uint32_t i=0;i<255;++i) indices.push_back(rng()%(size_t(M)*N));
    }
    std::vector<double> gold; std::vector<float> blockedFma, dq(K);
    for(size_t i:indices) {
        uint32_t row=i%M,n=i/M; const Block* wp=&w[size_t(row)*K/width];
        if constexpr(std::is_same_v<Block,block_q5_K>) dequant_row_q5_K(wp,dq.data(),K);
        if constexpr(std::is_same_v<Block,block_q6_K>) dequant_row_q6_K(wp,dq.data(),K);
        if constexpr(std::is_same_v<Block,block_q8_0>) dequant_row_q8_0(wp,dq.data(),K);
        if constexpr(std::is_same_v<Block,block_q5_1>) dequant_row_q5_1(wp,dq.data(),K);
        double sum=0; for(uint32_t k=0;k<K;++k) sum+=double(dq[k])*bx.ptr[pc.xOff+size_t(n)*pc.xStride+k];
        gold.push_back(sum);
        float acc=0;
        for(uint32_t k0=0;k0<K;k0+=32) {
            float part=0;
            for(uint32_t k=k0;k<k0+32;++k)
                part=std::fma(dq[k],bx.ptr[pc.xOff+size_t(n)*pc.xStride+k],part);
            acc+=part;
        }
        blockedFma.push_back(acc);
    }
    const char* baseline[]={"qwen4_gemm_q5k.spv","qwen4_gemm_q6k.spv","qwen4_gemm_q8_0.spv","qwen4_gemm_q5_1.spv"};
    const char* compact[]={"qwen4_gemm_compact_q5k.spv","qwen4_gemm_compact_q6k.spv","qwen4_gemm_compact_q8_0.spv","qwen4_gemm_compact_q5_1.spv"};
    const std::string candidate=compact[fmt];
    // ABBA order, two timed cells per implementation, identical per-cell warmup.
    std::vector<std::string> names={baseline[fmt],candidate,candidate,baseline[fmt]};
    std::vector<float> exact; bool all=true;
    printf("memory fmt=%u M=%u K=%u N=%u weight=%s staged=%s\n",fmt,M,K,N,
           memTypeDesc(c,bw.b.memType).c_str(),bw.staged?"yes":"no"); fflush(stdout);
    for(size_t j=0;j<names.size();++j) {
        Kernel kernel(c,names[j].c_str(),{&bw,&bx,&by},sizeof pc);
        auto emit=[&]{kernel.dispatch(pc,(M+127)/128,1,(N+63)/64);};
        std::fill(by.ptr,by.ptr+ny,std::numeric_limits<float>::quiet_NaN());
        timed(c,1,emit);
        std::fill(by.ptr,by.ptr+ny,std::numeric_limits<float>::quiet_NaN());
        double us=timed(c,reps,emit);
        bool pass=true; double worst=0; size_t fp64LegacyMisses=0, fmaMismatches=0;
        for(uint32_t n=0;n<N;++n) for(uint32_t m=0;m<M;++m)
            pass &= std::isfinite(by.ptr[pc.yOff+size_t(n)*pc.yStride+m]);
        for(size_t p=0;p<ny;++p) {
            bool live=p>=pc.yOff && (p-pc.yOff)/pc.yStride<N && (p-pc.yOff)%pc.yStride<M;
            if(!live) pass &= std::isnan(by.ptr[p]);
        }
        for(size_t t=0;t<indices.size();++t) {
            size_t i=indices[t], row=i%M,n=i/M;
            double error=std::abs(double(by.ptr[pc.yOff+n*pc.yStride+row])-gold[t]);
            worst=std::max(worst,error);
            fp64LegacyMisses += error>2e-5*(1+std::abs(gold[t]));
            fmaMismatches += by.ptr[pc.yOff+n*pc.yStride+row] != blockedFma[t];
        }
        bool same=true;
        if(j==0) exact.assign(by.ptr,by.ptr+ny);
        else same=memcmp(exact.data(),by.ptr,ny*4)==0;
        pass &= same && fmaMismatches==0; all &= pass;
        printf("{\"shader\":\"%s\",\"order\":%zu,\"fmt\":%u,\"M\":%u,\"K\":%u,\"N\":%u,\"gpu_us\":%.6f,\"fp64_samples\":%zu,\"max_abs_error\":%.9g,\"fp64_legacy_misses\":%zu,\"cpu_blocked_fma_mismatches\":%zu,\"baseline_exact\":%s,\"result\":\"%s\"}\n",
               names[j].c_str(),j,fmt,M,K,N,us,indices.size(),worst,fp64LegacyMisses,fmaMismatches,j==0?"null":(same?"true":"false"),pass?"PASS":"FAIL");
        fflush(stdout);
    }
    return all;
}
}
int main(int argc,char**argv) {
    uint32_t reps=8;
    if(argc>2) return 2;
    if(argc==2) { char* e=nullptr; long v=strtol(argv[1],&e,10); if(e==argv[1]||*e||v<1||v>100)return 2; reps=uint32_t(v); }
    unsetenv("QK_DEVICE"); unsetenv("QK_DEVICE_PCI"); setenv("QK_DEVICE_NAME","STRIX_HALO",1);
    setenv("QK_OPERATOR_DEVICE_LOCAL","1",1);
    VkCtx c; initVk(c,argv[0]);
    if(c.props.vendorID!=0x1002||c.props.deviceID!=0x1586||pciBdf(c.phys)!="0000:c1:00.0")return 2;
    bool pass=true;
    // fmt,M,K from GGUF metadata for blk.0 and blk.3 dense/HC/shared projections.
    const std::vector<std::vector<uint32_t>> shapes={
        {0,512,2560},{0,2560,6144},{0,12288,2560},{0,640,2560},
        {0,320,10240},{0,4,10240},{0,6144,2560},{0,48,2560},
        {1,512,2560},{1,320,10240},{1,10240,2560},
        {2,2560,640},{3,10240,320}};
    for(auto shape:shapes) for(uint32_t N:{64u,128u,256u,512u}) {
        uint32_t fmt=shape[0],M=shape[1],K=shape[2];
        switch(fmt) {
            case 0: pass &= test<block_q5_K>(c,fmt,M,K,N,reps); break;
            case 1: pass &= test<block_q6_K>(c,fmt,M,K,N,reps); break;
            case 2: pass &= test<block_q8_0>(c,fmt,M,K,N,reps); break;
            case 3: pass &= test<block_q5_1>(c,fmt,M,K,N,reps); break;
        }
    }
    VK_CHECK(vkDeviceWaitIdle(c.dev)); vkDestroyCommandPool(c.dev,c.pool,nullptr);
    vkDestroyDevice(c.dev,nullptr); vkDestroyInstance(c.inst,nullptr);
    return pass?0:1;
}
