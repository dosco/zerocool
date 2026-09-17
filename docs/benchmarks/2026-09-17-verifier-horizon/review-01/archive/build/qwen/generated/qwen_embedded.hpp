#pragma once
#include <string_view>
namespace freellm::qwen {
inline constexpr const char* BuildFingerprint = "51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594";
inline constexpr std::string_view MetalSource = R"FREELLM_MSL(// Qwen3.8-Flash-Next, M1-compatible Metal. Affine Q4 is NOT GGUF Q4_0/Q4_K.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void copy_words(device const uint* source [[buffer(0)]],device uint* destination [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]) destination[i]=source[i];
}
inline float bf(float x) {
    uint u=as_type<uint>(x);
    if((u&0x7fffffffu)>0x7f800000u) return NAN;
    return as_type<float>((u+0x7fffu+((u>>16)&1u))&0xffff0000u);
}
inline float b16(ushort x) { return as_type<float>(uint(x)<<16); }
inline float scalar(device const uchar* p,uint i,uint dtype) {
    if(dtype==1) return ((device const float*)p)[i];
    if(dtype==2) return float(((device const half*)p)[i]);
    return b16(((device const ushort*)p)[i]);
}
inline float sigmoid_f(float x) {
    float y=1.0f/(1.0f+precise::exp(abs(x))); return x<0?y:1.0f-y;
}
// MLX's BF16 functors round the exponential, denominator and reciprocal.
// These boundaries matter before later matmuls, even when outputs are BF16.
inline float sigmoid_bf(float x) {
    float y=bf(1.0f/bf(1.0f+bf(precise::exp(abs(x))))); return x<0?y:bf(1.0f-y);
}
inline float softplus_bf(float x) {
    float e=bf(precise::exp(-abs(x))),one=1.0f+e;
    float tail=one==1.0f?e:e*(log(one)/(one-1.0f));
    return bf(max(x,0.0f)+bf(tail));
}

// Affine Q4 GEMV arithmetic follows MLX 0.31.1's load_vector/qdot:
// the bias multiplies sums of four BF16 inputs, rounded after each addition.
// Keep this arithmetic independent of token chunk size and cache residency.
// Adapted from MLX quantized.h; Apple MIT notice in docs/licenses/mlx.txt.
inline float affine_dot(device const uint* w,device const ushort* s,device const ushort* b,
    device const float* x,uint row,uint K,uint N,uint group,uint lane) {
    const uint width=(K%512==0 && N%8==0)?16:8;
    float result=0;
    for(uint base=lane*width;base<K;base+=32*width) {
        float dot=0,sum=0;
        for(uint i=0;i<width && base+i<K;i+=4) {
            uint k=base+i,packed=(w[row*(K/8)+k/8]>>(4*(k%8)))&0xffff;
            float a=x[k],c=x[k+1],d=x[k+2],e=x[k+3];
            sum+=bf(bf(bf(a+c)+d)+e);
            dot+=a*float(packed&15)+c*float((packed>>4)&15)+d*float((packed>>8)&15)+e*float((packed>>12)&15);
        }
        uint g=row*(K/group)+base/group;
        result+=b16(s[g])*dot+sum*b16(b[g]);
    }
    return simd_sum(result);
}
kernel void q4_mm(device const uint* w [[buffer(0)]],device const ushort* s [[buffer(1)]],
    device const ushort* b [[buffer(2)]],device const float* x [[buffer(3)]],
    device float* out [[buffer(4)]],constant uint* p [[buffer(5)]],uint2 gid [[thread_position_in_grid]]) {
    uint row=gid.x/32,lane=gid.x%32,t=gid.y;
    if(row>=p[1] || t>=p[2]) return;
    float sum=affine_dot(w,s,b,x+t*p[0],row,p[0],p[1],p[3],lane);
    if(!lane) out[t*p[1]+row]=p[4]?sum:bf(sum);
}
kernel void q4_gate_up(device const uint* gw [[buffer(0)]],device const ushort* gs [[buffer(1)]],
    device const ushort* gb [[buffer(2)]],device const uint* uw [[buffer(3)]],device const ushort* us [[buffer(4)]],
    device const ushort* ub [[buffer(5)]],device const float* x [[buffer(6)]],device const int* rows [[buffer(7)]],
    device float* out [[buffer(8)]],constant uint* p [[buffer(9)]],uint2 gid [[thread_position_in_grid]]) {
    uint row=gid.x/32,lane=gid.x%32,t=gid.y,K=p[0],N=p[1];
    if(row>=N || t>=p[2]) return;
    uint src=p[4]?uint(rows[t]):t;
    float gate=bf(affine_dot(gw,gs,gb,x+src*K,row,K,N,p[3],lane));
    float up=bf(affine_dot(uw,us,ub,x+src*K,row,K,N,p[3],lane));
    if(!lane) out[t*N+row]=bf(bf(gate*sigmoid_bf(gate))*up);
}
// MLX affine Q8 sums inputs in FP32 one at a time. The Q4 sum-of-four
// BF16 rounding is NOT part of Q8 arithmetic. Keep the QMV partition fixed
// across token counts, matching MLX 0.31.1 qmv/qmv_fast for one token.
// Developer probe: 24 little-endian code bytes + BF16 scale/bias per 64 weights.
inline float q3_probe_dot(device const uchar* record,device const float* x,uint row,uint K,uint N,uint lane) {
    const uint width=(K%512==0 && N%8==0)?16:8;
    float result=0;
    for(uint base=lane*width;base<K;base+=32*width) {
        device const uchar* g=record+(row*(K/64)+base/64)*28;
        float dot=0,sum=0;
        for(uint i=0;i<width && base+i<K;i+=4) {
            uint k=base+i,bit=(k%64)*3,at=bit/8;
            uint packed=(uint(g[at])|(uint(g[at+1])<<8))>>(bit%8);
            float a=x[k],c=x[k+1],d=x[k+2],e=x[k+3];
            sum+=bf(bf(bf(a+c)+d)+e);
            dot+=a*float(packed&7)+c*float((packed>>3)&7)+d*float((packed>>6)&7)+e*float((packed>>9)&7);
        }
        result+=b16(ushort(uint(g[24])|(uint(g[25])<<8)))*dot+sum*b16(ushort(uint(g[26])|(uint(g[27])<<8)));
    }
    return simd_sum(result);
}
kernel void q3_probe_gate_up(device const uchar* gate [[buffer(0)]],device const uchar* up [[buffer(1)]],
    device const float* x [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint row=gid.x/32,lane=gid.x%32,t=gid.y;
    if(row>=p[1] || t>=p[2]) return;
    float g=bf(q3_probe_dot(gate,x+t*p[0],row,p[0],p[1],lane));
    float u=bf(q3_probe_dot(up,x+t*p[0],row,p[0],p[1],lane));
    if(!lane) out[t*p[1]+row]=bf(bf(g*sigmoid_bf(g))*u);
}
inline float affine8_dot(device const uchar* w,device const ushort* s,device const ushort* b,
    device const float* x,uint row,uint K,uint N,uint group,uint lane) {
    const uint width=(K%512==0 && N%8==0)?8:4;
    float result=0;
    for(uint base=lane*width;base<K;base+=32*width) {
        float dot=0,sum=0;
        for(uint i=0;i<width && base+i<K;++i) {
            float value=x[base+i];sum+=value;dot+=value*float(w[row*K+base+i]);
        }
        uint g=row*(K/group)+base/group;
        result+=b16(s[g])*dot+sum*b16(b[g]);
    }
    return simd_sum(result);
}
kernel void q8_mm(device const uchar* w [[buffer(0)]],device const ushort* s [[buffer(1)]],
    device const ushort* b [[buffer(2)]],device const float* x [[buffer(3)]],
    device float* out [[buffer(4)]],constant uint* p [[buffer(5)]],uint2 gid [[thread_position_in_grid]]) {
    uint row=gid.x/32,lane=gid.x%32,t=gid.y;
    if(row>=p[1] || t>=p[2]) return;
    float sum=affine8_dot(w,s,b,x+t*p[0],row,p[0],p[1],p[3],lane);
    if(!lane) out[t*p[1]+row]=p[4]?sum:bf(sum);
}
kernel void q8_gate_up(device const uchar* gw [[buffer(0)]],device const ushort* gs [[buffer(1)]],
    device const ushort* gb [[buffer(2)]],device const uchar* uw [[buffer(3)]],device const ushort* us [[buffer(4)]],
    device const ushort* ub [[buffer(5)]],device const float* x [[buffer(6)]],device const int* rows [[buffer(7)]],
    device float* out [[buffer(8)]],constant uint* p [[buffer(9)]],uint2 gid [[thread_position_in_grid]]) {
    uint row=gid.x/32,lane=gid.x%32,t=gid.y,K=p[0],N=p[1];
    if(row>=N || t>=p[2]) return;
    uint src=p[4]?uint(rows[t]):t;
    float gate=bf(affine8_dot(gw,gs,gb,x+src*K,row,K,N,p[3],lane));
    float up=bf(affine8_dot(uw,us,ub,x+src*K,row,K,N,p[3],lane));
    if(!lane) out[t*N+row]=bf(bf(gate*sigmoid_bf(gate))*up);
}
// Independent output accumulators share input loads and the affine bias sum.
// Each output retains the reference lane partition and scalar reduction order.
template<uint Bits,uint Rows,uint Tile>
inline void affine_block(device const uchar* w,device const ushort* s,device const ushort* b,
    device const float* x,uint row0,uint t0,uint K,uint N,uint T,uint group,uint lane,
    thread float* result) {
    const uint width=(K%512==0 && N%8==0)?(Bits==4?16:8):(Bits==4?8:4);
    for(uint r=0;r<Rows;++r) for(uint t=0;t<Tile;++t) result[r*Tile+t]=0;
    for(uint base=lane*width;base<K;base+=32*width) {
        float dot[Rows*Tile],sum[Tile];
        for(uint t=0;t<Tile;++t) sum[t]=0;
        for(uint r=0;r<Rows;++r) for(uint t=0;t<Tile;++t) dot[r*Tile+t]=0;
        for(uint i=0;i<width && base+i<K;i+=(Bits==4?4:1)) {
            const uint k=base+i;uint codes[Rows];
            for(uint r=0;r<Rows;++r) if(row0+r<N)
                codes[r]=Bits==4?((((device const uint*)w)[(row0+r)*(K/8)+k/8]>>(4*(k%8)))&0xffff):w[(row0+r)*K+k];
            for(uint t=0;t<Tile;++t) if(t0+t<T) {
                device const float* v=x+(t0+t)*K+k;
                if(Bits==4) {
                    float a=v[0],c=v[1],d=v[2],e=v[3];sum[t]+=bf(bf(bf(a+c)+d)+e);
                    for(uint r=0;r<Rows;++r) if(row0+r<N) {
                        uint code=codes[r];dot[r*Tile+t]+=a*float(code&15)+c*float((code>>4)&15)+d*float((code>>8)&15)+e*float((code>>12)&15);
                    }
                } else {
                    const float v0=v[0];sum[t]+=v0;
                    for(uint r=0;r<Rows;++r) if(row0+r<N) dot[r*Tile+t]+=v0*float(codes[r]);
                }
            }
        }
        for(uint r=0;r<Rows;++r) if(row0+r<N) {
            const uint g=(row0+r)*(K/group)+base/group;
            const float scale=b16(s[g]),bias=b16(b[g]);
            for(uint t=0;t<Tile;++t) result[r*Tile+t]+=scale*dot[r*Tile+t]+sum[t]*bias;
        }
    }
    for(uint r=0;r<Rows;++r) for(uint t=0;t<Tile;++t) result[r*Tile+t]=simd_sum(result[r*Tile+t]);
}
#define AFFINE_BLOCK(Bits,Rows,Tile) \
kernel void q##Bits##_mm_r##Rows##_t##Tile(device const uchar* w [[buffer(0)]],device const ushort* s [[buffer(1)]], \
    device const ushort* b [[buffer(2)]],device const float* x [[buffer(3)]],device float* out [[buffer(4)]], \
    constant uint* p [[buffer(5)]],uint2 gid [[thread_position_in_grid]]) { \
    const uint row0=(gid.x/32)*Rows,lane=gid.x%32,t0=gid.y*Tile; \
    if(row0>=p[1] || t0>=p[2]) return;float sums[Rows*Tile]; \
    affine_block<Bits,Rows,Tile>(w,s,b,x,row0,t0,p[0],p[1],p[2],p[3],lane,sums); \
    if(!lane) for(uint r=0;r<Rows && row0+r<p[1];++r) for(uint t=0;t<Tile && t0+t<p[2];++t) \
        out[(t0+t)*p[1]+row0+r]=p[4]?sums[r*Tile+t]:bf(sums[r*Tile+t]); \
}
AFFINE_BLOCK(8,2,4)
AFFINE_BLOCK(8,2,8)
AFFINE_BLOCK(8,4,4)
AFFINE_BLOCK(8,4,8)
AFFINE_BLOCK(8,2,1)
AFFINE_BLOCK(4,2,1)

// Decode-only packed Q8: Width is the reference lane partition, not a new
// reduction. Word loads and independent output rows share input/bias work.
// Every scalar multiply/add and final SIMD reduction keeps its original order.
template<uint Rows,uint Width>
inline void q8_mv_packed(device const uint* w,device const ushort* s,device const ushort* b,
    device const float* x,device float* out,constant uint* p,uint tid) {
    const uint row0=(tid/32)*Rows,lane=tid%32,K=p[0],N=p[1],G=p[2];
    if(row0>=N) return;
    float result[Rows];
    #pragma unroll
    for(uint r=0;r<Rows;++r) result[r]=0;
    for(uint base=lane*Width;base<K;base+=32*Width) {
        uint words[Rows][Width/4];float dot[Rows];
        #pragma unroll
        for(uint r=0;r<Rows;++r) {
            dot[r]=0;
            #pragma unroll
            for(uint i=0;i<Width/4;++i) if(row0+r<N) words[r][i]=w[(row0+r)*(K/4)+base/4+i];
        }
        float sum=0;
        #pragma unroll
        for(uint i=0;i<Width;++i) {
            const float v=x[base+i];sum+=v;
            #pragma unroll
            for(uint r=0;r<Rows;++r) if(row0+r<N) dot[r]+=v*float((words[r][i/4]>>(8*(i%4)))&255);
        }
        #pragma unroll
        for(uint r=0;r<Rows;++r) if(row0+r<N) {
            const uint g=(row0+r)*(K/G)+base/G;
            result[r]+=b16(s[g])*dot[r]+sum*b16(b[g]);
        }
    }
    #pragma unroll
    for(uint r=0;r<Rows;++r) {
        const float value=simd_sum(result[r]);
        if(!lane && row0+r<N) out[row0+r]=p[3]?value:bf(value);
    }
}
#define Q8_PACKED(Rows,Width) \
kernel void q8_mv_packed_r##Rows##_w##Width(device const uint* w [[buffer(0)]], \
    device const ushort* s [[buffer(1)]],device const ushort* b [[buffer(2)]], \
    device const float* x [[buffer(3)]],device float* out [[buffer(4)]],constant uint* p [[buffer(5)]], \
    uint tid [[thread_position_in_grid]]) {q8_mv_packed<Rows,Width>(w,s,b,x,out,p,tid);}
Q8_PACKED(2,4)
Q8_PACKED(2,8)
Q8_PACKED(4,4)
Q8_PACKED(4,8)
Q8_PACKED(8,4)
Q8_PACKED(8,8)

// Gate/up has two independent dot products but exactly the same input bias sum.
template<uint Bits>
inline float affine_gate_pair(device const uchar* gw,device const ushort* gs,device const ushort* gb,
    device const uchar* uw,device const ushort* us,device const ushort* ub,device const float* x,
    uint row,uint K,uint N,uint group,uint lane) {
    const uint width=(K%512==0 && N%8==0)?(Bits==4?16:8):(Bits==4?8:4);
    float gate=0,up=0;
    for(uint base=lane*width;base<K;base+=32*width) {
        float gd=0,ud=0,sum=0;
        for(uint i=0;i<width && base+i<K;i+=(Bits==4?4:1)) {
            uint k=base+i;
            if(Bits==4) {
                uint gc=(((device const uint*)gw)[row*(K/8)+k/8]>>(4*(k%8)))&0xffff;
                uint uc=(((device const uint*)uw)[row*(K/8)+k/8]>>(4*(k%8)))&0xffff;
                float a=x[k],c=x[k+1],d=x[k+2],e=x[k+3];sum+=bf(bf(bf(a+c)+d)+e);
                gd+=a*float(gc&15)+c*float((gc>>4)&15)+d*float((gc>>8)&15)+e*float((gc>>12)&15);
                ud+=a*float(uc&15)+c*float((uc>>4)&15)+d*float((uc>>8)&15)+e*float((uc>>12)&15);
            } else {float v=x[k];sum+=v;gd+=v*float(gw[row*K+k]);ud+=v*float(uw[row*K+k]);}
        }
        uint g=row*(K/group)+base/group;
        gate+=b16(gs[g])*gd+sum*b16(gb[g]);up+=b16(us[g])*ud+sum*b16(ub[g]);
    }
    gate=bf(simd_sum(gate));up=bf(simd_sum(up));return bf(bf(gate*sigmoid_bf(gate))*up);
}
#define AFFINE_PAIR(Bits) \
kernel void q##Bits##_gate_up_pair(device const uchar* gw [[buffer(0)]],device const ushort* gs [[buffer(1)]], \
    device const ushort* gb [[buffer(2)]],device const uchar* uw [[buffer(3)]],device const ushort* us [[buffer(4)]], \
    device const ushort* ub [[buffer(5)]],device const float* x [[buffer(6)]],device const int* rows [[buffer(7)]], \
    device float* out [[buffer(8)]],constant uint* p [[buffer(9)]],uint2 gid [[thread_position_in_grid]]) { \
    uint row=gid.x/32,lane=gid.x%32,t=gid.y;if(row>=p[1] || t>=p[2]) return; \
    float value=affine_gate_pair<Bits>(gw,gs,gb,uw,us,ub,x+(p[4]?uint(rows[t]):t)*p[0],row,p[0],p[1],p[3],lane); \
    if(!lane) out[t*p[1]+row]=value; \
}
AFFINE_PAIR(4)
AFFINE_PAIR(8)

kernel void affine_embedding(device const uint* w [[buffer(0)]],device const ushort* s [[buffer(1)]],
    device const ushort* b [[buffer(2)]],device const int* ids [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint2 g [[thread_position_in_grid]]) {
    if(g.x>=p[1] || g.y>=p[2]) return;
    uint token=uint(ids[g.y]),d=g.x%p[0],group=token*(p[0]/p[3])+d/p[3],pack=32/p[4];
    uint code=(w[token*(p[0]/pack)+d/pack]>>(p[4]*(d%pack)))&((1u<<p[4])-1);
    out[g.y*p[1]+g.x]=bf(float(code)*b16(s[group])+b16(b[group]));
}

// Reuse each packed weight across independent token accumulators. In contrast
// to GEMM reduction, each accumulator follows precisely affine_dot/affine8_dot:
// same lane partition, scalar addition order, and BF16 rounding boundaries.
template<uint Bits,uint Tile>
inline void affine_multi(device const uchar* w,device const ushort* s,device const ushort* b,
    device const float* x,device const int* rows,bool gathered,uint t0,uint T,
    uint row,uint K,uint N,uint group,uint lane,thread float* result) {
    const uint width=(K%512==0 && N%8==0)?(Bits==4?16:8):(Bits==4?8:4);
    for(uint t=0;t<Tile;++t) result[t]=0;
    for(uint base=lane*width;base<K;base+=32*width) {
        float dot[Tile],sum[Tile];
        for(uint t=0;t<Tile;++t) {dot[t]=0;sum[t]=0;}
        for(uint i=0;i<width && base+i<K;i+=(Bits==4?4:1)) {
            uint k=base+i;
            uint code=Bits==4?((((device const uint*)w)[row*(K/8)+k/8]>>(4*(k%8)))&0xffff):w[row*K+k];
            for(uint t=0;t<Tile;++t) if(t0+t<T) {
                uint src=gathered?uint(rows[t0+t]):t0+t;
                device const float* v=x+src*K+k;
                if(Bits==4) {
                    float a=v[0],c=v[1],d=v[2],e=v[3];
                    sum[t]+=bf(bf(bf(a+c)+d)+e);
                    dot[t]+=a*float(code&15)+c*float((code>>4)&15)+d*float((code>>8)&15)+e*float((code>>12)&15);
                } else {sum[t]+=v[0];dot[t]+=v[0]*float(code);}
            }
        }
        uint g=row*(K/group)+base/group;
        float scale=b16(s[g]),bias=b16(b[g]);
        for(uint t=0;t<Tile;++t) result[t]+=scale*dot[t]+sum[t]*bias;
    }
    for(uint t=0;t<Tile;++t) result[t]=simd_sum(result[t]);
}
#define AFFINE_MULTI(Bits,Tile) \
kernel void q##Bits##_mm_t##Tile(device const uchar* w [[buffer(0)]],device const ushort* s [[buffer(1)]], \
    device const ushort* b [[buffer(2)]],device const float* x [[buffer(3)]],device float* out [[buffer(4)]], \
    constant uint* p [[buffer(5)]],uint2 gid [[thread_position_in_grid]]) { \
    uint row=gid.x/32,lane=gid.x%32,t0=gid.y*Tile; \
    if(row>=p[1] || t0>=p[2]) return;float sums[Tile]; \
    affine_multi<Bits,Tile>(w,s,b,x,(device const int*)x,false,t0,p[2],row,p[0],p[1],p[3],lane,sums); \
    if(!lane) for(uint t=0;t<Tile && t0+t<p[2];++t) out[(t0+t)*p[1]+row]=p[4]?sums[t]:bf(sums[t]); \
} \
kernel void q##Bits##_gate_up_t##Tile(device const uchar* gw [[buffer(0)]],device const ushort* gs [[buffer(1)]], \
    device const ushort* gb [[buffer(2)]],device const uchar* uw [[buffer(3)]],device const ushort* us [[buffer(4)]], \
    device const ushort* ub [[buffer(5)]],device const float* x [[buffer(6)]],device const int* rows [[buffer(7)]], \
    device float* out [[buffer(8)]],constant uint* p [[buffer(9)]],uint2 gid [[thread_position_in_grid]]) { \
    uint row=gid.x/32,lane=gid.x%32,t0=gid.y*Tile; \
    if(row>=p[1] || t0>=p[2]) return;float gates[Tile],ups[Tile]; \
    affine_multi<Bits,Tile>(gw,gs,gb,x,rows,bool(p[4]),t0,p[2],row,p[0],p[1],p[3],lane,gates); \
    affine_multi<Bits,Tile>(uw,us,ub,x,rows,bool(p[4]),t0,p[2],row,p[0],p[1],p[3],lane,ups); \
    if(!lane) for(uint t=0;t<Tile && t0+t<p[2];++t) {float g=bf(gates[t]),u=bf(ups[t]); \
        out[(t0+t)*p[1]+row]=bf(bf(g*sigmoid_bf(g))*u);} \
}
AFFINE_MULTI(4,2)
AFFINE_MULTI(4,4)
AFFINE_MULTI(4,8)
AFFINE_MULTI(8,2)
AFFINE_MULTI(8,4)
AFFINE_MULTI(8,8)
#undef AFFINE_MULTI
kernel void plain_mm(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 gid [[thread_position_in_grid]]) {
    uint row=gid.x/32,lane=gid.x%32,t=gid.y,K=p[0],N=p[1];
    if(row>=N || t>=p[2]) return;
    float sum=0;
    for(uint k=lane;k<K;k+=32) sum+=x[t*K+k]*scalar(w,row*K+k,p[3]);
    sum=simd_sum(sum); if(!lane) out[t*N+row]=p[4]?sum:bf(sum);
}
// The FP32 router is unusually sensitive to reduction order: one last-bit
// change can cross a later BF16 rounding boundary. Use the M1 SIMD matrix
// operation and sixteen contiguous K partitions, matching MLX's small-prefill
// GEMM for this checkpoint. The partitioning stays fixed for every chunk size.
// Fragment coordinates follow MLX steel/gemm/mma.h (Apple MIT notice).
kernel void router_mm(device const ushort* w [[buffer(0)]],device const float* x [[buffer(1)]],
    device float* partial [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint col=group.x*8,row=group.y*8,split=group.z,T=p[0];
    uint qid=lane/4,fr=(qid&4)+((lane/2)%4),fc=(qid&2)*2+(lane%2)*2;
    simdgroup_float8x8 a,b,c;
    c.thread_elements()[0]=0; c.thread_elements()[1]=0;
    for(uint k=split*160;k<(split+1)*160;k+=8) {
        for(uint j=0;j<2;++j) {
            a.thread_elements()[j]=row+fr<T?x[(row+fr)*2560+k+fc+j]:0;
            b.thread_elements()[j]=b16(w[(col+fc+j)*2560+k+fr]);
        }
        simdgroup_multiply_accumulate(c,a,b,c);
    }
    for(uint j=0;j<2;++j) if(row+fr<T)
        partial[(split*T+row+fr)*512+col+fc+j]=c.thread_elements()[j];
}
kernel void router_accumulate(device const float* partial [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint count=p[0]*512; if(i>=count) return;
    float sum=0; for(uint split=0;split<16;++split) sum+=partial[split*count+i];
    out[i]=sum;
}
kernel void embedding(device const uint* w [[buffer(0)]],device const ushort* s [[buffer(1)]],
    device const ushort* b [[buffer(2)]],device const int* ids [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint2 g [[thread_position_in_grid]]) {
    if(g.x>=p[0] || g.y>=p[1]) return;
    uint token=uint(ids[g.y]),d=g.x%2560,group=token*40+d/64;
    float v=float((w[token*320+d/8]>>(4*(d%8)))&15)*b16(s[group])+b16(b[group]);
    out[g.y*p[0]+g.x]=bf(v);
}
// Same four-value local and two-level SIMD reduction as MLX rms_single_row.
// All supported RMS widths fit at most 32 groups of 128 elements.
inline float rms_square_sum(device const float* x,uint D,uint lane) {
    float contribution=0;
    for(uint block=0;block<D;block+=128) {
        float acc=0;
        for(uint i=0;i<4;++i) {uint d=block+lane*4+i;if(d<D) acc+=x[d]*x[d];}
        acc=simd_sum(acc);
        if(lane==block/128) contribution=acc;
    }
    return simd_sum(contribution);
}
kernel void norm(device const float* x [[buffer(0)]],device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
    uint group=gid/32,lane=gid%32,D=p[0],width=p[1];
    if(group>=p[2]*(width/D)) return;
    float inv=precise::rsqrt(rms_square_sum(x+group*D,D,lane)/float(D)+1e-6f);
    for(uint d=lane;d<D;d+=32) {
        uint i=group*D+d,wi=i%width;
        float v=x[i]*inv;
        v=bf(v); // MLX rounds normalized values before the BF16 weight multiply
        out[i]=bf(v*scalar(w,wi,p[3]));
    }
}
kernel void unary(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]) return;
    float v=x[i]; uint op=p[1];
    if(op==0) v=v*sigmoid_bf(v);
    if(op==1) v=sigmoid_bf(v);
    if(op==2) { v=bf(v/4); v=v*sigmoid_bf(v); }
    if(op==3) v=2*bf(sigmoid_bf(bf(v/4)));
    if(op==4) v=softplus_bf(v);
    y[i]=bf(v);
}
kernel void binary(device const float* x [[buffer(0)]],device const float* y [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p[0]) out[i]=bf(p[1]?x[i]*y[i]:x[i]+y[i]);
}
kernel void hc_mix(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 i [[thread_position_in_grid]]) {
    if(i.x>=2560 || i.y>=p[0]) return;
    float sum=0;
    for(uint h=0;h<4;++h) { uint k=i.y*10240+h*2560+i.x; sum=bf(sum+bf(x[k]*w[k])); }
    out[i.y*2560+i.x]=bf(sum/4);
}
kernel void hc_add(device const float* hyper [[buffer(0)]],device const float* x [[buffer(1)]],
    device const float* inject [[buffer(2)]],device float* out [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 i [[thread_position_in_grid]]) {
    if(i.x>=10240 || i.y>=p[0]) return;
    uint h=i.x/2560,d=i.x%2560;
    out[i.y*10240+i.x]=bf(hyper[i.y*10240+i.x]+bf(x[i.y*2560+d]*inject[i.y*4+h]));
}
kernel void conv(device const float* x [[buffer(0)]],device const float* state [[buffer(1)]],
    device const uchar* w [[buffer(2)]],device float* out [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 gid [[thread_position_in_grid]]) {
    uint C=p[0],T=p[1],K=p[2],dilation=p[3],c=gid.x,t=gid.y;
    if(c>=C || t>=T) return;
    float v=0; int history=int((K-1)*dilation);
    for(uint j=0;j<K;++j) {
        int pos=int(t)+int(j*dilation)-history;
        float a=pos<0?state[(pos+history)*int(C)+int(c)]:x[uint(pos)*C+c];
        v+=a*scalar(w,c*K+j,p[4]);
    }
    v=bf(v); out[t*C+c]=bf(v*sigmoid_bf(v));
}
kernel void conv_update(device const float* x [[buffer(0)]],device const float* old [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint C=p[0],T=p[1],history=p[2]; if(i>=C*history) return;
    int pos=int(T)+int(i/C)-int(history);
    out[i]=pos<0?old[(pos+int(history))*int(C)+int(i%C)]:x[uint(pos)*C+i%C];
}
kernel void gdn_qk(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint2 gid [[thread_position_in_grid]]) {
    uint h=gid.x/32,lane=gid.x%32,t=gid.y;
    if(h>=32 || t>=p[0]) return;
    uint base=t*10240+h*128; float ss=0;
    // MLX's 128-element FP32 row reduction gives each lane four adjacent
    // values. Match that order and its precise reciprocal square root:
    // last-bit differences can cross the following BF16 rounding boundary.
    for(uint j=0;j<4;++j) {uint i=lane*4+j;ss+=x[base+i]*x[base+i];}
    float inv=precise::rsqrt(simd_sum(ss)+1e-6f);
    for(uint i=lane;i<128;i+=32) {
        float v=bf(x[base+i]*inv); if(h<16) v=bf(v*bf(0.08838834764831845f)); out[base+i]=v;
    }
    // Each of 32 groups copies a disjoint part of the 6144 value dimensions.
    for(uint i=h*32+lane;i<6144;i+=1024) out[t*10240+4096+i]=x[t*10240+4096+i];
}
// The FP32 state remains in registers across the chunk. The recurrence follows
// mlx-lm/Slotstream's gated delta update, with no BF16 state truncation.
kernel void gdn_scan(device const float* qkv [[buffer(0)]],device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],device const uchar* alog [[buffer(3)]],
    device const uchar* dt [[buffer(4)]],device float* state [[buffer(5)]],
    device float* out [[buffer(6)]],constant uint* p [[buffer(7)]],uint3 gid [[thread_position_in_grid]]) {
    uint lane=gid.x,dv=gid.y,h=gid.z,hk=h/3; if(lane>=32 || dv>=128 || h>=48) return;
    float st[4]; uint base=(h*128+dv)*128+lane*4;
    for(uint j=0;j<4;++j) st[j]=state[base+j];
    for(uint t=0;t<p[0];++t) {
        float at=bf(a[t*48+h]+scalar(dt,h,p[2]));
        float sp=softplus_bf(at);
        float decay=bf(precise::exp(-precise::exp(scalar(alog,h,p[1]))*sp));
        float beta=sigmoid_bf(b[t*48+h]);
        uint kbase=t*10240+2048+hk*128+lane*4;
        float memory=0;
        for(uint j=0;j<4;++j) {
            st[j]*=decay;
            memory+=st[j]*qkv[kbase+j];
        }
        memory=simd_sum(memory);
        float delta=(qkv[t*10240+4096+h*128+dv]-memory)*beta;
        float y=0;
        for(uint j=0;j<4;++j) {
            st[j]+=qkv[kbase+j]*delta;
            y+=st[j]*qkv[t*10240+hk*128+lane*4+j];
        }
        y=simd_sum(y); if(!lane) out[t*6144+h*128+dv]=bf(y);
    }
    for(uint j=0;j<4;++j) state[base+j]=st[j];
}
// One pair per token/head. Keep every original rounding boundary; the
// subsequent scan merely loads these same FP32 representations.
kernel void gdn_prepare(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device const uchar* alog [[buffer(2)]],device const uchar* dt [[buffer(3)]],
    device float2* gates [[buffer(4)]],constant uint* p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*48) return;uint h=i%48;
    float at=bf(a[i]+scalar(dt,h,p[2]));
    float sp=softplus_bf(at);
    gates[i]=float2(bf(precise::exp(-precise::exp(scalar(alog,h,p[1]))*sp)),sigmoid_bf(b[i]));
}
kernel void gdn_scan_prepared(device const float* qkv [[buffer(0)]],device const float2* gates [[buffer(1)]],
    device float* state [[buffer(2)]],device float* out [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint3 gid [[thread_position_in_grid]]) {
    uint lane=gid.x,dv=gid.y,h=gid.z,hk=h/3;if(lane>=32 || dv>=128 || h>=48) return;
    float st[4];uint base=(h*128+dv)*128+lane*4;
    for(uint j=0;j<4;++j) st[j]=state[base+j];
    for(uint t=0;t<p[0];++t) {
        float2 gate=gates[t*48+h];uint kbase=t*10240+2048+hk*128+lane*4;
        float memory=0;
        for(uint j=0;j<4;++j) {st[j]*=gate.x;memory+=st[j]*qkv[kbase+j];}
        memory=simd_sum(memory);
        float delta=(qkv[t*10240+4096+h*128+dv]-memory)*gate.y,y=0;
        for(uint j=0;j<4;++j) {st[j]+=qkv[kbase+j]*delta;y+=st[j]*qkv[t*10240+hk*128+lane*4+j];}
        y=simd_sum(y);if(!lane) out[t*6144+h*128+dv]=bf(y);
    }
    for(uint j=0;j<4;++j) state[base+j]=st[j];
}
// Staging changes only where inputs are loaded. Each SIMD group still owns
// one value row and performs the original four-adjacent-elements reduction.
template<uint Rows,uint Block>
inline void gdn_staged(device const float* qkv,device const float2* gates,device float* state,
    device float* out,uint T,uint3 gid,uint3 tid,threadgroup float* q,threadgroup float* k,
    threadgroup float* v,threadgroup float2* gb) {
    uint lane=gid.x,dv=gid.y,h=gid.z,hk=h/3,local=tid.y*32+tid.x;
    float st[4];uint base=(h*128+dv)*128+lane*4;
    for(uint j=0;j<4;++j) st[j]=state[base+j];
    for(uint t0=0;t0<T;t0+=Block) {
        uint count=min(Block,T-t0);
        for(uint n=local;n<count*128;n+=32*Rows) {
            uint t=t0+n/128,d=n%128;
            q[n]=qkv[t*10240+hk*128+d];k[n]=qkv[t*10240+2048+hk*128+d];
        }
        for(uint n=local;n<count*Rows;n+=32*Rows)
            v[n]=qkv[(t0+n/Rows)*10240+4096+h*128+(dv/Rows)*Rows+n%Rows];
        for(uint n=local;n<count;n+=32*Rows) gb[n]=gates[(t0+n)*48+h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint t=0;t<count;++t) {
            float memory=0;uint index=t*128+lane*4;
            for(uint j=0;j<4;++j) {st[j]*=gb[t].x;memory+=st[j]*k[index+j];}
            memory=simd_sum(memory);
            float delta=(v[t*Rows+tid.y]-memory)*gb[t].y,y=0;
            for(uint j=0;j<4;++j) {st[j]+=k[index+j]*delta;y+=st[j]*q[index+j];}
            y=simd_sum(y);if(!lane) out[(t0+t)*6144+h*128+dv]=bf(y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;++j) state[base+j]=st[j];
}
#define GDN_STAGED(Rows,Block) \
kernel void gdn_scan_r##Rows##_b##Block(device const float* qkv [[buffer(0)]],device const float2* gates [[buffer(1)]], \
    device float* state [[buffer(2)]],device float* out [[buffer(3)]],constant uint* p [[buffer(4)]], \
    uint3 gid [[thread_position_in_grid]],uint3 tid [[thread_position_in_threadgroup]]) { \
    threadgroup float q[Block*128],k[Block*128],v[Block*Rows];threadgroup float2 gb[Block]; \
    gdn_staged<Rows,Block>(qkv,gates,state,out,p[0],gid,tid,q,k,v,gb); \
}
GDN_STAGED(4,4)
GDN_STAGED(4,8)
GDN_STAGED(4,16)
GDN_STAGED(8,4)
GDN_STAGED(8,8)
GDN_STAGED(8,16)
#undef GDN_STAGED
kernel void gdn_gate(device const float* x [[buffer(0)]],device const float* z [[buffer(1)]],
    device const uchar* w [[buffer(2)]],device float* out [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 gid [[thread_position_in_grid]]) {
    uint h=gid.x/32,lane=gid.x%32,t=gid.y; if(h>=48 || t>=p[0]) return;
    uint base=t*6144+h*128;
    float inv=precise::rsqrt(rms_square_sum(x+base,128,lane)/128+1e-6f);
    for(uint d=lane;d<128;d+=32) out[base+d]=bf(bf(bf(x[base+d]*inv)*scalar(w,d,p[1]))*sigmoid_f(z[base+d]));
}
kernel void route(device const float* logits [[buffer(0)]],device int* ids [[buffer(1)]],
    device float* weights [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint t=gid/32; if(t>=p[0]) return;
    threadgroup float scores[10]; threadgroup int selected[10];
    if(!lane) {
        for(uint j=0;j<10;++j) {scores[j]=-INFINITY;selected[j]=-1;}
        for(int e=0;e<512;++e) {
            float v=logits[t*512+uint(e)];
            for(int j=0;j<10;++j) if(v>scores[j]) {
                for(int k=9;k>j;--k) {scores[k]=scores[k-1];selected[k]=selected[k-1];}
                scores[j]=v; selected[j]=e; break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // MLX softmax uses four values per lane, a SIMD sum, and a reciprocal.
    float values[4],sum=0;
    for(uint i=0;i<4;++i) {uint j=lane*4+i;values[i]=j<10?fast::exp(scores[j]-scores[0]):0;sum+=values[i];}
    float inverse=1.0f/simd_sum(sum);
    for(uint i=0;i<4;++i) {
        uint j=lane*4+i;if(j<10) {ids[t*10+j]=selected[j];weights[t*10+j]=values[i]*inverse;}
    }
}
// One SIMD group per token. Only selection is parallel: scores are reloaded
// from their original indices, then use the reference softmax below unchanged.
kernel void route_simd(device const float* logits [[buffer(0)]],device int* ids [[buffer(1)]],
    device float* weights [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint t=gid/32; if(t>=p[0]) return;
    threadgroup float scores[10]; threadgroup int selected[10];
    float candidates[16];
    for(uint i=0;i<16;++i) candidates[i]=logits[t*512+lane+i*32];
    for(uint j=0;j<10;++j) {
        float best=-INFINITY;uint index=512;
        for(uint i=0;i<16;++i) if(candidates[i]>best) {best=candidates[i];index=lane+i*32;}
        float maximum=simd_max(best);
        uint winner=simd_min(best==maximum?index:512u);
        if(!lane) {
            selected[j]=winner<512?int(winner):-1;
            // Preserve signed zero and the original tie order. NaN and -inf
            // never beat the serial insertion kernel's -inf sentinel.
            scores[j]=winner<512?logits[t*512+winner]:-INFINITY;
        }
        for(uint i=0;i<16;++i) if(lane+i*32==winner) candidates[i]=-INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float values[4],sum=0;
    for(uint i=0;i<4;++i) {uint j=lane*4+i;values[i]=j<10?fast::exp(scores[j]-scores[0]):0;sum+=values[i];}
    float inverse=1.0f/simd_sum(sum);
    for(uint i=0;i<4;++i) {
        uint j=lane*4+i;if(j<10) {ids[t*10+j]=selected[j];weights[t*10+j]=values[i]*inverse;}
    }
}
kernel void gather_rows(device const float* x [[buffer(0)]],device const int* rows [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 gid [[thread_position_in_grid]]) {
    if(gid.x<p[0] && gid.y<p[1]) out[gid.y*p[0]+gid.x]=x[uint(rows[gid.y])*p[0]+gid.x];
}
kernel void scatter_experts(device const float* x [[buffer(0)]],device const int* positions [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 gid [[thread_position_in_grid]]) {
    if(gid.x<2560 && gid.y<p[0]) out[uint(positions[gid.y])*2560+gid.x]=x[gid.y*2560+gid.x];
}
kernel void moe_sum(device const float* experts [[buffer(0)]],device const float* weights [[buffer(1)]],
    device const float* shared [[buffer(2)]],device const float* gate [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint2 gid [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if(gid.x>=2560 || gid.y>=p[0]) return;
    // Match MLX's eight-row strided reduction: rows 0/8 and 1/9 are
    // accumulated first, then the eight partials are combined in order.
    float products[10];
    for(uint k=0;k<10;++k) products[k]=experts[(gid.y*10+k)*2560+gid.x]*weights[gid.y*10+k];
    float sum=(products[0]+products[8])+(products[1]+products[9]);
    for(uint k=2;k<8;++k) sum+=products[k];
    out[gid.y*2560+gid.x]=bf(bf(sum)+bf(bf(sigmoid_bf(gate[gid.y]))*shared[gid.y*2560+gid.x]));
}
kernel void norm_rope(device const float* x [[buffer(0)]],device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 gid [[thread_position_in_grid]]) {
    uint head=gid.x/32,lane=gid.x%32,t=gid.y,D=p[0],H=p[1],inwidth=p[2],stride=p[3],offset=p[4];
    if(head>=H || t>=p[5]) return;
    uint base=t*inwidth+head*stride;
    float inv=precise::rsqrt(rms_square_sum(x+base,D,lane)/float(D)+1e-6f);
    for(uint d=lane;d<D;d+=32) {
        float v=bf(bf(x[base+d]*inv)*scalar(w,d,p[6]));
        if(d<64) {
            uint pair=d<32?d+32:d-32;
            float other=bf(bf(x[base+pair]*inv)*scalar(w,pair,p[6]));
            float angle=float(offset+t)*pow(10000000.0f,-float(d%32)/32.0f);
            v=bf(bf(v*bf(cos(angle)))+bf((d<32?-other:other)*bf(sin(angle))));
        }
        out[(t*H+head)*D+d]=v;
    }
}
kernel void kv_store(device const float* x [[buffer(0)]],device float* cache [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint2 gid [[thread_position_in_grid]]) {
    if(gid.x<p[0] && gid.y<p[1]) cache[(p[2]+gid.y)*p[0]+gid.x]=x[gid.y*p[0]+gid.x];
}
kernel void index_store(device const float* x [[buffer(0)]],device float* cache [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint2 gid [[thread_position_in_grid]]) {
    if(gid.x<128 && gid.y<p[0]) cache[(p[1]+gid.y)*128+gid.x]=x[gid.y*640+512+gid.x];
}
kernel void index_pool(device const float* raw [[buffer(0)]],device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
    uint block=gid/32,lane=gid%32; if(block>=p[0]) return;
    float v[4],ss=0;
    for(uint j=0;j<4;++j) {
        uint d=lane+j*32; float sum=0;
        for(uint k=0;k<4;++k) sum+=raw[(block*4+k)*128+d];
        v[j]=bf(sum/4); ss+=v[j]*v[j];
    }
    float inv=rsqrt(simd_sum(ss)/128+1e-6f);
    for(uint j=0;j<4;++j) {
        uint d=lane+j*32; float value=bf(bf(v[j]*inv)*scalar(w,d,p[1]));
        if(d<64) {
            uint pair=d<32?d+32:d-32; float other=bf(bf(v[d<32?1:0]*inv)*scalar(w,pair,p[1]));
            float angle=float(block*4)*pow(10000000.0f,-float(d%32)/32.0f);
            value=bf(bf(value*bf(cos(angle)))+bf((d<32?-other:other)*bf(sin(angle))));
        }
        out[block*128+d]=value;
    }
}
kernel void index_scores(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
    device float* scores [[buffer(2)]],constant uint* p [[buffer(3)]],uint2 gid [[thread_position_in_grid]]) {
    uint block=gid.x/32,lane=gid.x%32,t=gid.y; if(block>=p[0] || t>=p[1]) return;
    float sum=0;
    for(uint h=0;h<4;++h) {
        float dot=0; for(uint d=lane;d<128;d+=32) dot+=q[(t*4+h)*128+d]*k[block*128+d];
        sum+=max(simd_sum(dot),0.0f);
    }
    if(!lane) scores[t*p[0]+block]=(block*4+3<=p[2]+t)?sum*0.08838834764831845f:-INFINITY;
}
// The CPU oracle ranks before causal filtering. Padded entries sort last even
// when real entries are -infinity; invalid inputs flag failure independently of rank.
// Integer keys preserve subnormal ordering on GPUs that flush floating operands.
// Canonicalize signed zero, then map IEEE sign/magnitude to increasing unsigned keys.
inline bool sparse_before(uint a,uint ai,uint b,uint bi) { return a==b?ai<bi:a>b; }
kernel void sparse_select(device const uint* input [[buffer(0)]],device uchar* mask [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],constant uint* p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint values[2048];threadgroup uint indices[2048];
    uint t=group.y,L=p[2],blocks=L/4,count=1;
    while(count<blocks) count<<=1;
    for(uint i=lane;i<count;i+=256) {
        uint bits=i<blocks?input[t*blocks+i]:0xff800000u;
        if((bits&0x7fffffffu)>0x7f800000u || bits==0x7f800000u) {
            atomic_fetch_or_explicit(status,1u,memory_order_relaxed);bits=0xff800000u;
        }
        if(!(bits&0x7fffffffu)) bits=0;
        values[i]=(bits&0x80000000u)?~bits:(bits^0x80000000u);indices[i]=i<blocks?i:0xffffffffu;
    }
    for(uint i=lane;i<L;i+=256) mask[t*L+i]=0;
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    for(uint width=2;width<=count;width<<=1) for(uint step=width/2;step;step>>=1) {
        for(uint i=lane;i<count;i+=256) {
            uint j=i^step;
            if(j>i) {
                bool before=sparse_before(values[i],indices[i],values[j],indices[j]);
                if(before!=((i&width)==0)) {
                    uint v=values[i];values[i]=values[j];values[j]=v;
                    uint index=indices[i];indices[i]=indices[j];indices[j]=index;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint rank=lane;rank<min(512u,blocks);rank+=256) {
        uint b=indices[rank];
        if(values[rank]>0x007fffffu && b<blocks && b*4+3<=p[1]+t)
            for(uint d=0;d<4;++d) mask[t*L+b*4+d]=1;
    }
    if(!lane) for(uint i=((p[1]+t+1)/4)*4;i<=p[1]+t;++i) mask[t*L+i]=1;
}
// Explicit BF16 score/probability boundaries follow the original checkpoint's
// MLX attention fallback (256-wide heads, GQA=12, prefill >2 tokens). Keep the
// same arithmetic during decoding so splitting a prefix cannot change it.
// The score buffer is bounded by chunk * 24 * context and charged to scratch.
kernel void attention_scores(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],device float* scores [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint col=group.x*8,row=group.y*8,h=group.z,T=p[0],L=p[2];
    uint qid=lane/4,fr=(qid&4)+((lane/2)%4),fc=(qid&2)*2+(lane%2)*2;
    simdgroup_float8x8 a,b,c;c.thread_elements()[0]=0;c.thread_elements()[1]=0;
    for(uint d=0;d<256;d+=8) {
        for(uint j=0;j<2;++j) {
            a.thread_elements()[j]=row+fr<T?q[((row+fr)*24+h)*256+d+fc+j]*0.0625f:0;
            b.thread_elements()[j]=col+fc+j<L?k[((col+fc+j)*2+h/12)*256+d+fr]:0;
        }
        simdgroup_multiply_accumulate(c,a,b,c);
    }
    for(uint j=0;j<2;++j) if(row+fr<T && col+fc+j<L) {
        uint t=row+fr,key=col+fc+j;
        bool visible=key<=p[1]+t && (!p[3] || mask[t*L+key]);
        scores[(t*24+h)*L+key]=visible?bf(c.thread_elements()[j]):-INFINITY;
    }
}
kernel void attention_scores_skip_masked(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],device float* scores [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint col=group.x*8,row=group.y*8,h=group.z,T=p[0],L=p[2];
    uint qid=lane/4,fr=(qid&4)+((lane/2)%4),fc=(qid&2)*2+(lane%2)*2;
    bool visible=false;
    for(uint j=0;j<2;++j) if(row+fr<T && col+fc+j<L) {
        uint t=row+fr,key=col+fc+j;
        visible|=key<=p[1]+t && (!p[3] || mask[t*L+key]);
    }
    if(!simd_any(visible)) {
        for(uint j=0;j<2;++j) if(row+fr<T && col+fc+j<L)
            scores[((row+fr)*24+h)*L+col+fc+j]=-INFINITY;
        return;
    }
    simdgroup_float8x8 a,b,c;c.thread_elements()[0]=0;c.thread_elements()[1]=0;
    for(uint d=0;d<256;d+=8) {
        for(uint j=0;j<2;++j) {
            a.thread_elements()[j]=row+fr<T?q[((row+fr)*24+h)*256+d+fc+j]*0.0625f:0;
            b.thread_elements()[j]=col+fc+j<L?k[((col+fc+j)*2+h/12)*256+d+fr]:0;
        }
        simdgroup_multiply_accumulate(c,a,b,c);
    }
    for(uint j=0;j<2;++j) if(row+fr<T && col+fc+j<L) {
        uint t=row+fr,key=col+fc+j;
        bool visible=key<=p[1]+t && (!p[3] || mask[t*L+key]);
        scores[(t*24+h)*L+key]=visible?bf(c.thread_elements()[j]):-INFINITY;
    }
}
kernel void attention_softmax(device float* scores [[buffer(0)]],constant uint* p [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint L=p[0];float maximum=-INFINITY;
    for(uint base=0;base<L;base+=128) for(uint j=0;j<4;++j) {
        uint key=base+lane*4+j;if(key<L) maximum=max(maximum,scores[row*L+key]);
    }
    maximum=simd_max(maximum);float normalizer=0;
    // Four contiguous values per lane, then a SIMD reduction. Zero masked
    // tails leave the summation order independent of the current chunk size.
    for(uint base=0;base<L;base+=128) {
        float partial=0;
        for(uint j=0;j<4;++j) {uint key=base+lane*4+j;if(key<L) partial+=fast::exp(scores[row*L+key]-maximum);}
        normalizer+=simd_sum(partial);
    }
    float inverse=1.0f/normalizer;
    for(uint key=lane;key<L;key+=32) scores[row*L+key]=bf(fast::exp(scores[row*L+key]-maximum)*inverse);
}
kernel void attention_values(device const float* probs [[buffer(0)]],device const float* v [[buffer(1)]],
    device const float* qg [[buffer(2)]],device float* out [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint col=group.x*8,row=group.y*8,h=group.z,T=p[0],L=p[1];
    uint qid=lane/4,fr=(qid&4)+((lane/2)%4),fc=(qid&2)*2+(lane%2)*2;
    simdgroup_float8x8 a,b,c;c.thread_elements()[0]=0;c.thread_elements()[1]=0;
    for(uint key=0;key<L;key+=8) {
        for(uint j=0;j<2;++j) {
            a.thread_elements()[j]=row+fr<T && key+fc+j<L?probs[((row+fr)*24+h)*L+key+fc+j]:0;
            b.thread_elements()[j]=key+fr<L?v[((key+fr)*2+h/12)*256+col+fc+j]:0;
        }
        simdgroup_multiply_accumulate(c,a,b,c);
    }
    for(uint j=0;j<2;++j) if(row+fr<T) {
        uint t=row+fr,d=col+fc+j;
        out[(t*24+h)*256+d]=bf(bf(c.thread_elements()[j])*sigmoid_bf(qg[t*12288+h*512+256+d]));
    }
}
kernel void ple_gate(device const float* key [[buffer(0)]],device const float* query [[buffer(1)]],
    device const float* value [[buffer(2)]],device float* out [[buffer(3)]],
    constant uint* p [[buffer(4)]],uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint tid=lid.x,h=gid.x,t=gid.y; if(h>=4 || t>=p[0]) return;
    // MLX 0.31.1 row_reduce_simple: 640 threads each add four adjacent
    // BF16 products with BF16 local accumulation. Each of the twenty SIMD
    // partials and the final reduction also round to BF16. A single FP32
    // dot misses a real mixed-checkpoint gate boundary (-314 versus -312).
    threadgroup float partials[20];
    uint base=t*10240+h*2560; float sum=0;
    for(uint i=0;i<4;++i) {uint d=tid*4+i;sum=bf(bf(key[base+d]*query[base+d])+sum);}
    sum=bf(simd_sum(sum));
    if(!lane) partials[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!simd) {
        sum=bf(simd_sum(lane<20?partials[lane]:0.0f));
        if(!lane) partials[0]=sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gate=bf(partials[0]/bf(sqrt(2560.0f)));
    gate=bf(bf(sqrt(max(abs(gate),1e-6f)))*sign(gate)); gate=bf(sigmoid_bf(gate));
    for(uint d=tid;d<2560;d+=640) out[base+d]=bf(gate*value[t*2560+d]);
}

// Independent expert coordinate; the original affine dot and BF16 boundaries
// are shared with the reference kernels. No cross-expert accumulation here.
kernel void q4_expert_gate_group(device const uchar* r0 [[buffer(0)]],
    device const uchar* r1 [[buffer(1)]],
    device const uchar* r2 [[buffer(2)]],
    device const uchar* r3 [[buffer(3)]],
    device const uchar* r4 [[buffer(4)]],
    device const uchar* r5 [[buffer(5)]],
    device const uchar* r6 [[buffer(6)]],
    device const uchar* r7 [[buffer(7)]],
    device const float* x [[buffer(8)]],device float* out [[buffer(9)]],
    constant uint* p [[buffer(10)]],uint2 gid [[thread_position_in_grid]]) {
    uint row=gid.x/32,lane=gid.x%32,e=gid.y;if(row>=640 || e>=p[0]) return;
    device const uchar* r=e==0?r0:e==1?r1:e==2?r2:e==3?r3:e==4?r4:e==5?r5:e==6?r6:r7;
    if(p[7]) {
        float value=affine_gate_pair<4>(r+p[1],(device const ushort*)(r+p[2]),(device const ushort*)(r+p[3]),
            r+p[4],(device const ushort*)(r+p[5]),(device const ushort*)(r+p[6]),x,row,2560,640,64,lane);
        if(!lane) out[e*640+row]=value;return;
    }
    float gate=bf(affine_dot((device const uint*)(r+p[1]),(device const ushort*)(r+p[2]),(device const ushort*)(r+p[3]),x,row,2560,640,64,lane));
    float up=bf(affine_dot((device const uint*)(r+p[4]),(device const ushort*)(r+p[5]),(device const ushort*)(r+p[6]),x,row,2560,640,64,lane));
    if(!lane) out[e*640+row]=bf(bf(gate*sigmoid_bf(gate))*up);
}
kernel void q4_expert_down_group(device const uchar* r0 [[buffer(0)]],
    device const uchar* r1 [[buffer(1)]],
    device const uchar* r2 [[buffer(2)]],
    device const uchar* r3 [[buffer(3)]],
    device const uchar* r4 [[buffer(4)]],
    device const uchar* r5 [[buffer(5)]],
    device const uchar* r6 [[buffer(6)]],
    device const uchar* r7 [[buffer(7)]],
    device const float* x [[buffer(8)]],device float* out [[buffer(9)]],
    constant uint* p [[buffer(10)]],uint2 gid [[thread_position_in_grid]]) {
    uint row=(gid.x/32)*p[12],lane=gid.x%32,e=gid.y;if(row>=2560 || e>=p[0]) return;
    device const uchar* r=e==0?r0:e==1?r1:e==2?r2:e==3?r3:e==4?r4:e==5?r5:e==6?r6:r7;
    if(p[12]==2) {float sums[2];
        affine_block<4,2,1>(r+p[1],(device const ushort*)(r+p[2]),(device const ushort*)(r+p[3]),x+e*640,row,0,640,2560,1,64,lane,sums);
        if(!lane) for(uint i=0;i<2 && row+i<2560;++i) out[p[4+e]*2560+row+i]=bf(sums[i]);return;
    }
    float value=affine_dot((device const uint*)(r+p[1]),(device const ushort*)(r+p[2]),(device const ushort*)(r+p[3]),x+e*640,row,640,2560,64,lane);
    if(!lane) out[p[4+e]*2560+row]=bf(value);
}

// Exact fixed-shape decode kernels; arithmetic sealed by the packed-Q4 probe.
kernel void q4_gate_up_packed_r2(device const uint* gw [[buffer(0)]],device const ushort* gs [[buffer(1)]],
    device const ushort* gb [[buffer(2)]],device const uint* uw [[buffer(3)]],device const ushort* us [[buffer(4)]],
    device const ushort* ub [[buffer(5)]],device const float* x [[buffer(6)]],device const int* rows [[buffer(7)]],
    device float* out [[buffer(8)]],constant uint* p [[buffer(9)]],uint tid [[thread_position_in_grid]]) {
    const uint K=2560,N=640,row0=(tid/32)*2,lane=tid%32;
    if(row0>=N)return;
    float gate[2]={0,0},up[2]={0,0};
    for(uint base=lane*16;base<K;base+=512) {
        uint gwords[2][2],uwords[2][2];float gd[2]={0,0},ud[2]={0,0},sum=0;
        #pragma unroll
        for(uint r=0;r<2;++r) {
            #pragma unroll
            for(uint j=0;j<2;++j) {
                gwords[r][j]=gw[(row0+r)*(K/8)+base/8+j];
                uwords[r][j]=uw[(row0+r)*(K/8)+base/8+j];
            }
        }
        #pragma unroll
        for(uint i=0;i<16;i+=4) {
            float a=x[base+i],c=x[base+i+1],d=x[base+i+2],e=x[base+i+3];
            sum+=bf(bf(bf(a+c)+d)+e);
            #pragma unroll
            for(uint r=0;r<2;++r) {
                uint gc=(gwords[r][i/8]>>(4*(i%8)))&0xffff,uc=(uwords[r][i/8]>>(4*(i%8)))&0xffff;
                gd[r]+=a*float(gc&15)+c*float((gc>>4)&15)+d*float((gc>>8)&15)+e*float((gc>>12)&15);
                ud[r]+=a*float(uc&15)+c*float((uc>>4)&15)+d*float((uc>>8)&15)+e*float((uc>>12)&15);
            }
        }
        #pragma unroll
        for(uint r=0;r<2;++r) {
            uint g=(row0+r)*(K/64)+base/64;
            gate[r]+=b16(gs[g])*gd[r]+sum*b16(gb[g]);up[r]+=b16(us[g])*ud[r]+sum*b16(ub[g]);
        }
    }
    #pragma unroll
    for(uint r=0;r<2;++r) {
        float g=bf(simd_sum(gate[r])),u=bf(simd_sum(up[r]));
        if(!lane)out[row0+r]=bf(bf(g*sigmoid_bf(g))*u);
    }
}

kernel void q4_down_packed_r2(device const uint* w [[buffer(0)]],device const ushort* s [[buffer(1)]],
    device const ushort* b [[buffer(2)]],device const float* x [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint tid [[thread_position_in_grid]]) {
    const uint K=640,N=2560,row0=(tid/32)*2,lane=tid%32;
    if(row0>=N)return;
    float result[2]={0,0};
    for(uint base=lane*8;base<K;base+=256) {
        uint words[2]={w[row0*(K/8)+base/8],w[(row0+1)*(K/8)+base/8]};
        float dot[2]={0,0},sum=0;
        #pragma unroll
        for(uint i=0;i<8;i+=4) {
            float a=x[base+i],c=x[base+i+1],d=x[base+i+2],e=x[base+i+3];
            sum+=bf(bf(bf(a+c)+d)+e);
            #pragma unroll
            for(uint r=0;r<2;++r) {
                uint code=(words[r]>>(4*i))&0xffff;
                dot[r]+=a*float(code&15)+c*float((code>>4)&15)+d*float((code>>8)&15)+e*float((code>>12)&15);
            }
        }
        #pragma unroll
        for(uint r=0;r<2;++r) {
            uint g=(row0+r)*(K/64)+base/64;
            result[r]+=b16(s[g])*dot[r]+sum*b16(b[g]);
        }
    }
    #pragma unroll
    for(uint r=0;r<2;++r) {
        float v=simd_sum(result[r]);if(!lane)out[row0+r]=p[4]?v:bf(v);
    }
}
)FREELLM_MSL";
inline constexpr std::string_view ModelLock = R"FREELLM_LOCK({
  "model": "Qwen3.8-Flash-Next",
  "repo": "pipenetwork/Qwen3.8-Flash-Next-MLX-4bit",
  "revision": "aa7c790e804bbf9d491ddb109c3d61bc4a555f7c",
  "slotstream_revision": "9342e70cec78db060f61098ffe011f722ef77cf0",
  "slotstream_baseline_release": "v0.2.11",
  "slotpack_revision": "13ec15dcebdddc817b57f0f9087c5ef82018f10e",
  "ds4_revision": "f62ca29a308724cde5bc99134ede19104b2a3260",
  "files": [
    {
      "path": "LICENSE",
      "size": 3235,
      "sha256": "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08",
      "optional": false
    },
    {
      "path": "README.md",
      "size": 6139,
      "sha256": "db1e0d8575543e7a8d324fd22afd2d596a002e9163ed1f971b6b8df0abbe65fd",
      "optional": false
    },
    {
      "path": "chat_template.jinja",
      "size": 8952,
      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
      "optional": false
    },
    {
      "path": "config.json",
      "size": 33408,
      "sha256": "0da22a8ed4323fbe969bf982aeb054743b315206791f28ef74a309c707080ba5",
      "optional": false
    },
    {
      "path": "generation_config.json",
      "size": 202,
      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
      "optional": false
    },
    {
      "path": "merges.txt",
      "size": 3353259,
      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
      "optional": false
    },
    {
      "path": "model-00001.safetensors",
      "size": 10039592993,
      "sha256": "206c2e6ee138c902115f0686a43e0d56097518945bbcd6d3ab10bf916278f86c",
      "optional": false
    },
    {
      "path": "model-00002.safetensors",
      "size": 10000066971,
      "sha256": "48f925c8652fd3210bd8593b41b9fbd66f02567e0e1e7abc28b75e2601f8af26",
      "optional": false
    },
    {
      "path": "model-00003.safetensors",
      "size": 10000066984,
      "sha256": "cb0a951b9aed16ddad882a75c4c07b2615fa364db7c83cfcf7166dc481def85a",
      "optional": false
    },
    {
      "path": "model-00004.safetensors",
      "size": 10170248438,
      "sha256": "f57df447a6acc2d16e69a66c708291d692f0f7476f74ab20aee3d2554c2896c8",
      "optional": false
    },
    {
      "path": "model-00005.safetensors",
      "size": 10194989755,
      "sha256": "bc133543936364e026b413f1ab4b30f2847846d5a8cc1d201cadfd9396327320",
      "optional": false
    },
    {
      "path": "model-00006.safetensors",
      "size": 10262727991,
      "sha256": "9231085f2723a8a3e26fc00836a789400527320ef8a6e219df6974ea4f8eee95",
      "optional": false
    },
    {
      "path": "model-00007.safetensors",
      "size": 10190937668,
      "sha256": "03dffb750368b02fe517f2f1bfd7d5db69c419630801c3ffce88d8a6971046d5",
      "optional": false
    },
    {
      "path": "model-00008.safetensors",
      "size": 10231122683,
      "sha256": "910401c0e420a50b901b170895e7a180c5908a45634be9c92903977ecaa14986",
      "optional": false
    },
    {
      "path": "model-00009.safetensors",
      "size": 10250305804,
      "sha256": "0c77b22503b8e783cc345cd8afe63e007f9d0d0581cbda3f366250e38e096634",
      "optional": false
    },
    {
      "path": "model-00010.safetensors",
      "size": 10237786674,
      "sha256": "115466ffb3e92a8e2a338d72395f7c32a176dadc791d2049ff9a8daadb347ed7",
      "optional": false
    },
    {
      "path": "model-00011.safetensors",
      "size": 2192353120,
      "sha256": "9128a9cbf81cd25bc4cbebe8adeefa12fa36ee96fcccb6f5fe378343b45131f0",
      "optional": false
    },
    {
      "path": "model.safetensors.index.json",
      "size": 317973,
      "sha256": "072cc2c60b8af6cce82a387f62e39ca88754c3a6fccae0816dd21bb89d27470d",
      "optional": false
    },
    {
      "path": "preprocessor_config.json",
      "size": 390,
      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
      "optional": false
    },
    {
      "path": "qwen4_exp.py",
      "size": 35046,
      "sha256": "6fae4ec0decbf77ca4a4571de683bc5580ec75e84325ecb432dfcd2fc81df75e",
      "optional": false
    },
    {
      "path": "tokenizer.json",
      "size": 12809320,
      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
      "optional": false
    },
    {
      "path": "tokenizer_config.json",
      "size": 17928,
      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
      "optional": false
    },
    {
      "path": "video_preprocessor_config.json",
      "size": 385,
      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
      "optional": false
    },
    {
      "path": "vocab.json",
      "size": 6722759,
      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
      "optional": false
    },
    {
      "path": "mtp.safetensors",
      "size": 1470955171,
      "sha256": "c80b58faae46eeacb94dea49dd3453566ee05597fbd28c7c647eccb2862ab744",
      "optional": true
    }
  ],
  "prepared_control": {
    "format": "freellm-affine-records-v1",
    "recipe": "q4-control-lossless",
    "manifest_sha256": "c4bb4db3220a0de2218128738b2a6ac650a3d65086690c8dc2e8c23d097160da",
    "prepared_bytes": 100048541696
  }
}
)FREELLM_LOCK";
inline constexpr std::string_view MixedModelLock = R"FREELLM_MIXED({
  "model": "Qwen3.8-Flash-Next",
  "repo": "pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit",
  "revision": "b2c422f3c643e36f04227a64d61796b44a4b1029",
  "scope": "Pinned complete mixed 4/8-bit artifact; native full-model qualification pending",
  "files": [
    {
      "path": "LICENSE",
      "size": 3235,
      "sha256": "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"
    },
    {
      "path": "README.md",
      "size": 6195,
      "sha256": "f38cdcf3cea1cb2bb905669a23cda9c738967d3e0696e251f928f69994b84976"
    },
    {
      "path": "chat_template.jinja",
      "size": 8952,
      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    },
    {
      "path": "config.json",
      "size": 135742,
      "sha256": "e3304833db1235a0368fd59bc66d7a8724943262c3e465e629e9e6a892e10ea1"
    },
    {
      "path": "generation_config.json",
      "size": 202,
      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    },
    {
      "path": "merges.txt",
      "size": 3353259,
      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
    },
    {
      "path": "model-00001.safetensors",
      "size": 10131671103,
      "sha256": "f944a290da33b369aca7af4d1d2ac7f35421917cd354e94b2c84104d06589e4f"
    },
    {
      "path": "model-00002.safetensors",
      "size": 10000066971,
      "sha256": "48f925c8652fd3210bd8593b41b9fbd66f02567e0e1e7abc28b75e2601f8af26"
    },
    {
      "path": "model-00003.safetensors",
      "size": 10000066984,
      "sha256": "cb0a951b9aed16ddad882a75c4c07b2615fa364db7c83cfcf7166dc481def85a"
    },
    {
      "path": "model-00004.safetensors",
      "size": 10314427650,
      "sha256": "1b5375ca95f7537986d3d4b04a631d02c5733ec3f51ede0c86af2778fdf2aef5"
    },
    {
      "path": "model-00005.safetensors",
      "size": 10008047700,
      "sha256": "001a119e0e7397bcacffd91b20d246ccfc1260c5fc4f9508925f1b718fd272a4"
    },
    {
      "path": "model-00006.safetensors",
      "size": 10412855614,
      "sha256": "c03d042c95831b6e79eb2e5cb361423d681f55ebecb7905688015ecf8cb47bb5"
    },
    {
      "path": "model-00007.safetensors",
      "size": 10003320652,
      "sha256": "11c2e0ba7d8c24430a1cc3b16f6c127315d16f18109b4871c0ccb7d50203c4e4"
    },
    {
      "path": "model-00008.safetensors",
      "size": 10022166479,
      "sha256": "ebb07a5d262f2627225a72d6dd594b79e88e7291a0f56b0351d56e57b542f5f8"
    },
    {
      "path": "model-00009.safetensors",
      "size": 10413658096,
      "sha256": "750853b2812719c2b76136677f1378bca9c316c3c350588bb50ccce8bdfde187"
    },
    {
      "path": "model-00010.safetensors",
      "size": 10011809426,
      "sha256": "f68e9de983c1cd2cbd56ec4fad8f6da15ee09e3c79732347d63e87021b3d7e3f"
    },
    {
      "path": "model-00011.safetensors",
      "size": 4876940562,
      "sha256": "8e7b68292208146a0298e8e07548efd25889ba1c36d39a5ed27ab46b0d635436"
    },
    {
      "path": "model.safetensors.index.json",
      "size": 317973,
      "sha256": "2e00305e9a602d2b061344445aca52bb4d013a9b78cec6eb2decfbd3b41dfb52"
    },
    {
      "path": "preprocessor_config.json",
      "size": 390,
      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    },
    {
      "path": "qwen4_exp.py",
      "size": 35046,
      "sha256": "6fae4ec0decbf77ca4a4571de683bc5580ec75e84325ecb432dfcd2fc81df75e"
    },
    {
      "path": "tokenizer.json",
      "size": 12809320,
      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
    },
    {
      "path": "tokenizer_config.json",
      "size": 17928,
      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
    },
    {
      "path": "video_preprocessor_config.json",
      "size": 385,
      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    },
    {
      "path": "vocab.json",
      "size": 6722759,
      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
    }
  ]
}
)FREELLM_MIXED";
inline constexpr std::string_view MixedReuseLock = R"FREELLM_REUSE({
  "schema": 1,
  "source_revision": "aa7c790e804bbf9d491ddb109c3d61bc4a555f7c",
  "consumer_revision": "b2c422f3c643e36f04227a64d61796b44a4b1029",
  "file_locks_sha256": {
    "models.lock.json": "3ea4eca6568e725e329be172973f777c7a33572db0e04066d71182f000094d77",
    "mixed-models.lock.json": "a8257f7f5feb9aa5ae9614c684d9ebb332988e3d3a9755368bbeb049aebdb187"
  },
  "prepared_manifest_sha256": "c4bb4db3220a0de2218128738b2a6ac650a3d65086690c8dc2e8c23d097160da",
  "tensor_count": 816,
  "expert_payload_bytes": 67947724800,
  "ngram_payload_bytes": 32000153600,
  "evidence_path": "docs/benchmarks/2026-09-08-mixed/payload-equivalence.json",
  "evidence_sha256": "b07f0427783f312c1fa96b14cb859659a1d713ca9425df8fcef8f77c6c9d8550",
  "scope": "Lossless expert/ngram payload compatibility only; not numerical or performance qualification"
}
)FREELLM_REUSE";
}
