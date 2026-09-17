// Developer-only Q8 word loads for four tokens and one output row.
// These three GDN shapes use width eight in the existing affine_multi kernel.
// Preserve its lane partition, scalar additions, SIMD reduction and BF16 round.
kernel void q8_block_packed_t4_w8(device const uint* w [[buffer(0)]],
    device const ushort* s [[buffer(1)]],device const ushort* b [[buffer(2)]],
    device const float* x [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint tid [[thread_position_in_grid]]) {
    const uint K=p[0],N=p[1],row=tid/32,lane=tid%32;
    if(row>=N || p[2]!=4 || p[3]!=64 ||
       !((K==2560 && (N==10240 || N==6144)) || (K==6144 && N==2560))) return;
    float result[4]={0,0,0,0};
    for(uint base=lane*8;base<K;base+=32*8) {
        const uint words[2]={w[row*(K/4)+base/4],w[row*(K/4)+base/4+1]};
        float dot[4]={0,0,0,0},sum[4]={0,0,0,0};
        #pragma unroll
        for(uint i=0;i<8;++i) {
            const uint code=(words[i/4]>>(8*(i%4)))&255;
            #pragma unroll
            for(uint t=0;t<4;++t) {
                const float v=x[t*K+base+i];
                sum[t]+=v;dot[t]+=v*float(code);
            }
        }
        const uint g=row*(K/64)+base/64;
        const float scale=b16(s[g]),bias=b16(b[g]);
        #pragma unroll
        for(uint t=0;t<4;++t) result[t]+=scale*dot[t]+sum[t]*bias;
    }
    #pragma unroll
    for(uint t=0;t<4;++t) {
        const float value=simd_sum(result[t]);
        if(!lane) out[t*N+row]=p[4]?value:bf(value);
    }
}
