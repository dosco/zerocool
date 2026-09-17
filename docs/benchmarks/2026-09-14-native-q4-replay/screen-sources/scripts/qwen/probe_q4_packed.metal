// Developer candidate, appended to the current native shader by the probe.
// Each SIMD group owns two independent output rows. Packed loads share input
// and bias work; the four-value BF16 sums and all reduction orders stay fixed.
kernel void probe_q4_gate_r2(device const uint* gw [[buffer(0)]],device const ushort* gs [[buffer(1)]],
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

kernel void probe_q4_down_r2(device const uint* w [[buffer(0)]],device const ushort* s [[buffer(1)]],
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
