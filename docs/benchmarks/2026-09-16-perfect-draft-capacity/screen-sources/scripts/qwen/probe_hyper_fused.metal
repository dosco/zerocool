// Developer-only decode candidate, appended after kernels/metal/qwen.metal.
// Requires its bf(), b16(), and sigmoid_bf() helpers and the same strict
// floating-point compile settings as the reference (fast math disabled).
//
// Fixed contract: one token; affine Q8 K=320, N=10240, group=64; packed uint
// codes and BF16 scales/biases; BF16-valued float input[320] and normalized
// hyper[10240]; distinct float output[2560]. No input/output aliasing.
// Bindings 0..5 are weights, scales, biases, low input, normalized hyper, output.
// Dispatch 2560*32 threads with a threadgroup size divisible by 32.
// Each SIMD group owns one feature and its four rows h*2560+feature.
//
// Each row uses the q8_mv_packed<2,4> lane partition and scalar accumulation:
// base=lane*4, stride=128, four byte products in ascending order, then the
// affine scale/bias update. Sharing input loads across independent rows does
// not change their reductions. Keep every BF16 boundary from Q8 -> unary(1)
// -> hc_mix, including the ordered h=0,1,2,3 product/add and division by four.
// This is an exactness candidate; standalone byte checks precede timing.
kernel void probe_hyper_up_mix_w4(
    device const uint* weights [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const float* low [[buffer(3)]],
    device const float* normalized [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    const uint feature=tid/32,lane=tid%32;
    if(feature>=2560) return;
    float result[4]={0,0,0,0};
    for(uint base=lane*4;base<320;base+=128) {
        uint words[4];float dot[4]={0,0,0,0};
        #pragma unroll
        for(uint h=0;h<4;++h) {
            const uint row=h*2560+feature;
            words[h]=weights[row*80+base/4];
        }
        float sum=0;
        #pragma unroll
        for(uint i=0;i<4;++i) {
            const float value=low[base+i];sum+=value;
            #pragma unroll
            for(uint h=0;h<4;++h)
                dot[h]+=value*float((words[h]>>(8*i))&255);
        }
        #pragma unroll
        for(uint h=0;h<4;++h) {
            const uint group=(h*2560+feature)*5+base/64;
            result[h]+=b16(scales[group])*dot[h]+sum*b16(biases[group]);
        }
    }
    float mixed=0;
    #pragma unroll
    for(uint h=0;h<4;++h) {
        // The collective runs on every SIMD lane, even though only lane zero
        // consumes its result. Unary(1) rounds sigmoid output once more.
        const float projected=bf(simd_sum(result[h]));
        if(!lane) {
            const float gate=bf(sigmoid_bf(projected));
            mixed=bf(mixed+bf(normalized[h*2560+feature]*gate));
        }
    }
    if(!lane) output[feature]=bf(mixed/4);
}
