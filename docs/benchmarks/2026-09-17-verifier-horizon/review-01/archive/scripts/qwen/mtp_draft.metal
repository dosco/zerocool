// MTP's pre-FC normalization spans all four streams. The target RMS kernel
// intentionally only supports <=4096; do not silently route 10240 through it.
kernel void mtp_wide_norm(device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],device float* out [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
    uint t=gid/32,lane=gid%32;if(t>=p[0]) return;
    float square=0;
    for(uint d=lane;d<10240;d+=32) square+=x[t*10240+d]*x[t*10240+d];
    float inv=precise::rsqrt(simd_sum(square)/10240.0f+1e-6f);
    for(uint d=lane;d<10240;d+=32) out[t*10240+d]=bf(bf(x[t*10240+d]*inv)*w[d]);
}
