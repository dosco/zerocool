#pragma once

#include "types.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <print>
#include <cstdio>

namespace freellm::quant::q4_0 {

// Helper to quantize a float array to Q4_0
inline void quantize(const float* src, void* dst, size_t n) {
    if (n % BLOCK_SIZE != 0) {
        // Should handle padding, but for now assume aligned
    }

    int n_blocks = n / BLOCK_SIZE;
    BlockQ4_0* blocks = static_cast<BlockQ4_0*>(dst);

    for (int i = 0; i < n_blocks; ++i) {
        const float* src_block = src + i * BLOCK_SIZE;
        BlockQ4_0& block = blocks[i];

        // Find max abs value
        float max_abs = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            max_abs = std::max(max_abs, std::abs(src_block[j]));
        }

        float scale = max_abs / 7.0f; // -7 to +7
        
        if (std::isnan(scale) || std::isinf(scale)) {
            std::println(stderr, "Warning: NaN/Inf scale detected at block {}", i);
            scale = 0.0f;
        }
        
        block.scale = scale;
        
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;

        for (int j = 0; j < BLOCK_SIZE / 2; ++j) {
            float v0 = src_block[2*j];
            float v1 = src_block[2*j+1];

            int8_t q0 = std::round(v0 * inv_scale) + 8; // Offset to 0..15
            int8_t q1 = std::round(v1 * inv_scale) + 8;

            q0 = std::max(0, std::min(15, (int)q0));
            q1 = std::max(0, std::min(15, (int)q1));

            // Pack: low 4 bits = q0, high 4 bits = q1
            block.qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }
}

inline size_t size_bytes(size_t n_elements) {
    return (n_elements / BLOCK_SIZE) * sizeof(BlockQ4_0);
}

} // namespace freellm::quant::q4_0
