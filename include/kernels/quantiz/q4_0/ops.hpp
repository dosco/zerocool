#pragma once

#include "types.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <print>

namespace freellm::quant::q4_0 {

// Helper to dequantize Q4_0 to float array
inline void dequantize_row(const void* src, float* dst, size_t k) {
    const BlockQ4_0* blocks = static_cast<const BlockQ4_0*>(src);
    int n_blocks = k / BLOCK_SIZE;

    for (int i = 0; i < n_blocks; ++i) {
        const BlockQ4_0& block = blocks[i];
        float scale = block.scale;

        for (int j = 0; j < BLOCK_SIZE / 2; ++j) {
            uint8_t packed = block.qs[j];
            int8_t q0 = packed & 0x0F;
            int8_t q1 = (packed >> 4) & 0x0F;

            // (q - 8) * scale
            dst[i * BLOCK_SIZE + 2 * j] = (q0 - 8) * scale;
            dst[i * BLOCK_SIZE + 2 * j + 1] = (q1 - 8) * scale;
        }
    }
}

} // namespace freellm::quant::q4_0
