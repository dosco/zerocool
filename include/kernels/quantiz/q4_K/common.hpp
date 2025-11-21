#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>

/**
 * @file quantiz/q4_K/common.hpp
 * @brief Q4_K common dequantization utilities
 */

namespace freellm::quant::q4_K {
namespace detail {

/**
 * @brief Fully dequantize a single Q4_K super-block into 256 float values.
 *
 * The resulting buffer is laid out contiguously and can be partially consumed
 * by callers (e.g., when the final block in a row is padded).
 */
inline void dequantize_block(const block_q4_K& block, float* dst) {
    const float d = block.scale();
    const float dm = block.min_scale();
    const uint8_t* q = block.qs;

    int scale_index = 0;
    size_t offset = 0;

    for (size_t chunk = 0; chunk < QK_K; chunk += 64) {
        uint8_t sc0 = 0;
        uint8_t m0 = 0;
        get_scale_min(scale_index + 0, block.scales, &sc0, &m0);
        uint8_t sc1 = 0;
        uint8_t m1 = 0;
        get_scale_min(scale_index + 1, block.scales, &sc1, &m1);

        const float d0 = d * static_cast<float>(sc0);
        const float d1 = d * static_cast<float>(sc1);
        const float mshift0 = dm * static_cast<float>(m0);
        const float mshift1 = dm * static_cast<float>(m1);

        for (int i = 0; i < 32; ++i) {
            dst[offset++] = d0 * static_cast<float>(q[i] & 0x0F) - mshift0;
        }
        for (int i = 0; i < 32; ++i) {
            dst[offset++] = d1 * static_cast<float>(q[i] >> 4) - mshift1;
        }

        q += 32;
        scale_index += 2;
    }
}

} // namespace detail
} // namespace freellm::quant::q4_K

