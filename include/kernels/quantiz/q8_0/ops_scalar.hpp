#pragma once

#include "types.hpp"

#include <algorithm>
#include <cstddef>

/**
 * @file quantiz/q8_0/ops_scalar.hpp
 * @brief Q8_0 scalar (non-SIMD) operations
 */

namespace freellm::quant::q8_0 {
namespace scalar {

inline void dequantize_row(const block_q8_0* src, float* dst, size_t k) {
    const size_t nb = num_blocks(k);
    size_t offset = 0;

    for (size_t ib = 0; ib < nb; ++ib) {
        const block_q8_0& block = src[ib];
        const float scale = block.scale();
        const size_t block_elems = std::min(static_cast<size_t>(QK8_0), k - offset);

        for (size_t i = 0; i < block_elems; ++i) {
            dst[offset + i] = scale * static_cast<float>(block.qs[i]);
        }

        offset += block_elems;
    }
}

inline float dot_row(const block_q8_0* row, const float* vec, size_t cols) {
    const size_t blocks = num_blocks(cols);
    float acc = 0.0f;
    size_t col = 0;

    for (size_t b = 0; b < blocks; ++b) {
        const block_q8_0& block = row[b];
        const float scale = block.scale();
        const size_t block_elems = std::min(static_cast<size_t>(QK8_0), cols - col);

        for (size_t i = 0; i < block_elems; ++i) {
            acc += scale * static_cast<float>(block.qs[i]) * vec[col + i];
        }

        col += block_elems;
    }

    return acc;
}

inline void matvec(const void* weights, size_t rows, size_t cols,
                   const float* vec, float* dst) {
    const auto* blocks = static_cast<const block_q8_0*>(weights);
    const size_t blocks_per_row = num_blocks(cols);

    for (size_t r = 0; r < rows; ++r) {
        const block_q8_0* row = blocks + r * blocks_per_row;
        dst[r] = dot_row(row, vec, cols);
    }
}

} // namespace scalar
} // namespace freellm::quant::q8_0

