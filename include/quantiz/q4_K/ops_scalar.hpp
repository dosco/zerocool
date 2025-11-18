#pragma once

#include "types.hpp"
#include "common.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

/**
 * @file quantiz/q4_K/ops_scalar.hpp
 * @brief Q4_K scalar (non-SIMD) operations
 */

namespace freellm::quant::q4_K {
namespace scalar {

inline void dequantize_row(const block_q4_K* src, float* dst, size_t k) {
    const size_t nb = num_blocks(k);
    size_t offset = 0;
    std::array<float, QK_K> scratch{};

    for (size_t ib = 0; ib < nb; ++ib) {
        detail::dequantize_block(src[ib], scratch.data());
        const size_t block_elems = std::min(static_cast<size_t>(QK_K), k - offset);
        std::copy_n(scratch.data(), block_elems, dst + offset);
        offset += block_elems;
    }
}

inline float dot_row(const block_q4_K* row, const float* vec, size_t cols) {
    const size_t blocks = num_blocks(cols);
    std::array<float, QK_K> scratch{};
    float acc = 0.0f;
    size_t col = 0;

    for (size_t b = 0; b < blocks; ++b) {
        detail::dequantize_block(row[b], scratch.data());
        const size_t block_elems = std::min(static_cast<size_t>(QK_K), cols - col);
        for (size_t i = 0; i < block_elems; ++i) {
            acc += scratch[i] * vec[col + i];
        }
        col += block_elems;
    }

    return acc;
}

inline void matvec(const void* weights, size_t rows, size_t cols,
                   const float* vec, float* dst) {
    const auto* blocks = static_cast<const block_q4_K*>(weights);
    const size_t blocks_per_row = num_blocks(cols);

    for (size_t r = 0; r < rows; ++r) {
        const block_q4_K* row = blocks + r * blocks_per_row;
        dst[r] = dot_row(row, vec, cols);
    }
}

} // namespace scalar
} // namespace freellm::quant::q4_K

