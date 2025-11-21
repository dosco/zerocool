#pragma once

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

/**
 * @file quantiz/q8_0/quantize.hpp
 * @brief Q8_0 quantization functions (FP32 -> Q8_0)
 */

namespace freellm::quant::q8_0 {

namespace detail {

// Integer rounding helper
inline int nearest_int(float value) {
    return static_cast<int>(std::lrintf(value));
}

template <typename T>
inline T clamp(T value, T low, T high) {
    return std::min(std::max(value, low), high);
}

} // namespace detail

// -----------------------------------------------------------------------------
// Q8_0 quantization (scalar reference path)
// -----------------------------------------------------------------------------

inline void quantize_row(const float* src, void* dst_void, size_t k) {
    if (k == 0) {
        return;
    }

    auto* dst = static_cast<block_q8_0*>(dst_void);
    const size_t nb = num_blocks(k);

    for (size_t ib = 0; ib < nb; ++ib) {
        const size_t offset = ib * QK8_0;
        const size_t reps = std::min(static_cast<size_t>(QK8_0), k - offset);
        float amax = 0.0f;

        for (size_t i = 0; i < reps; ++i) {
            amax = std::max(amax, std::fabs(src[offset + i]));
        }

        const float scale = (reps > 0 && amax > 0.0f) ? amax / 127.0f : 0.0f;
        dst[ib].set_scale(scale);

        const float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;
        for (size_t i = 0; i < reps; ++i) {
            int q = detail::nearest_int(src[offset + i] * inv_scale);
            q = detail::clamp(q, -127, 127);
            dst[ib].qs[i] = static_cast<int8_t>(q);
        }

        for (size_t i = reps; i < QK8_0; ++i) {
            dst[ib].qs[i] = 0;
        }
    }
}

inline void quantize_matrix(const float* src, size_t rows, size_t cols, void* dst) {
    auto* row_dst = static_cast<uint8_t*>(dst);
    const size_t blocks_per_row = num_blocks(cols);
    const size_t row_bytes = blocks_per_row * sizeof(block_q8_0);

    for (size_t r = 0; r < rows; ++r) {
        quantize_row(
            src + r * cols,
            row_dst + r * row_bytes,
            cols);
    }
}

} // namespace freellm::quant::q8_0

