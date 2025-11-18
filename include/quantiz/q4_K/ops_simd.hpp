#pragma once

#include "types.hpp"
#include "common.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
    #if defined(_MSC_VER)
        #include <immintrin.h>
    #else
        #include <x86intrin.h>
    #endif
#endif

/**
 * @file quantiz/q4_K/ops_simd.hpp
 * @brief Q4_K SIMD-optimized operations (AVX2)
 */

namespace freellm::quant::q4_K {
namespace simd {

#if defined(__AVX2__)

inline float hsum256_ps(__m256 v) {
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(vlow, vhigh);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

inline void dequantize_row_avx2(const block_q4_K* src, float* dst, size_t k) {
    const size_t nb = num_blocks(k);
    size_t offset = 0;
    std::array<float, QK_K> scratch{};

    for (size_t ib = 0; ib < nb; ++ib) {
        detail::dequantize_block(src[ib], scratch.data());
        const size_t block_elems = std::min(static_cast<size_t>(QK_K), k - offset);
        std::memcpy(dst + offset, scratch.data(), block_elems * sizeof(float));
        offset += block_elems;
    }
}

inline float dot_row_avx2(const block_q4_K* row, const float* vec, size_t cols) {
    const size_t blocks = num_blocks(cols);
    std::array<float, QK_K> scratch{};
    __m256 acc = _mm256_setzero_ps();
    float tail = 0.0f;
    size_t col = 0;

    for (size_t b = 0; b < blocks; ++b) {
        detail::dequantize_block(row[b], scratch.data());
        const size_t block_elems = std::min(static_cast<size_t>(QK_K), cols - col);

        size_t i = 0;
        for (; i + 8 <= block_elems; i += 8) {
            const __m256 vals = _mm256_loadu_ps(scratch.data() + i);
            const __m256 vec_vals = _mm256_loadu_ps(vec + col + i);
            #ifdef __FMA__
            acc = _mm256_fmadd_ps(vals, vec_vals, acc);
            #else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(vals, vec_vals));
            #endif
        }

        for (; i < block_elems; ++i) {
            tail += scratch[i] * vec[col + i];
        }

        col += block_elems;
    }

    return tail + hsum256_ps(acc);
}

inline void matvec_avx2(const block_q4_K* weights, size_t rows, size_t cols,
                        const float* vec, float* dst) {
    const size_t blocks_per_row = num_blocks(cols);
    for (size_t r = 0; r < rows; ++r) {
        const block_q4_K* row = weights + r * blocks_per_row;
        dst[r] = dot_row_avx2(row, vec, cols);
    }
}

#endif // __AVX2__

} // namespace simd
} // namespace freellm::quant::q4_K

