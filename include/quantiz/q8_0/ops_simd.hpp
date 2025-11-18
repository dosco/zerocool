#pragma once

#include "types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
    #if defined(_MSC_VER)
        #include <immintrin.h>
    #else
        #include <x86intrin.h>
    #endif
#endif

/**
 * @file quantiz/q8_0/ops_simd.hpp
 * @brief Q8_0 SIMD-optimized operations (AVX2)
 */

namespace freellm::quant::q8_0 {
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

inline void dequantize_row_avx2(const block_q8_0* src, float* dst, size_t k) {
    const size_t nb = num_blocks(k);
    size_t offset = 0;

    for (size_t ib = 0; ib < nb; ++ib) {
        const block_q8_0& block = src[ib];
        const float scale = block.scale();
        const __m256 scale_vec = _mm256_set1_ps(scale);
        const size_t block_elems = std::min(static_cast<size_t>(QK8_0), k - offset);

        size_t i = 0;
        for (; i + 8 <= block_elems; i += 8) {
            const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.qs + i));
            const __m256i ints = _mm256_cvtepi8_epi32(bytes);
            const __m256 floats = _mm256_mul_ps(_mm256_cvtepi32_ps(ints), scale_vec);
            _mm256_storeu_ps(dst + offset + i, floats);
        }

        for (; i < block_elems; ++i) {
            dst[offset + i] = scale * static_cast<float>(block.qs[i]);
        }

        offset += block_elems;
    }
}

inline float dot_row_avx2(const block_q8_0* row, const float* vec, size_t cols) {
    const size_t blocks = num_blocks(cols);
    __m256 acc = _mm256_setzero_ps();
    float tail = 0.0f;
    size_t col = 0;

    for (size_t b = 0; b < blocks; ++b) {
        const block_q8_0& block = row[b];
        const float scale = block.scale();
        const __m256 scale_vec = _mm256_set1_ps(scale);
        const size_t block_elems = std::min(static_cast<size_t>(QK8_0), cols - col);

        size_t i = 0;
        for (; i + 8 <= block_elems; i += 8) {
            const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.qs + i));
            const __m256i ints = _mm256_cvtepi8_epi32(bytes);
            const __m256 quant_vals = _mm256_mul_ps(_mm256_cvtepi32_ps(ints), scale_vec);
            const __m256 vec_vals = _mm256_loadu_ps(vec + col + i);
            #ifdef __FMA__
            acc = _mm256_fmadd_ps(quant_vals, vec_vals, acc);
            #else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(quant_vals, vec_vals));
            #endif
        }

        for (; i < block_elems; ++i) {
            tail += scale * static_cast<float>(block.qs[i]) * vec[col + i];
        }

        col += block_elems;
    }

    return tail + hsum256_ps(acc);
}

inline void matvec_avx2(const block_q8_0* weights, size_t rows, size_t cols,
                        const float* vec, float* dst) {
    const size_t blocks_per_row = num_blocks(cols);
    for (size_t r = 0; r < rows; ++r) {
        const block_q8_0* row = weights + r * blocks_per_row;
        dst[r] = dot_row_avx2(row, vec, cols);
    }
}

#endif // __AVX2__

} // namespace simd
} // namespace freellm::quant::q8_0

