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

#if defined(__ARM_NEON)
    #include <arm_neon.h>
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

#if defined(__ARM_NEON)

inline void dequantize_row_neon(const block_q8_0* src, float* dst, size_t k) {
    const size_t nb = num_blocks(k);
    size_t offset = 0;

    for (size_t ib = 0; ib < nb; ++ib) {
        const block_q8_0& block = src[ib];
        const float scale = block.scale();
        const float32x4_t scale_vec = vdupq_n_f32(scale);
        const size_t block_elems = std::min(static_cast<size_t>(QK8_0), k - offset);

        size_t i = 0;
        for (; i + 16 <= block_elems; i += 16) {
            const int8x16_t q_vals = vld1q_s8(block.qs + i);
            
            // Low 8
            int16x8_t q_low = vmovl_s8(vget_low_s8(q_vals));
            int32x4_t q_low_low = vmovl_s16(vget_low_s16(q_low));
            int32x4_t q_low_high = vmovl_s16(vget_high_s16(q_low));
            
            float32x4_t f_low_low = vcvtq_f32_s32(q_low_low);
            float32x4_t f_low_high = vcvtq_f32_s32(q_low_high);
            
            vst1q_f32(dst + offset + i, vmulq_f32(f_low_low, scale_vec));
            vst1q_f32(dst + offset + i + 4, vmulq_f32(f_low_high, scale_vec));

            // High 8
            int16x8_t q_high = vmovl_s8(vget_high_s8(q_vals));
            int32x4_t q_high_low = vmovl_s16(vget_low_s16(q_high));
            int32x4_t q_high_high = vmovl_s16(vget_high_s16(q_high));

            float32x4_t f_high_low = vcvtq_f32_s32(q_high_low);
            float32x4_t f_high_high = vcvtq_f32_s32(q_high_high);

            vst1q_f32(dst + offset + i + 8, vmulq_f32(f_high_low, scale_vec));
            vst1q_f32(dst + offset + i + 12, vmulq_f32(f_high_high, scale_vec));
        }

        for (; i < block_elems; ++i) {
            dst[offset + i] = scale * static_cast<float>(block.qs[i]);
        }

        offset += block_elems;
    }
}

inline float dot_row_neon(const block_q8_0* row, const float* vec, size_t cols) {
    const size_t blocks = num_blocks(cols);
    float32x4_t acc = vdupq_n_f32(0.0f);
    float tail = 0.0f;
    size_t col = 0;

    for (size_t b = 0; b < blocks; ++b) {
        const block_q8_0& block = row[b];
        const float scale = block.scale();
        const float32x4_t scale_vec = vdupq_n_f32(scale);
        const size_t block_elems = std::min(static_cast<size_t>(QK8_0), cols - col);

        size_t i = 0;
        for (; i + 16 <= block_elems; i += 16) {
            const int8x16_t q_vals = vld1q_s8(block.qs + i);
            
            // Low 8
            int16x8_t q_low = vmovl_s8(vget_low_s8(q_vals));
            int32x4_t q_low_low = vmovl_s16(vget_low_s16(q_low));
            int32x4_t q_low_high = vmovl_s16(vget_high_s16(q_low));
            
            float32x4_t f_low_low = vmulq_f32(vcvtq_f32_s32(q_low_low), scale_vec);
            float32x4_t f_low_high = vmulq_f32(vcvtq_f32_s32(q_low_high), scale_vec);
            
            float32x4_t v_low_low = vld1q_f32(vec + col + i);
            float32x4_t v_low_high = vld1q_f32(vec + col + i + 4);
            
            acc = vmlaq_f32(acc, f_low_low, v_low_low);
            acc = vmlaq_f32(acc, f_low_high, v_low_high);

            // High 8
            int16x8_t q_high = vmovl_s8(vget_high_s8(q_vals));
            int32x4_t q_high_low = vmovl_s16(vget_low_s16(q_high));
            int32x4_t q_high_high = vmovl_s16(vget_high_s16(q_high));

            float32x4_t f_high_low = vmulq_f32(vcvtq_f32_s32(q_high_low), scale_vec);
            float32x4_t f_high_high = vmulq_f32(vcvtq_f32_s32(q_high_high), scale_vec);

            float32x4_t v_high_low = vld1q_f32(vec + col + i + 8);
            float32x4_t v_high_high = vld1q_f32(vec + col + i + 12);

            acc = vmlaq_f32(acc, f_high_low, v_high_low);
            acc = vmlaq_f32(acc, f_high_high, v_high_high);
        }

        for (; i < block_elems; ++i) {
            tail += scale * static_cast<float>(block.qs[i]) * vec[col + i];
        }

        col += block_elems;
    }

    return tail + vaddvq_f32(acc);
}

inline void matvec_neon(const block_q8_0* weights, size_t rows, size_t cols,
                        const float* vec, float* dst) {
    const size_t blocks_per_row = num_blocks(cols);
    for (size_t r = 0; r < rows; ++r) {
        const block_q8_0* row = weights + r * blocks_per_row;
        dst[r] = dot_row_neon(row, vec, cols);
    }
}

#endif // __ARM_NEON

} // namespace simd
} // namespace freellm::quant::q8_0

