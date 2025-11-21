#pragma once

#include "core/tensor.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

// AVX2 intrinsics for vectorized causal masking
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace freellm {

/**
 * @brief Apply causal masking to attention scores
 * 
 * Sets scores[i, j] = -inf for all j > (i + q_offset) to prevent looking into the future.
 * Supports AVX2 and NEON vectorization.
 */
inline void apply_causal_mask(
    Tensor& scores,
    size_t seq_len_q,
    size_t seq_len_k,
    size_t q_offset
) {
    // Causal mask ensures position i can only attend to positions 0...i (past/current)
    // This prevents "looking into the future" which would break autoregressive generation
    //
    // Without causal masking:
    //   - Tokens can see future tokens during attention
    //   - Model predictions become inconsistent
    //   - Often leads to degenerate behavior (repeated tokens, incoherence)
    //
    // Implementation: Set scores[i, j] = -inf for all j > (i + q_offset) (future positions)
    // After softmax, these positions will have probability ~0
    //
    // KV CACHE SUPPORT:
    // When using KV cache, Q positions are relative (e.g., [0]) but K positions are absolute (e.g., [0,1,2,3,4]).
    // q_offset tells us the absolute position of Q[0]. For example:
    //   - q_offset=4 means Q[0] is at absolute position 4
    //   - Q[0] should see K positions [0,1,2,3,4], mask positions [5,6,7,...]
    //   - Without q_offset, we'd incorrectly mask based on relative position (mask [1,2,3,4])
    //
    // We use two implementations:
    //   1. AVX2 SIMD (8x faster): Vectorized masking using 256-bit registers
    //   2. Scalar fallback: Portable std::fill_n for non-AVX2 systems

#if defined(__AVX2__)
    // AVX2 vectorized implementation: process 8 floats at a time
    const __m256 neg_inf_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

    for (size_t i = 0; i < seq_len_q; ++i) {
        size_t absolute_pos = i + q_offset;  // Absolute position in sequence
        size_t start_mask = absolute_pos + 1;  // Mask positions > absolute_pos

        if (start_mask < seq_len_k) {
            float* row_ptr = &scores[i * seq_len_k + start_mask];
            size_t num_future = seq_len_k - start_mask;
            size_t j = 0;

            // Process 8 floats at a time with AVX2
            constexpr size_t simd_width = 8;
            for (; j + simd_width <= num_future; j += simd_width) {
                _mm256_storeu_ps(row_ptr + j, neg_inf_vec);
            }

            // Handle remaining elements (< 8 floats)
            for (; j < num_future; ++j) {
                row_ptr[j] = -std::numeric_limits<float>::infinity();
            }
        }
    }
#elif defined(__ARM_NEON)
    // NEON vectorized implementation: process 4 floats at a time
    const float32x4_t neg_inf_vec = vdupq_n_f32(-std::numeric_limits<float>::infinity());

    for (size_t i = 0; i < seq_len_q; ++i) {
        size_t absolute_pos = i + q_offset;  // Absolute position in sequence
        size_t start_mask = absolute_pos + 1;  // Mask positions > absolute_pos

        if (start_mask < seq_len_k) {
            float* row_ptr = &scores[i * seq_len_k + start_mask];
            size_t num_future = seq_len_k - start_mask;
            size_t j = 0;

            // Process 4 floats at a time with NEON
            constexpr size_t simd_width = 4;
            for (; j + simd_width <= num_future; j += simd_width) {
                vst1q_f32(row_ptr + j, neg_inf_vec);
            }

            // Handle remaining elements (< 4 floats)
            for (; j < num_future; ++j) {
                row_ptr[j] = -std::numeric_limits<float>::infinity();
            }
        }
    }
#else
    // Scalar fallback: use std::fill_n (still quite fast, uses memset internally)
    for (size_t i = 0; i < seq_len_q; ++i) {
        size_t absolute_pos = i + q_offset;  // Absolute position in sequence
        size_t start_mask = absolute_pos + 1;  // Mask positions > absolute_pos

        if (start_mask < seq_len_k) {
            float* row_ptr = &scores[i * seq_len_k + start_mask];
            size_t num_future = seq_len_k - start_mask;
            std::fill_n(row_ptr, num_future, -std::numeric_limits<float>::infinity());
        }
    }
#endif
}

} // namespace freellm
