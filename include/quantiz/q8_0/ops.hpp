#pragma once

#include "ops_scalar.hpp"
#include "ops_simd.hpp"
#include "../../cpu_features.hpp"

#include <cstddef>

/**
 * @file quantiz/q8_0/ops.hpp
 * @brief Q8_0 operations dispatch layer (auto-selects scalar or SIMD)
 */

namespace freellm::quant::q8_0 {

inline void dequantize_row(const void* src, float* dst, size_t k) {
#if defined(__AVX2__)
    if (freellm::cpu::get_cpu_features().avx2) {
        simd::dequantize_row_avx2(static_cast<const block_q8_0*>(src), dst, k);
        return;
    }
#endif
    scalar::dequantize_row(static_cast<const block_q8_0*>(src), dst, k);
}

inline float dot_row(const block_q8_0* row, const float* vec, size_t cols) {
#if defined(__AVX2__)
    if (freellm::cpu::get_cpu_features().avx2) {
        return simd::dot_row_avx2(row, vec, cols);
    }
#endif
    return scalar::dot_row(row, vec, cols);
}

inline void matvec(const void* weights, size_t rows, size_t cols,
                   const float* vec, float* dst) {
#if defined(__AVX2__)
    if (freellm::cpu::get_cpu_features().avx2) {
        simd::matvec_avx2(static_cast<const block_q8_0*>(weights), rows, cols, vec, dst);
        return;
    }
#endif
    scalar::matvec(weights, rows, cols, vec, dst);
}

} // namespace freellm::quant::q8_0

