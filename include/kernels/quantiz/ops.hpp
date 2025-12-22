#pragma once

#include "q8_0/ops.hpp"
#include "q4_K/ops.hpp"
#include "q4_0/ops.hpp"

/**
 * @file quantiz/ops.hpp
 * @brief Unified quantization operations interface
 * 
 * This file provides backward-compatible operation functions
 * that dispatch to the appropriate type-specific implementation.
 */

namespace freellm::quant {

// Q8_0 operations
inline void dequantize_row_q8_0(const void* src, float* dst, size_t k) {
    q8_0::dequantize_row(src, dst, k);
}

inline float dot_row_q8_0(const block_q8_0* row, const float* vec, size_t cols) {
    return q8_0::dot_row(row, vec, cols);
}

inline void matvec_q8_0(const void* weights, size_t rows, size_t cols,
                        const float* vec, float* dst) {
    q8_0::matvec(weights, rows, cols, vec, dst);
}

// Q4_K operations
inline void dequantize_row_q4_K(const void* src, float* dst, size_t k) {
    q4_K::dequantize_row(src, dst, k);
}

inline float dot_row_q4_K(const block_q4_K* row, const float* vec, size_t cols) {
    return q4_K::dot_row(row, vec, cols);
}

inline void matvec_q4_K(const void* weights, size_t rows, size_t cols,
                        const float* vec, float* dst) {
    q4_K::matvec(weights, rows, cols, vec, dst);
}

// Q4_0 operations (Scalar only for now)
inline void dequantize_row_q4_0(const void* src, float* dst, size_t k) {
    q4_0::dequantize_row(src, dst, k);
}

} // namespace freellm::quant

