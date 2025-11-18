#pragma once

#include "q8_0/quantize.hpp"
#include "q4_K/quantize.hpp"

/**
 * @file quantiz/quantize.hpp
 * @brief Unified quantization interface
 * 
 * This file provides backward-compatible quantization functions
 * that dispatch to the appropriate type-specific implementation.
 */

namespace freellm::quant {

// Q8_0 quantization
inline void quantize_row_q8_0(const float* src, void* dst, size_t k) {
    q8_0::quantize_row(src, dst, k);
}

inline void quantize_matrix_q8_0(const float* src, size_t rows, size_t cols, void* dst) {
    q8_0::quantize_matrix(src, rows, cols, dst);
}

// Q4_K quantization
inline void quantize_row_q4_K(const float* src, void* dst, size_t k) {
    q4_K::quantize_row(src, dst, k);
}

inline void quantize_matrix_q4_K(const float* src, size_t rows, size_t cols, void* dst) {
    q4_K::quantize_matrix(src, rows, cols, dst);
}

} // namespace freellm::quant

