#pragma once

#include "common.hpp"
#include "q8_0/types.hpp"
#include "q4_0/types.hpp"
#include "q4_K/types.hpp"

/**
 * @file quantiz/types.hpp
 * @brief Unified quantization types interface
 * 
 * This file provides backward-compatible type aliases and utilities
 * that aggregate all quantization types.
 */

namespace freellm::quant {

// Type aliases for backward compatibility
using block_q8_0 = q8_0::block_q8_0;
using block_q4_0 = q4_0::BlockQ4_0;
using block_q4_K = q4_K::block_q4_K;

// Undefine macros to avoid conflicts when defining constexpr variables
// These macros are only needed within the sub-namespace implementations
#ifdef QK8_0
#undef QK8_0
#endif
#ifdef QK_K
#undef QK_K
#endif

// Constants (redefined as constexpr to avoid macro expansion issues)
constexpr size_t QK8_0 = 32;   // Block size for Q8_0 quantization
constexpr size_t QK4_0 = 32;   // Block size for Q4_0 quantization
constexpr size_t QK_K = 256;   // Block size for Q4_K quantization

// Helper functions that dispatch to the appropriate type
inline size_t num_blocks_q8_0(size_t n) {
    return q8_0::num_blocks(n);
}

inline size_t num_blocks_q4_K(size_t n) {
    return q4_K::num_blocks(n);
}

inline size_t num_blocks_q4_0(size_t n) {
    return (n + QK4_0 - 1) / QK4_0;
}

inline size_t q8_0_memory_size(size_t n) {
    return q8_0::memory_size(n);
}

inline size_t q4_K_memory_size(size_t n) {
    return q4_K::memory_size(n);
}

inline size_t q4_0_memory_size(size_t n) {
    return num_blocks_q4_0(n) * sizeof(block_q4_0);
}

// Get block size for quantization type
inline size_t quant_block_size(QuantType type) {
    switch (type) {
        case QuantType::Q8_0: return QK8_0;
        case QuantType::Q4_0: return QK4_0;
        case QuantType::Q4_K: return QK_K;
        default: return 0;
    }
}

// Get bytes per block for quantization type
inline size_t quant_block_bytes(QuantType type) {
    switch (type) {
        case QuantType::Q8_0: return sizeof(block_q8_0);
        case QuantType::Q4_0: return sizeof(block_q4_0);
        case QuantType::Q4_K: return sizeof(block_q4_K);
        default: return 0;
    }
}

// Calculate memory size for quantized data
inline size_t quant_memory_size(QuantType type, size_t num_elements) {
    switch (type) {
        case QuantType::Q8_0: return q8_0_memory_size(num_elements);
        case QuantType::Q4_0: return q4_0_memory_size(num_elements);
        case QuantType::Q4_K: return q4_K_memory_size(num_elements);
        default: return 0;
    }
}

// Calculate number of blocks
inline size_t quant_num_blocks(QuantType type, size_t num_elements) {
    switch (type) {
        case QuantType::Q8_0: return num_blocks_q8_0(num_elements);
        case QuantType::Q4_0: return num_blocks_q4_0(num_elements);
        case QuantType::Q4_K: return num_blocks_q4_K(num_elements);
        default: return 0;
    }
}

inline size_t quant_blocks_per_row(QuantType type, size_t row_size) {
    switch (type) {
        case QuantType::Q8_0: return num_blocks_q8_0(row_size);
        case QuantType::Q4_0: return num_blocks_q4_0(row_size);
        case QuantType::Q4_K: return num_blocks_q4_K(row_size);
        default: return 0;
    }
}

inline size_t quant_row_bytes(QuantType type, size_t row_size) {
    return quant_blocks_per_row(type, row_size) * quant_block_bytes(type);
}

// Get effective bits per weight (including overhead)
inline float quant_bits_per_weight(QuantType type) {
    switch (type) {
        case QuantType::Q8_0: return 8.5f;
        case QuantType::Q4_0: return 5.0f; // 4 bits + 4 bytes scale / 32 = 4 + 1 = 5 bits
        case QuantType::Q4_K: return 4.5f;
        default: return 32.0f;
    }
}

// Q4_K helper functions (for backward compatibility)
inline void get_scale_min_k4(int block_idx, const uint8_t* scales_array,
                              uint8_t* scale_out, uint8_t* min_out) {
    q4_K::get_scale_min(block_idx, scales_array, scale_out, min_out);
}

inline void set_scale_min_k4(int block_idx, uint8_t* scales_array,
                              uint8_t scale_val, uint8_t min_val) {
    q4_K::set_scale_min(block_idx, scales_array, scale_val, min_val);
}

inline uint8_t get_nibble_low(const uint8_t* data, size_t idx) {
    return q4_K::get_nibble_low(data, idx);
}

inline uint8_t get_nibble_high(const uint8_t* data, size_t idx) {
    return q4_K::get_nibble_high(data, idx);
}

inline uint8_t get_nibble(const uint8_t* data, size_t idx) {
    return q4_K::get_nibble(data, idx);
}

inline void set_nibble(uint8_t* data, size_t idx, uint8_t value) {
    q4_K::set_nibble(data, idx, value);
}

} // namespace freellm::quant

