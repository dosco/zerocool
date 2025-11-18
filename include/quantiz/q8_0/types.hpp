#pragma once

#include "../common.hpp"

#include <cstddef>
#include <cstdint>

/**
 * @file quantiz/q8_0/types.hpp
 * @brief Q8_0 quantization block structure and utilities
 *
 * Q8_0: 8-bit Symmetric Quantization
 * Block size: 32 elements
 * Formula: weight = scale * int8_value
 * Storage: 2 bytes (fp16 scale) + 32 bytes (int8 data) = 34 bytes per block
 * Effective bits per weight: (34 * 8) / 32 = 8.5 bpw
 * Quality: Very high, ~0.1-0.3 perplexity increase
 */

namespace freellm {
namespace quant {
namespace q8_0 {

#define QK8_0 32

struct block_q8_0 {
    uint16_t d;           // Delta (scale factor) in FP16 format
    int8_t qs[QK8_0];     // Quantized values (signed 8-bit)

    // Get scale as FP32
    float scale() const {
        return fp16_to_fp32(d);
    }

    // Set scale from FP32
    void set_scale(float s) {
        d = fp32_to_fp16(s);
    }
};

static_assert(sizeof(block_q8_0) == 2 + QK8_0 * sizeof(int8_t),
              "block_q8_0 size mismatch");

// Calculate number of blocks needed for n elements
inline size_t num_blocks(size_t n) {
    return (n + QK8_0 - 1) / QK8_0;
}

// Calculate total memory needed for Q8_0 quantized data
inline size_t memory_size(size_t n) {
    return num_blocks(n) * sizeof(block_q8_0);
}

} // namespace q8_0
} // namespace quant
} // namespace freellm

