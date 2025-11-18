#pragma once

#include "../common.hpp"

#include <cstddef>
#include <cstdint>

/**
 * @file quantiz/q4_K/types.hpp
 * @brief Q4_K quantization block structure and utilities
 *
 * Q4_K: 4-bit Hierarchical Quantization (K-Quants)
 * Block size: 256 elements (super-block of 8 x 32-element blocks)
 * Formula: weight = d * scale_block * q - dmin * min_block
 * Storage: 4 bytes (2x fp16) + 12 bytes (scales) + 128 bytes (4-bit data) = 144 bytes
 * Effective bits per weight: (144 * 8) / 256 = 4.5 bpw
 * Quality: High, ~0.3-0.5 perplexity increase vs FP32
 */

namespace freellm {
namespace quant {
namespace q4_K {

#define QK_K 256

// Number of bytes for storing scales and mins (6 bits each, 8 values = 96 bits = 12 bytes)
#define K_SCALE_SIZE 12

struct block_q4_K {
    uint16_t d;                    // Super-block scale for quantized scales (FP16)
    uint16_t dmin;                 // Super-block scale for quantized mins (FP16)
    uint8_t scales[K_SCALE_SIZE];  // Scales and mins, quantized with 6 bits each
    uint8_t qs[QK_K / 2];          // 4-bit quantized values (nibbles, 2 per byte)

    // Get super-block scale as FP32
    float scale() const {
        return fp16_to_fp32(d);
    }

    // Get super-block min scale as FP32
    float min_scale() const {
        return fp16_to_fp32(dmin);
    }

    // Set super-block scale from FP32
    void set_scale(float s) {
        d = fp32_to_fp16(s);
    }

    // Set super-block min scale from FP32
    void set_min_scale(float s) {
        dmin = fp32_to_fp16(s);
    }
};

static_assert(sizeof(block_q4_K) == 2 * sizeof(uint16_t) + K_SCALE_SIZE + QK_K / 2,
              "block_q4_K size mismatch");

// Calculate number of blocks needed for n elements
inline size_t num_blocks(size_t n) {
    return (n + QK_K - 1) / QK_K;
}

// Calculate total memory needed for Q4_K quantized data
inline size_t memory_size(size_t n) {
    return num_blocks(n) * sizeof(block_q4_K);
}

// =============================================================================
// Helper Functions for Q4_K
// =============================================================================

// Extract 6-bit scale and min from the packed scales array
// The scales array packs 6-bit values using GGUF's specific encoding:
// - 8 scales (6 bits each = 48 bits)
// - 8 mins (6 bits each = 48 bits)
// Total: 96 bits = 12 bytes
//
// The encoding uses a complex bit-packing scheme where the upper 2 bits
// of scales and mins are stored separately to enable efficient SIMD unpacking.
//
// Reference: llama.cpp's get_scale_min_k4 function
inline void get_scale_min(int block_idx, const uint8_t* scales_array,
                          uint8_t* scale_out, uint8_t* min_out) {
    // This follows the exact bit-packing used by llama.cpp/GGUF
    if (block_idx < 4) {
        // For indices 0-3: lower 6 bits are stored directly
        *scale_out = scales_array[block_idx] & 63;
        *min_out = scales_array[block_idx + 4] & 63;
    } else {
        // For indices 4-7: use upper 2 bits from earlier bytes + nibbles from later bytes
        *scale_out = (scales_array[block_idx + 4] & 0xF) | ((scales_array[block_idx - 4] >> 6) << 4);
        *min_out = (scales_array[block_idx + 4] >> 4) | ((scales_array[block_idx] >> 6) << 4);
    }
}

// Pack 6-bit scale and min into the scales array using GGUF's encoding
inline void set_scale_min(int block_idx, uint8_t* scales_array,
                          uint8_t scale_val, uint8_t min_val) {
    // Ensure values fit in 6 bits
    scale_val &= 0x3F;
    min_val &= 0x3F;

    if (block_idx < 4) {
        // For indices 0-3: store lower 6 bits directly
        scales_array[block_idx] = (scales_array[block_idx] & 0xC0) | scale_val;
        scales_array[block_idx + 4] = (scales_array[block_idx + 4] & 0xC0) | min_val;
    } else {
        // For indices 4-7: split into nibbles and upper bits
        // Store lower 4 bits of scale and upper 4 bits of min in scales_array[block_idx + 4]
        scales_array[block_idx + 4] = (scale_val & 0x0F) | ((min_val & 0x0F) << 4);
        
        // Store upper 2 bits of scale in scales_array[block_idx - 4]
        scales_array[block_idx - 4] = (scales_array[block_idx - 4] & 0x3F) | ((scale_val >> 4) << 6);
        
        // Store upper 2 bits of min in scales_array[block_idx]
        scales_array[block_idx] = (scales_array[block_idx] & 0x3F) | ((min_val >> 4) << 6);
    }
}

// Extract lower and upper nibbles (4-bit values) from packed byte array
inline uint8_t get_nibble_low(const uint8_t* data, size_t idx) {
    return data[idx / 2] & 0x0F;
}

inline uint8_t get_nibble_high(const uint8_t* data, size_t idx) {
    return (data[idx / 2] >> 4) & 0x0F;
}

inline uint8_t get_nibble(const uint8_t* data, size_t idx) {
    if (idx % 2 == 0) {
        return get_nibble_low(data, idx);
    } else {
        return get_nibble_high(data, idx);
    }
}

// Set nibble value in packed byte array
inline void set_nibble(uint8_t* data, size_t idx, uint8_t value) {
    value &= 0x0F;  // Ensure 4 bits
    if (idx % 2 == 0) {
        // Lower nibble
        data[idx / 2] = (data[idx / 2] & 0xF0) | value;
    } else {
        // Upper nibble
        data[idx / 2] = (data[idx / 2] & 0x0F) | (value << 4);
    }
}

} // namespace q4_K
} // namespace quant
} // namespace freellm

