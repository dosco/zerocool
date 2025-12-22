#pragma once

#include <cstdint>
#include <cstddef>

/**
 * @file quantiz/common.hpp
 * @brief Common utilities for quantization (FP16 conversion, QuantType enum)
 */

namespace freellm {
namespace quant {

// =============================================================================
// FP16 Conversion Utilities
// =============================================================================

// Convert FP16 (IEEE 754 half precision) to FP32
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exponent = (h & 0x7C00u) >> 10;
    uint32_t mantissa = (h & 0x03FFu) << 13;

    if (exponent == 0) {
        // Denormalized or zero
        if (mantissa == 0) {
            // Zero
            uint32_t f = sign;
            return *reinterpret_cast<float*>(&f);
        }
        // Denormalized - normalize it
        exponent = 1;
        while ((mantissa & 0x00800000u) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= ~0x00800000u;  // Remove implicit leading 1
        exponent += 127 - 15;
    } else if (exponent == 31) {
        // Infinity or NaN
        exponent = 255;
    } else {
        // Normalized
        exponent += 127 - 15;
    }

    uint32_t f = sign | (exponent << 23) | mantissa;
    return *reinterpret_cast<float*>(&f);
}

// Convert FP32 to FP16 (IEEE 754 half precision)
inline uint16_t fp32_to_fp16(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x & 0x80000000u) >> 16;
    uint32_t exponent = (x & 0x7F800000u) >> 23;
    uint32_t mantissa = (x & 0x007FFFFFu);

    if (exponent == 0) {
        // Zero or denormalized
        return static_cast<uint16_t>(sign);
    } else if (exponent == 255) {
        // Infinity or NaN
        return static_cast<uint16_t>(sign | 0x7C00u | (mantissa ? 0x0200u : 0));
    }

    // Normalized number
    int32_t new_exp = static_cast<int32_t>(exponent) - 127 + 15;

    if (new_exp <= 0) {
        // Underflow to zero
        return static_cast<uint16_t>(sign);
    } else if (new_exp >= 31) {
        // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    uint16_t h = static_cast<uint16_t>(sign | ((new_exp << 10) & 0x7C00u) | (mantissa >> 13));
    return h;
}

// =============================================================================
// Quantization Type Enum
// =============================================================================

enum class QuantType {
    NONE,    // Not quantized (FP32)
    Q8_0,    // 8-bit symmetric
    Q4_0,    // 4-bit symmetric (Metal friendly)
    Q4_K,    // 4-bit k-quant hierarchical
};

// Get string name for quantization type
inline const char* quant_type_name(QuantType type) {
    switch (type) {
        case QuantType::NONE: return "FP32";
        case QuantType::Q8_0: return "Q8_0";
        case QuantType::Q4_0: return "Q4_0";
        case QuantType::Q4_K: return "Q4_K";
        default: return "Unknown";
    }
}

} // namespace quant
} // namespace freellm

