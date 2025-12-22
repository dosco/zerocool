#pragma once

#include <cstdint>

namespace freellm::quant::q4_0 {

// Q4_0 Block size
constexpr int BLOCK_SIZE = 32;

// Q4_0 Block structure (matches Metal struct)
struct BlockQ4_0 {
    float scale;       // 4 bytes
    uint8_t qs[16];    // 16 bytes (32 * 4 bits)
}; // Total 20 bytes per 32 weights

} // namespace freellm::quant::q4_0
