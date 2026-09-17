#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../external/doctest.h"
#include "../include/kernels/quantiz/types.hpp"
#include "../include/kernels/quantiz/q4_0/quantize.hpp"
#include <vector>
#include <cmath>
#include <print>
#include <random>

using namespace freellm;
using namespace freellm::quant;

// Helper to dequantize a block for testing
void dequantize_q4_0_block(const q4_0::BlockQ4_0& block, float* dst) {
    float scale = block.scale;
    for (int j = 0; j < 32; ++j) {
        uint8_t q_packed = block.qs[j / 2];
        int8_t q = (j % 2 == 0) ? (q_packed & 0x0F) : (q_packed >> 4);
        dst[j] = (q - 8) * scale;
    }
}

TEST_CASE("Q4_0 Quantization Logic") {
    SUBCASE("Exact Values") {
        // Test specific values to verify packing logic
        // Block size is 32.
        std::vector<float> src(32);
        // Fill with 0
        std::fill(src.begin(), src.end(), 0.0f);
        
        // Set some specific values
        // Max abs = 7.0 -> Scale = 1.0
        src[0] = 7.0f;  // Should map to 15 (7+8)
        src[1] = -7.0f; // Should map to 1 (-7+8)
        src[2] = 0.0f;  // Should map to 8 (0+8)
        src[3] = 3.5f;  // Should map to 11.5 -> 12 (3.5+8)
        src[4] = -3.5f; // Should map to 4.5 -> 5 (-3.5+8)
        
        std::vector<uint8_t> dst(q4_0::size_bytes(32));
        q4_0::quantize(src.data(), dst.data(), 32);
        
        q4_0::BlockQ4_0* b = (q4_0::BlockQ4_0*)dst.data();
        
        CHECK(b->scale == doctest::Approx(1.0f));
        
        // Check packed values
        // qs[0] contains src[0] (low) and src[1] (high)
        // src[0] = 7.0 -> q = 15 (0xF)
        // src[1] = -7.0 -> q = 1 (0x1)
        // qs[0] = 0xF | (0x1 << 4) = 0x1F
        CHECK(b->qs[0] == 0x1F);
        
        // qs[1] contains src[2] and src[3]
        // src[2] = 0.0 -> q = 8 (0x8)
        // src[3] = 3.5 -> q = 12 (0xC)
        // qs[1] = 0x8 | (0xC << 4) = 0xC8
        CHECK(b->qs[1] == 0xC8);
        
        // qs[2] contains src[4] and src[5]
        // src[4] = -3.5 -> q = 5 (0x5) if floor(x+0.5), but 4 (0x4) if round(-3.5) -> -4
        // std::round(-3.5) = -4.0. -4 + 8 = 4.
        // src[5] = 0.0 -> q = 8 (0x8)
        // qs[2] = 0x4 | (0x8 << 4) = 0x84
        CHECK(b->qs[2] == 0x84);
    }

    SUBCASE("Scale Calculation") {
        std::vector<float> src(32, 0.0f);
        src[0] = 14.0f; // Max abs = 14.0 -> Scale = 2.0
        
        std::vector<uint8_t> dst(q4_0::size_bytes(32));
        q4_0::quantize(src.data(), dst.data(), 32);
        
        q4_0::BlockQ4_0* b = (q4_0::BlockQ4_0*)dst.data();
        CHECK(b->scale == doctest::Approx(2.0f));
        
        // src[0] = 14.0 -> 14.0 / 2.0 = 7.0 -> q = 15
        // src[1] = 0.0 -> 0.0 / 2.0 = 0.0 -> q = 8
        // qs[0] = 0xF | (0x8 << 4) = 0x8F
        CHECK(b->qs[0] == 0x8F);
    }
    
    SUBCASE("Round Trip Accuracy") {
        std::vector<float> src(32);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for(int i=0; i<32; ++i) src[i] = dist(gen);
        
        std::vector<uint8_t> dst(q4_0::size_bytes(32));
        q4_0::quantize(src.data(), dst.data(), 32);
        
        q4_0::BlockQ4_0* b = (q4_0::BlockQ4_0*)dst.data();
        std::vector<float> out(32);
        dequantize_q4_0_block(*b, out.data());
        
        // Check error
        // Max quantization error should be roughly scale / 2
        // scale = max_abs / 7.0
        // error <= max_abs / 14.0
        
        float max_abs = 0.0f;
        for(float v : src) max_abs = std::max(max_abs, std::abs(v));
        float expected_max_err = max_abs / 14.0f + 1e-5f; // Add epsilon
        
        for(int i=0; i<32; ++i) {
            float err = std::abs(src[i] - out[i]);
            CHECK(err <= expected_max_err);
        }
    }
}
