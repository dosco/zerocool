#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../external/doctest.h"

#include <print>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include "../include/core/tensor.hpp"
#include "../include/kernels/tensor_ops.hpp"
#include "../include/kernels/quantiz/types.hpp"
#include "../include/kernels/quantiz/quantized_tensor.hpp"
#include "../include/kernels/quantiz/quant_linear.hpp"
#include "../include/kernels/quantiz/quant_config.hpp"
#include "../include/kernels/quantiz/quantize.hpp"
#include "../include/kernels/quantiz/ops.hpp"

using namespace freellm;
using namespace freellm::ops;

/**
 * @brief Helper function to calculate maximum absolute error
 */
float max_abs_error(const Tensor& a, const Tensor& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Tensors must have same size");
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, std::fabs(a[i] - b[i]));
    }
    return max_err;
}

/**
 * @brief Helper function to calculate mean absolute error
 */
float mean_abs_error(const Tensor& a, const Tensor& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Tensors must have same size");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::fabs(a[i] - b[i]);
    }
    return sum / a.size();
}

TEST_CASE("Q8_0 Quantization") {
    std::println("\n========================================");
    std::println("Test: Q8_0 Quantization");
    std::println("========================================\n");

    // Create test tensor with known values
    Tensor input({4, 8});  // 32 elements (1 block)
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) - 16.0f;  // Range: -16 to 15
    }

    std::println("Input tensor: shape [{}, {}]", input.shape()[0], input.shape()[1]);
    std::println("Value range: [{:.2f}, {:.2f}]", input[0], input[input.size()-1]);

    // Quantize
    auto start = std::chrono::high_resolution_clock::now();
    QuantizedTensor quantized = QuantizedTensor::from_tensor(input, quant::QuantType::Q8_0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::println("\nQuantization:");
    std::println("  Time: {:.3f} μs", static_cast<float>(duration.count()));
    std::println("  Original size: {} bytes", input.size() * sizeof(float));
    std::println("  Quantized size: {} bytes", quantized.memory_bytes());
    std::println("  Compression ratio: {:.2f}x", quantized.compression_ratio());
    std::println("  Bits per weight: {:.1f}", quant::quant_bits_per_weight(quant::QuantType::Q8_0));

    // Dequantize
    start = std::chrono::high_resolution_clock::now();
    Tensor output = quantized.dequantize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::println("\nDequantization:");
    std::println("  Time: {:.3f} μs", static_cast<float>(duration.count()));

    // Calculate error
    float max_err = max_abs_error(input, output);
    float mean_err = mean_abs_error(input, output);

    std::println("\nAccuracy:");
    std::println("  Max absolute error: {:.6f}", max_err);
    std::println("  Mean absolute error: {:.6f}", mean_err);

    // Q8_0 should be very accurate (8-bit quantization)
    CHECK(max_err < 0.5f);
    std::println("\n✓ Q8_0 quantization test passed!");
}

TEST_CASE("Q4_K Quantization") {
    std::println("\n========================================");
    std::println("Test: Q4_K Quantization");
    std::println("========================================\n");

    // Create test tensor (must be multiple of 256 for Q4_K)
    Tensor input({8, 256});  // 2048 elements (8 blocks)
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = dist(gen);
    }

    std::println("Input tensor: shape [{}, {}]", input.shape()[0], input.shape()[1]);
    std::println("Random normal distribution (mean=0, std=1)");

    // Quantize
    auto start = std::chrono::high_resolution_clock::now();
    QuantizedTensor quantized = QuantizedTensor::from_tensor(input, quant::QuantType::Q4_K);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::println("\nQuantization:");
    std::println("  Time: {:.3f} μs", static_cast<float>(duration.count()));
    std::println("  Original size: {} bytes", input.size() * sizeof(float));
    std::println("  Quantized size: {} bytes", quantized.memory_bytes());
    std::println("  Compression ratio: {:.2f}x", quantized.compression_ratio());
    std::println("  Bits per weight: {:.1f}", quant::quant_bits_per_weight(quant::QuantType::Q4_K));
    std::println("  Number of blocks: {}", quantized.num_blocks());

    // Dequantize
    start = std::chrono::high_resolution_clock::now();
    Tensor output = quantized.dequantize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::println("\nDequantization:");
    std::println("  Time: {:.3f} μs", static_cast<float>(duration.count()));

    // Calculate error
    float max_err = max_abs_error(input, output);
    float mean_err = mean_abs_error(input, output);

    std::println("\nAccuracy:");
    std::println("  Max absolute error: {:.6f}", max_err);
    std::println("  Mean absolute error: {:.6f}", mean_err);

    // Q4_K has lower precision but should still be reasonable
    CHECK(mean_err < 0.1f);
    std::println("\n✓ Q4_K quantization test passed!");
}

TEST_CASE("Quantized Linear Layer") {
    std::println("\n========================================");
    std::println("Test: Quantized Linear Layer");
    std::println("========================================\n");

    const size_t batch = 4;
    const size_t in_features = 256;
    const size_t out_features = 128;

    std::println("Linear layer: [{}, {}] → [{}, {}]", batch, in_features, batch, out_features);

    // Create random weight matrix
    Tensor weight({in_features, out_features});
    std::random_device rd;
    std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = dist(gen);
    }

    // Create random input
    Tensor input({batch, in_features});
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = dist(gen);
    }

    // Compute FP32 reference
    std::println("\nComputing FP32 reference...");
    auto start = std::chrono::high_resolution_clock::now();
    Tensor reference = matmul_naive(input, weight);
    auto end = std::chrono::high_resolution_clock::now();
    auto fp32_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::println("  Time: {:.3f} μs", static_cast<float>(fp32_time.count()));

    SUBCASE("Q8_0 quantized linear") {
        std::println("\nTesting Q8_0 quantized linear:");
        Tensor weight_T = ops::transpose(weight);
        std::optional<QuantizedTensor> q8_weight;
        q8_weight.emplace(QuantizedTensor::from_tensor(weight_T, quant::QuantType::Q8_0));

        start = std::chrono::high_resolution_clock::now();
        Tensor q8_output = quant::linear_forward(input, weight, q8_weight);
        end = std::chrono::high_resolution_clock::now();
        auto q8_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float q8_max_err = max_abs_error(reference, q8_output);
        float q8_mean_err = mean_abs_error(reference, q8_output);

        std::println("  Time: {:.3f} μs ({:.2f}x vs FP32)",
                     static_cast<float>(q8_time.count()),
                     static_cast<float>(fp32_time.count()) / static_cast<float>(q8_time.count()));
        std::println("  Max error: {:.6f}", q8_max_err);
        std::println("  Mean error: {:.6f}", q8_mean_err);

        CHECK(q8_mean_err < 0.05f);
    }

    SUBCASE("Q4_K quantized linear") {
        std::println("\nTesting Q4_K quantized linear:");
        Tensor weight_T = ops::transpose(weight);
        std::optional<QuantizedTensor> q4_weight;
        q4_weight.emplace(QuantizedTensor::from_tensor(weight_T, quant::QuantType::Q4_K));

        start = std::chrono::high_resolution_clock::now();
        Tensor q4_output = quant::linear_forward(input, weight, q4_weight);
        end = std::chrono::high_resolution_clock::now();
        auto q4_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float q4_max_err = max_abs_error(reference, q4_output);
        float q4_mean_err = mean_abs_error(reference, q4_output);

        std::println("  Time: {:.3f} μs ({:.2f}x vs FP32)",
                     static_cast<float>(q4_time.count()),
                     static_cast<float>(fp32_time.count()) / static_cast<float>(q4_time.count()));
        std::println("  Max error: {:.6f}", q4_max_err);
        std::println("  Mean error: {:.6f}", q4_mean_err);

        CHECK(q4_mean_err < 0.3f);
    }

    std::println("\n✓ Quantized linear layer test passed!");
}

TEST_CASE("QuantConfig") {
    std::println("\n========================================");
    std::println("Test: QuantConfig");
    std::println("========================================\n");

    SUBCASE("Disabled config") {
        quant::QuantConfig config1 = quant::QuantConfig::Disabled();
        std::println("Test 1: Disabled config");
        std::println("  Enabled: {}", config1.enabled());
        std::println("  Default type: {}", quant::quant_type_name(config1.default_type));

        CHECK_FALSE(config1.enabled());
    }

    SUBCASE("Full Q8_0 config") {
        quant::QuantConfig config2;
        config2.default_type = quant::QuantType::Q8_0;

        std::println("\nTest 2: Full Q8_0 config");
        std::println("  Enabled: {}", config2.enabled());
        std::println("  Default: {}", quant::quant_type_name(config2.default_type));
        std::println("  Embedding: {}", quant::quant_type_name(config2.resolve_embedding()));
        std::println("  Attention: {}", quant::quant_type_name(config2.resolve_attention()));
        std::println("  Feed-forward: {}", quant::quant_type_name(config2.resolve_feed_forward()));
        std::println("  LM head: {}", quant::quant_type_name(config2.resolve_lm_head()));

        CHECK(config2.enabled());
        CHECK(config2.resolve_attention() == quant::QuantType::Q8_0);
    }

    SUBCASE("Mixed precision config") {
        quant::QuantConfig config3;
        config3.default_type = quant::QuantType::Q4_K;
        config3.attention = quant::QuantType::Q8_0;

        std::println("\nTest 3: Mixed precision config");
        std::println("  Default: {}", quant::quant_type_name(config3.default_type));
        std::println("  Embedding: {}", quant::quant_type_name(config3.resolve_embedding()));
        std::println("  Attention: {}", quant::quant_type_name(config3.resolve_attention()));
        std::println("  Feed-forward: {}", quant::quant_type_name(config3.resolve_feed_forward()));
        std::println("  LM head: {}", quant::quant_type_name(config3.resolve_lm_head()));

        CHECK(config3.resolve_embedding() == quant::QuantType::Q4_K);
        CHECK(config3.resolve_attention() == quant::QuantType::Q8_0);
        CHECK(config3.resolve_feed_forward() == quant::QuantType::Q4_K);
    }

    SUBCASE("Selective quantization") {
        quant::QuantConfig config4;
        config4.attention = quant::QuantType::Q8_0;

        std::println("\nTest 4: Selective quantization (attention only)");
        std::println("  Default: {}", quant::quant_type_name(config4.default_type));
        std::println("  Embedding: {}", quant::quant_type_name(config4.resolve_embedding()));
        std::println("  Attention: {}", quant::quant_type_name(config4.resolve_attention()));
        std::println("  Feed-forward: {}", quant::quant_type_name(config4.resolve_feed_forward()));

        CHECK(config4.resolve_embedding() == quant::QuantType::NONE);
        CHECK(config4.resolve_attention() == quant::QuantType::Q8_0);
        CHECK(config4.resolve_feed_forward() == quant::QuantType::NONE);
    }

    std::println("\n✓ QuantConfig test passed!");
}

TEST_CASE("Memory Efficiency") {
    std::println("\n========================================");
    std::println("Test: Memory Efficiency");
    std::println("========================================\n");

    // Simulate model weights (1B parameter model)
    const size_t num_params = 1'000'000'000;  // 1 billion parameters
    const size_t test_size = 10'000'000;      // Test with 10M for speed

    std::println("Simulating model with {} parameters", num_params);
    std::println("Testing with {} parameters\n", test_size);

    // Create test tensor
    Tensor weights({test_size / 256, 256});  // Align to Q4_K block size
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dist(gen);
    }

    size_t fp32_size = weights.size() * sizeof(float);
    std::println("FP32 memory: {:.2f} MB", fp32_size / (1024.0f * 1024.0f));

    SUBCASE("Q8_0 memory efficiency") {
        auto q8_tensor = QuantizedTensor::from_tensor(weights, quant::QuantType::Q8_0);
        std::println("\nQ8_0:");
        std::println("  Memory: {:.2f} MB", q8_tensor.memory_mb());
        std::println("  Compression: {:.2f}x", q8_tensor.compression_ratio());
        std::println("  Bits per weight: {:.1f}", quant::quant_bits_per_weight(quant::QuantType::Q8_0));

        size_t q8_full_size = static_cast<size_t>(
            (num_params * quant::quant_bits_per_weight(quant::QuantType::Q8_0)) / 8.0f
        );
        std::println("  Full model (1B params): {:.2f} GB",
                     q8_full_size / (1024.0f * 1024.0f * 1024.0f));

        CHECK(q8_tensor.compression_ratio() > 1.0f);
    }

    SUBCASE("Q4_K memory efficiency") {
        auto q4_tensor = QuantizedTensor::from_tensor(weights, quant::QuantType::Q4_K);
        std::println("\nQ4_K:");
        std::println("  Memory: {:.2f} MB", q4_tensor.memory_mb());
        std::println("  Compression: {:.2f}x", q4_tensor.compression_ratio());
        std::println("  Bits per weight: {:.1f}", quant::quant_bits_per_weight(quant::QuantType::Q4_K));

        size_t q4_full_size = static_cast<size_t>(
            (num_params * quant::quant_bits_per_weight(quant::QuantType::Q4_K)) / 8.0f
        );
        std::println("  Full model (1B params): {:.2f} GB",
                     q4_full_size / (1024.0f * 1024.0f * 1024.0f));

        CHECK(q4_tensor.compression_ratio() > 1.0f);
    }

    std::println("\n✓ Memory efficiency test passed!");
}

TEST_CASE("Edge Cases") {
    std::println("\n========================================");
    std::println("Test: Edge Cases");
    std::println("========================================\n");

    SUBCASE("Small tensor") {
        std::println("Test 1: Small tensor (< block size)");
        Tensor small({16});  // Smaller than Q8_0 block size (32)
        for (size_t i = 0; i < small.size(); ++i) {
            small[i] = static_cast<float>(i);
        }

        auto q8_small = QuantizedTensor::from_tensor(small, quant::QuantType::Q8_0);
        Tensor dequant_small = q8_small.dequantize();
        float err = max_abs_error(small, dequant_small);
        std::println("  Max error: {:.6f}", err);

        CHECK(err < 1.0f);
    }

    SUBCASE("Large tensor") {
        std::println("\nTest 2: Large tensor");
        Tensor large({1024, 1024});  // 1M elements
        for (size_t i = 0; i < large.size(); ++i) {
            large[i] = std::sin(static_cast<float>(i) * 0.001f);
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto q4_large = QuantizedTensor::from_tensor(large, quant::QuantType::Q4_K);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::println("  Quantization time: {} ms", duration.count());
        std::println("  Memory: {:.2f} MB", q4_large.memory_mb());

        CHECK(q4_large.memory_mb() > 0);
    }

    SUBCASE("Zero tensor") {
        std::println("\nTest 3: Zero tensor");
        Tensor zeros({256});
        auto q8_zeros = QuantizedTensor::from_tensor(zeros, quant::QuantType::Q8_0);
        Tensor dequant_zeros = q8_zeros.dequantize();

        bool all_zeros = true;
        for (size_t i = 0; i < dequant_zeros.size(); ++i) {
            if (std::fabs(dequant_zeros[i]) > 1e-6f) {
                all_zeros = false;
                break;
            }
        }
        std::println("  All zeros preserved: {}", all_zeros ? "yes" : "no");

        CHECK(all_zeros);
    }

    std::println("\n✓ Edge cases test passed!");
}
