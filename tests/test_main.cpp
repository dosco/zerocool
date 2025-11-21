#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../external/doctest.h"

#include <print>
#include <chrono>
#include <format>
#include <filesystem>
#include <cmath>
#include <optional>
#include "../include/core/tensor.hpp"
#include "../include/kernels/tensor_ops.hpp"
#include "../include/kernels/tensor_ops_simd.hpp"
#include "../include/kernels/cpu_features.hpp"
#include "../include/kernels/rope.hpp"
#include "../include/core/layers/attention.hpp"
#include "../include/core/model_config.hpp"
#include "../include/infra/model_loader.hpp"
#include "../include/infra/safetensors_loader.hpp"
#include "../include/core/transformer_block.hpp"
#include "../include/core/llm_model.hpp"
#include "../include/core/sampling.hpp"
#include "../include/core/generation.hpp"
#include "../include/core/tokenizer.hpp"
#include "../include/kernels/quantiz/quant_linear.hpp"
#include "../include/kernels/quantiz/quant_config.hpp"

using namespace freellm;
using namespace freellm::ops;

TEST_CASE("CPU Feature Detection") {
    std::println("\n========================================");
    std::println("CPU Feature Detection");
    std::println("========================================\n");

    const auto& features = cpu::get_cpu_features();
    cpu::print_cpu_features(features);

    // Just verify we can call the functions without crashing
    CHECK(true);
}

TEST_CASE("Basic Tensor Operations") {
    SUBCASE("Element-wise operations") {
        Tensor t1({2, 3}, 2.0f);
        Tensor t2({2, 3}, 3.0f);

        Tensor t3 = add(t1, t2);
        CHECK(t3[0] == doctest::Approx(5.0f));

        Tensor t4 = multiply(t1, t2);
        CHECK(t4[0] == doctest::Approx(6.0f));
    }

    SUBCASE("Activation functions") {
        Tensor t_act({4});
        t_act[0] = -1.0f;
        t_act[1] = 0.0f;
        t_act[2] = 1.0f;
        t_act[3] = 2.0f;

        Tensor relu_out = relu(t_act);
        CHECK(relu_out[0] == doctest::Approx(0.0f));
        CHECK(relu_out[1] == doctest::Approx(0.0f));
        CHECK(relu_out[2] == doctest::Approx(1.0f));
        CHECK(relu_out[3] == doctest::Approx(2.0f));

        Tensor silu_out = silu(t_act);
        // Just verify it runs without error
        CHECK(silu_out.size() == 4);
    }
}

TEST_CASE("Quantized Linear Layers") {
    Tensor weight({3, 5});
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>(i) * 0.2f - 0.8f;
    }

    Tensor input({2, 3});
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i));
    }

    Tensor reference = matmul_naive(input, weight);
    Tensor weight_T = ops::transpose(weight);

    SUBCASE("Q8_0 quantization") {
        std::optional<QuantizedTensor> quant_tensor;
        quant_tensor.emplace(QuantizedTensor::from_tensor(weight_T, quant::QuantType::Q8_0));
        Tensor approx = quant::linear_forward(input, weight, quant_tensor);

        float max_error = 0.0f;
        for (size_t i = 0; i < reference.size(); ++i) {
            max_error = std::max(max_error, std::fabs(reference[i] - approx[i]));
        }

        CHECK(max_error < 0.1f);  // Should be very close
    }

    SUBCASE("Q4_K quantization") {
        std::optional<QuantizedTensor> quant_tensor;
        quant_tensor.emplace(QuantizedTensor::from_tensor(weight_T, quant::QuantType::Q4_K));
        Tensor approx = quant::linear_forward(input, weight, quant_tensor);

        float max_error = 0.0f;
        for (size_t i = 0; i < reference.size(); ++i) {
            max_error = std::max(max_error, std::fabs(reference[i] - approx[i]));
        }

        CHECK(max_error < 1.0f);  // Q4_K has more error
    }
}

TEST_CASE("Matrix Multiplication Benchmark") {
    const size_t M = 256;
    const size_t K = 512;
    const size_t N = 128;

    Tensor A({M, K}, 1.0f);
    Tensor B({K, N}, 1.0f);

    SUBCASE("Naive implementation") {
        Tensor C_naive = matmul_naive(A, B);
        CHECK(C_naive[0] == doctest::Approx(K));
    }

    SUBCASE("AVX2 implementation") {
        const auto& features = cpu::get_cpu_features();
        if (features.avx2) {
            Tensor C_avx2 = simd::matmul_avx2(A, B);
            CHECK(C_avx2[0] == doctest::Approx(K));
        }
    }
}

TEST_CASE("Normalization Layers") {
    SUBCASE("Softmax") {
        Tensor logits({2, 4});
        logits[0] = 1.0f; logits[1] = 2.0f; logits[2] = 3.0f; logits[3] = 4.0f;
        logits[4] = 1.0f; logits[5] = 1.0f; logits[6] = 1.0f; logits[7] = 1.0f;

        Tensor probs = softmax(logits);
        float sum_row0 = probs[0] + probs[1] + probs[2] + probs[3];
        CHECK(sum_row0 == doctest::Approx(1.0f).epsilon(0.0001));
    }

    SUBCASE("RMSNorm") {
        Tensor x({2, 4});
        for (size_t i = 0; i < 8; ++i) {
            x[i] = static_cast<float>(i + 1);
        }

        Tensor weight({4}, 1.0f);
        Tensor normed = rms_norm(x, weight);

        // Just verify it runs without error
        CHECK(normed.size() == x.size());
    }
}

TEST_CASE("Rotary Position Embeddings") {
    const size_t seq_len = 4;
    const size_t n_heads = 2;
    const size_t head_dim = 8;

    RoPECache rope(head_dim, 128);
    Tensor x({seq_len, n_heads, head_dim}, 1.0f);
    Tensor rotated = rope.apply(x, 0);

    CHECK(rotated.shape()[0] == seq_len);
    CHECK(rotated.shape()[1] == n_heads);
    CHECK(rotated.shape()[2] == head_dim);
}

TEST_CASE("Attention Mechanism") {
    const size_t seq_len = 4;
    const size_t n_heads = 2;
    const size_t head_dim = 8;

    Tensor Q({seq_len, n_heads, head_dim}, 1.0f);
    Tensor K({seq_len, n_heads, head_dim}, 1.0f);
    Tensor V({seq_len, n_heads, head_dim}, 2.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor output = scaled_dot_product_attention(Q, K, V, scale, n_heads);

    CHECK(output.shape()[0] == seq_len);
    CHECK(output.shape()[1] == n_heads);
    CHECK(output.shape()[2] == head_dim);
}

TEST_CASE("KV Cache") {
    const size_t max_seq_len = 16;
    const size_t n_kv_heads = 4;
    const size_t head_dim = 64;

    KVCache cache(max_seq_len, n_kv_heads, head_dim);

    REQUIRE(cache.current_length() == 0);

    SUBCASE("Add first token") {
        Tensor K1({1, n_kv_heads, head_dim}, 1.0f);
        Tensor V1({1, n_kv_heads, head_dim}, 2.0f);
        cache.update(K1, V1);

        CHECK(cache.current_length() == 1);

        Tensor K_out1 = cache.get_keys();
        Tensor V_out1 = cache.get_values();

        CHECK(K_out1.at({0, 0, 0}) == doctest::Approx(1.0f));
        CHECK(V_out1.at({0, 0, 0}) == doctest::Approx(2.0f));
    }

    SUBCASE("Add multiple tokens") {
        Tensor K1({1, n_kv_heads, head_dim}, 1.0f);
        Tensor V1({1, n_kv_heads, head_dim}, 2.0f);
        cache.update(K1, V1);

        Tensor K2({3, n_kv_heads, head_dim}, 3.0f);
        Tensor V2({3, n_kv_heads, head_dim}, 4.0f);
        cache.update(K2, V2);

        CHECK(cache.current_length() == 4);

        Tensor K_out = cache.get_keys();
        CHECK(K_out.at({0, 0, 0}) == doctest::Approx(1.0f));
        CHECK(K_out.at({1, 0, 0}) == doctest::Approx(3.0f));
    }

    SUBCASE("Reset cache") {
        Tensor K1({2, n_kv_heads, head_dim}, 1.0f);
        Tensor V1({2, n_kv_heads, head_dim}, 2.0f);
        cache.update(K1, V1);

        cache.reset();
        CHECK(cache.current_length() == 0);
    }
}

TEST_CASE("Model Configuration") {
    ModelConfig config;
    config.validate();

    CHECK(config.vocab_size > 0);
    CHECK(config.n_layers > 0);
    CHECK(config.n_heads > 0);
    CHECK(config.d_model > 0);

    SUBCASE("Test tiny config") {
        ModelConfig test_cfg = ModelConfig::test_tiny();
        CHECK(test_cfg.n_layers > 0);
        CHECK(test_cfg.d_model > 0);
    }
}

TEST_CASE("Model Loader") {
    auto config = ModelConfig::test_tiny();
    auto weights = ModelLoader::create_random_weights(
        config.vocab_size,
        config.n_layers,
        config.d_model,
        config.d_ff
    );

    CHECK(weights.size() > 0);
    CHECK(weights.count("embed.weight") > 0);
}

TEST_CASE("Transformer Block") {
    const size_t d_model = 128;
    const size_t d_ff = 512;
    const size_t n_heads = 4;
    const size_t seq_len = 8;

    TransformerBlock block(d_model, d_ff, n_heads);
    Tensor input({seq_len, d_model}, 1.0f);
    Tensor output = block.forward(input);

    CHECK(output.shape()[0] == seq_len);
    CHECK(output.shape()[1] == d_model);
}

TEST_CASE("Full LLM Model") {
    ModelConfig config = ModelConfig::test_tiny();
    LLMModel model(config);

    std::vector<int> token_ids = {1, 42, 100, 5, 999};
    Tensor logits = model.forward(token_ids);

    CHECK(logits.shape()[0] == token_ids.size());
    CHECK(logits.shape()[1] == config.vocab_size);
}

TEST_CASE("Sampling Methods") {
    Tensor logits({10});
    logits[0] = 5.0f;   // Highest
    logits[1] = 4.5f;
    logits[2] = 4.0f;
    logits[3] = 2.0f;
    logits[4] = 1.0f;
    logits[5] = 0.5f;
    logits[6] = 0.3f;
    logits[7] = 0.1f;
    logits[8] = 0.05f;
    logits[9] = 0.01f;  // Lowest

    SUBCASE("Greedy sampling") {
        int greedy_token = greedy_sample(logits);
        CHECK(greedy_token == 0);  // Should always pick highest
    }

    SUBCASE("Top-K sampling") {
        // Run multiple samples to verify it works
        for (int i = 0; i < 5; ++i) {
            int token = top_k_sample(logits, 3, 1.0f);
            CHECK(token >= 0);
            CHECK(token < 10);
        }
    }

    SUBCASE("Top-P sampling") {
        // Run multiple samples to verify it works
        for (int i = 0; i < 5; ++i) {
            int token = top_p_sample(logits, 0.9f, 1.0f);
            CHECK(token >= 0);
            CHECK(token < 10);
        }
    }
}

TEST_CASE("Text Generation") {
    ModelConfig config = ModelConfig::test_tiny();
    config.n_layers = 2;  // Even smaller for faster testing
    LLMModel model(config);

    std::vector<int> prompt = {1, 2, 3};

    SUBCASE("Greedy generation") {
        std::vector<int> generated = generate_greedy(model, prompt, 10);
        // generate returns prompt + new tokens
        CHECK(generated.size() == prompt.size() + 10);
    }

    SUBCASE("Top-P generation") {
        std::vector<int> generated = generate_top_p(model, prompt, 0.9f, 1.0f, 10);
        // generate returns prompt + new tokens
        CHECK(generated.size() == prompt.size() + 10);
    }
}

TEST_CASE("TinyLLaMA Weight Loading") {
    std::string model_path = "models/tinyllama/model.safetensors";

    if (!std::filesystem::exists(model_path)) {
        MESSAGE("⚠️  TinyLLaMA weights not found - skipping real model test");
        return;
    }

    ModelConfig config = ModelConfig::tinyllama_1_1b();
    quant::QuantConfig quant_config;
    quant_config.default_type = quant::QuantType::Q4_K;
    quant_config.attention = quant::QuantType::Q8_0;
    quant_config.feed_forward = quant::QuantType::Q4_K;
    quant_config.lm_head = quant::QuantType::Q4_K;

    LLMModel model(config, quant_config);
    WeightMap weights = load_safetensors(model_path);
    model.load_weights(weights);

    // Verify the model was loaded successfully
    CHECK(weights.size() > 0);

    // Test with tokenizer if available
    std::string tokenizer_path = "models/tinyllama/tokenizer.model";
    if (std::filesystem::exists(tokenizer_path)) {
        Tokenizer tokenizer(tokenizer_path);
        CHECK(tokenizer.vocab_size() > 0);

        std::string prompt_text = "Hello";
        std::vector<int> prompt = tokenizer.encode(prompt_text);
        CHECK(prompt.size() > 0);

        // Generate a few tokens
        GenerationConfig gen_config;
        gen_config.method = SamplingMethod::Greedy;
        gen_config.max_new_tokens = 5;
        std::vector<int> generated = generate(model, prompt, gen_config);
        CHECK(generated.size() == 5);
    }
}
