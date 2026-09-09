#pragma once

#include <cstddef>
#include <string>
#include <stdexcept>

namespace freellm {

/**
 * @brief Configuration for TinyLLaMA-style transformer model
 *
 * TinyLLaMA is a 1.1B parameter model using modern architecture:
 * - RoPE (Rotary Position Embeddings)
 * - RMSNorm (simpler than LayerNorm)
 * - SiLU activation
 * - No biases in linear layers
 *
 * This is a practical target for CPU inference on consumer hardware.
 */
struct ModelConfig {
    // Model identification
    std::string name = "tinyllama-1.1b";

    // Vocabulary and sequence
    size_t vocab_size = 32000;      // Size of vocabulary
    size_t max_seq_len = 2048;      // Maximum context window

    // Model dimensions
    size_t n_layers = 22;           // Number of transformer blocks
    size_t n_heads = 32;            // Number of attention heads (query heads)
    size_t n_kv_heads = 4;          // Number of key/value heads (for GQA)
    size_t d_model = 2048;          // Model/embedding dimension
    size_t d_ff = 5632;             // Feed-forward hidden dimension

    // Architecture specifics
    float norm_eps = 1e-5;
    float rope_theta = 10000.0f;    // RoPE base frequency

    // MoE configuration
    size_t num_experts = 0;           // Number of experts (0 = dense model)
    size_t num_experts_per_token = 0; // Number of active experts per token
    bool norm_topk_prob = true;       // Whether to normalize top-k probabilities (default: true)

    // Weight tying
    bool tie_word_embeddings = false; // Whether lm_head shares weights with token embedding

    /**
     * @brief Get head dimension
     */
    size_t get_head_dim() const {
        if (d_model % n_heads != 0) {
            throw std::runtime_error("d_model must be divisible by n_heads");
        }
        return d_model / n_heads;
    }

    /**
     * @brief Validate configuration
     */
    void validate() const {
        if (vocab_size == 0) {
            throw std::invalid_argument("vocab_size must be > 0");
        }
        if (max_seq_len == 0) {
            throw std::invalid_argument("max_seq_len must be > 0");
        }
        if (n_layers == 0) {
            throw std::invalid_argument("n_layers must be > 0");
        }
        if (n_heads == 0) {
            throw std::invalid_argument("n_heads must be > 0");
        }
        if (n_kv_heads == 0) {
            throw std::invalid_argument("n_kv_heads must be > 0");
        }
        if (d_model == 0) {
            throw std::invalid_argument("d_model must be > 0");
        }
        if (d_model % n_heads != 0) {
            throw std::invalid_argument("d_model must be divisible by n_heads");
        }
        if (n_heads % n_kv_heads != 0) {
            throw std::invalid_argument("n_heads must be divisible by n_kv_heads (for GQA)");
        }
    }

    /**
     * @brief Create TinyLLaMA 1.1B configuration (default)
     */
    static ModelConfig tinyllama_1_1b() {
        ModelConfig config;
        // Default values already match TinyLLaMA 1.1B
        return config;
    }

    /**
     * @brief Create smaller test configuration for development
     */
    static ModelConfig test_tiny() {
        ModelConfig config;
        config.name = "test-tiny";
        config.vocab_size = 1000;
        config.max_seq_len = 128;
        config.n_layers = 4;
        config.n_heads = 8;
        config.n_kv_heads = 8;  // MHA for testing
        config.d_model = 256;
        config.d_ff = 1024;
        return config;
    }
};

} // namespace freellm
