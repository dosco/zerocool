#pragma once

#include "core/llm_model.hpp"
#include "core/sampling.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <print>

namespace freellm {

/**
 * @brief Sampling method selection
 */
enum class SamplingMethod {
    Greedy,   // Deterministic: always pick highest probability
    TopK,     // Sample from K most likely tokens
    TopP      // Nucleus sampling: sample from smallest set with cumulative prob ≥ P
};

/**
 * @brief Configuration for text generation
 */
struct GenerationConfig {
    SamplingMethod method = SamplingMethod::TopP;  // Sampling strategy
    size_t max_new_tokens = 50;                     // Maximum tokens to generate
    float temperature = 1.0f;                       // Temperature scaling (> 0)
    int top_k = 50;                                 // Top-K parameter (if using TopK)
    float top_p = 0.9f;                             // Top-P parameter (if using TopP)
    int eos_token_id = -1;                          // End-of-sequence token (-1 = disabled)
    bool verbose = false;                           // Print generation progress

    // Optional callback: called after each token is generated
    // Parameters: (token_id, step_number, elapsed_ms)
    std::function<void(int, size_t, double)> token_callback = nullptr;
};

/**
 * @brief Autoregressive text generation
 *
 * This function implements the core loop for LLM text generation.
 * Given a prompt (sequence of tokens), it generates a continuation
 * one token at a time (autoregressively).
 *
 * ============================================================================
 * AUTOREGRESSIVE GENERATION: THE CORE LLM INFERENCE LOOP
 * ============================================================================
 *
 * "Autoregressive" means: generate one token at a time, using previous tokens
 *
 * Process:
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ 1. Start with prompt: [token_1, token_2, ..., token_n]             │
 * └─────────────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ 2. Run model.forward([token_1, ..., token_n])                      │
 * │    → Get logits: [seq_len, vocab_size]                             │
 * └─────────────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ 3. Take logits at last position: logits[-1]                        │
 * │    → This represents: "what token should come next?"               │
 * └─────────────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ 4. Sample next token using chosen method (greedy/top-k/top-p)     │
 * │    → Get token_new                                                  │
 * └─────────────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ 5. Append to sequence: [token_1, ..., token_n, token_new]         │
 * └─────────────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ 6. Repeat steps 2-5 until:                                          │
 * │    - Generate max_new_tokens, OR                                    │
 * │    - Generate EOS (end-of-sequence) token, OR                      │
 * │    - Reach max context length                                       │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * Example:
 *   Prompt: "The capital of France is"
 *   Tokens: [510, 8426, 315, 5444, 374]  ← tokenized prompt
 *
 *   Step 1: forward([510, 8426, 315, 5444, 374]) → sample → 12313 ("Paris")
 *   Step 2: forward([510, ..., 374, 12313]) → sample → 29889 (".")
 *   Step 3: forward([510, ..., 12313, 29889]) → sample → 450 ("The")
 *   ... continue until max_tokens or EOS ...
 *
 *   Result: "The capital of France is Paris. The city..."
 *
 * ============================================================================
 * KV CACHE OPTIMIZATION (NOW IMPLEMENTED!)
 * ============================================================================
 *
 * This implementation uses KV cache for efficient autoregressive generation.
 *
 * How it works:
 *   1. Initialize cache before generation: store K/V for all layers
 *   2. Process prompt: forward pass caches K/V for all prompt tokens
 *   3. Generation loop: only compute K/V for NEW token each step
 *   4. Reuse cached K/V for all previous tokens
 *
 * Benefits:
 *   - Time complexity: O(N) instead of O(N²) for generating N tokens
 *   - Speedup: 10-100x faster for typical sequences!
 *   - Memory cost: ~92 MB for TinyLLaMA (acceptable)
 *
 * Technical details:
 *   - Cache per layer: [max_seq_len, n_kv_heads, head_dim]
 *   - Supports GQA (Grouped Query Attention)
 *   - Cache grows incrementally as tokens are generated
 *   - Reset cache between different prompts
 *
 * ============================================================================
 *
 * @param model The LLM model to use for generation
 * @param prompt_tokens Initial sequence of token IDs
 * @param config Generation configuration (sampling method, parameters, etc.)
 * @return Vector of all tokens (prompt + generated)
 */
inline std::vector<int> generate(
    LLMModel& model,
    const std::vector<int>& prompt_tokens,
    const GenerationConfig& config = GenerationConfig{}
) {
    // ========================================================================
    // Validate inputs
    // ========================================================================

    if (prompt_tokens.empty()) {
        throw std::invalid_argument("generate: prompt_tokens cannot be empty");
    }

    if (config.max_new_tokens == 0) {
        // No generation requested, just return prompt
        return prompt_tokens;
    }

    if (config.temperature <= 0.0f) {
        throw std::invalid_argument("generate: temperature must be positive");
    }

    // ========================================================================
    // Initialize generation state
    // ========================================================================

    // Current sequence: starts with prompt, will grow as we generate
    std::vector<int> tokens = prompt_tokens;

    // Random number generator for sampling (if not greedy)
    std::mt19937 rng(std::random_device{}());

    // ========================================================================
    // Initialize KV Cache
    // ========================================================================
    // Initialize KV cache for efficient generation
    // This allocates memory (~92 MB for TinyLLaMA) and resets all caches
    model.init_kv_cache(model.config().max_seq_len);

    if (config.verbose) {
        std::println("Starting generation:");
        std::println("  Prompt length: {} tokens", prompt_tokens.size());
        std::println("  Max new tokens: {}", config.max_new_tokens);
        std::println("  Method: {}",
                     config.method == SamplingMethod::Greedy ? "Greedy" :
                     config.method == SamplingMethod::TopK ? "Top-K" : "Top-P");
        std::println("  Temperature: {:.2f}", config.temperature);
        std::println("");
    }

    // ========================================================================
    // Process prompt (prefill phase)
    // ========================================================================
    // Run forward pass on the entire prompt to populate KV cache
    // This processes all prompt tokens at once and caches their K/V values
    // After this, the cache contains K/V for all prompt tokens
    //
    // For a prompt of length N:
    //   - Input: [token_0, token_1, ..., token_{N-1}]
    //   - After forward: cache contains K/V for positions 0 to N-1
    //   - Cache length: N
    //
    // This is called "prefill" in LLM inference literature

    if (config.verbose) {
        std::println("Prefill: Processing {} prompt tokens...", prompt_tokens.size());
    }

    auto prefill_start = std::chrono::high_resolution_clock::now();
    Tensor prefill_logits = model.forward(prompt_tokens, 0);
    auto prefill_end = std::chrono::high_resolution_clock::now();

    auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start);
    double prefill_ms = prefill_duration.count();

    if (config.verbose) {
        std::println("  ✓ Prefill complete ({:.2f}s)", prefill_ms / 1000.0);
        std::println("  KV cache length: {}", model.kv_cache_length());
        std::println("");
    }

    // ========================================================================
    // Sample first token from prefill logits
    // ========================================================================
    // The prefill pass gives us logits for all positions in the prompt.
    // We need the logits at the LAST position to predict the next token.
    // After sampling, this becomes our first generated token.

    size_t vocab_size = model.config().vocab_size;
    size_t last_pos = prompt_tokens.size() - 1;

    // Extract logits from the last position of the prompt
    Tensor first_logits({vocab_size});
    for (size_t i = 0; i < vocab_size; ++i) {
        first_logits[i] = prefill_logits.at({last_pos, i});
    }

    // Sample first generated token
    int first_token;
    switch (config.method) {
        case SamplingMethod::Greedy:
            first_token = greedy_sample(first_logits);
            break;
        case SamplingMethod::TopK:
            first_token = top_k_sample(first_logits, config.top_k, config.temperature, rng);
            break;
        case SamplingMethod::TopP:
            first_token = top_p_sample(first_logits, config.top_p, config.temperature, rng);
            break;
        default:
            throw std::runtime_error("Unknown sampling method");
    }

    // Append first generated token
    tokens.push_back(first_token);

    // Call user callback for first token
    if (config.token_callback) {
        config.token_callback(first_token, 1, prefill_ms);
    }

    if (config.verbose) {
        std::println("  Step 1: Generated token ID {} ({:.2f}s)", first_token, prefill_ms / 1000.0);
    }

    // Check stopping conditions after first token
    if (config.eos_token_id >= 0 && first_token == config.eos_token_id) {
        if (config.verbose) {
            std::println("\nStopped: EOS token generated");
        }
        return tokens;
    }

    if (tokens.size() >= model.config().max_seq_len) {
        if (config.verbose) {
            std::println("\nStopped: Reached maximum context length");
        }
        return tokens;
    }

    // ========================================================================
    // Main generation loop: generate remaining tokens one at a time
    // ========================================================================

    for (size_t step = 1; step < config.max_new_tokens; ++step) {
        // --------------------------------------------------------------------
        // Step 1: Run model forward pass on ONLY the last token
        // --------------------------------------------------------------------
        // With KV cache:
        //   - Input: ONLY the last token [last_token]
        //   - The model uses cached K/V for all previous tokens
        //   - Only computes K/V for this new token
        //   - Output: logits for just this position [1, vocab_size]
        //
        // This is called "decode" or "incremental generation" phase
        //
        // Key insight: We already have K/V cached for all previous tokens,
        // so we only need to process the new token!
        //
        // Complexity: O(1) per step (instead of O(seq_len) without cache)

        // For generation step, we only pass the LAST generated token
        // position_offset tells RoPE where this token is in the sequence
        size_t position_offset = tokens.size() - 1;  // Position of the last token

        // Create single-token input: just the last token in the sequence
        std::vector<int> single_token = {tokens.back()};

        auto step_start = std::chrono::high_resolution_clock::now();
        Tensor logits = model.forward(single_token, position_offset);
        auto step_end = std::chrono::high_resolution_clock::now();

        // logits shape: [1, vocab_size] (just one position!)
        // This is the next-token prediction for the current sequence

        // --------------------------------------------------------------------
        // Step 2: Extract logits (already at the "last" position)
        // --------------------------------------------------------------------
        // Since we only passed one token, logits[0] is what we need

        // Extract logits into a 1D tensor (position 0, all vocab)
        Tensor last_logits({vocab_size});
        for (size_t i = 0; i < vocab_size; ++i) {
            last_logits[i] = logits.at({0, i});  // Position 0 (the only position!)
        }

        // --------------------------------------------------------------------
        // Step 3: Sample next token using chosen method
        // --------------------------------------------------------------------
        // Convert logits → probabilities → token ID
        // Different methods give different diversity/quality trade-offs

        int next_token;

        switch (config.method) {
            case SamplingMethod::Greedy:
                // Deterministic: always pick highest probability
                // Fast, but can be repetitive
                next_token = greedy_sample(last_logits);
                break;

            case SamplingMethod::TopK:
                // Sample from K most likely tokens
                // Provides controlled diversity
                next_token = top_k_sample(last_logits, config.top_k, config.temperature, rng);
                break;

            case SamplingMethod::TopP:
                // Nucleus sampling: sample from smallest set with cumulative prob ≥ P
                // Industry standard, adapts to model confidence
                next_token = top_p_sample(last_logits, config.top_p, config.temperature, rng);
                break;

            default:
                throw std::runtime_error("Unknown sampling method");
        }

        // --------------------------------------------------------------------
        // Step 4: Append generated token to sequence
        // --------------------------------------------------------------------
        tokens.push_back(next_token);

        // Calculate elapsed time for this step
        auto step_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);
        double elapsed_ms = step_duration.count();

        // Call user callback if provided
        if (config.token_callback) {
            config.token_callback(next_token, step + 1, elapsed_ms);
        }

        if (config.verbose) {
            std::println("  Step {}: Generated token ID {} ({:.2f}s)", step + 1, next_token, elapsed_ms / 1000.0);
        }

        // --------------------------------------------------------------------
        // Step 5: Check stopping conditions
        // --------------------------------------------------------------------

        // Stop if we hit end-of-sequence token
        if (config.eos_token_id >= 0 && next_token == config.eos_token_id) {
            if (config.verbose) {
                std::println("\nStopped: EOS token generated");
            }
            break;
        }

        // Stop if we reach maximum context length
        if (tokens.size() >= model.config().max_seq_len) {
            if (config.verbose) {
                std::println("\nStopped: Reached maximum context length");
            }
            break;
        }

        // Note: We already stop at max_new_tokens by the loop condition
    }

    // ========================================================================
    // Return complete sequence (prompt + generated tokens)
    // ========================================================================

    if (config.verbose) {
        std::println("\nGeneration complete:");
        std::println("  Total tokens: {}", tokens.size());
        std::println("  Generated: {} new tokens", tokens.size() - prompt_tokens.size());
    }

    return tokens;
}

/**
 * @brief Helper to generate with specific method more easily
 */
inline std::vector<int> generate_greedy(
    LLMModel& model,
    const std::vector<int>& prompt_tokens,
    size_t max_new_tokens = 50
) {
    GenerationConfig config;
    config.method = SamplingMethod::Greedy;
    config.max_new_tokens = max_new_tokens;
    return generate(model, prompt_tokens, config);
}

inline std::vector<int> generate_top_k(
    LLMModel& model,
    const std::vector<int>& prompt_tokens,
    int k = 50,
    float temperature = 1.0f,
    size_t max_new_tokens = 50
) {
    GenerationConfig config;
    config.method = SamplingMethod::TopK;
    config.top_k = k;
    config.temperature = temperature;
    config.max_new_tokens = max_new_tokens;
    return generate(model, prompt_tokens, config);
}

inline std::vector<int> generate_top_p(
    LLMModel& model,
    const std::vector<int>& prompt_tokens,
    float p = 0.9f,
    float temperature = 1.0f,
    size_t max_new_tokens = 50
) {
    GenerationConfig config;
    config.method = SamplingMethod::TopP;
    config.top_p = p;
    config.temperature = temperature;
    config.max_new_tokens = max_new_tokens;
    return generate(model, prompt_tokens, config);
}

} // namespace freellm
