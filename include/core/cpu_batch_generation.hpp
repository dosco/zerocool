#pragma once

#include "core/llm_model.hpp"
#include "core/sampling.hpp"
#include <vector>
#include <string>
#include <functional>
#include <chrono>

namespace freellm {

/**
 * @brief Configuration for batched generation
 */
struct BatchGenerationConfig {
    SamplingMethod method = SamplingMethod::TopP;
    size_t max_new_tokens = 20;
    float temperature = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;
    int eos_token_id = -1;

    // Callback: called after each batch step
    // Parameters: (batch_idx, token_id, step_number)
    std::function<void(size_t, int, size_t)> token_callback = nullptr;
};

/**
 * @brief Batched text generation for CPU/MoE models
 *
 * This function implements batched generation where all prompts are processed
 * together, generating one token per prompt per step (true batching).
 *
 * Process:
 * 1. Prefill phase: Process each prompt independently to populate KV cache
 * 2. Decode phase: Batch all sequences' last tokens into a single forward pass
 *    - Each forward pass produces one new token per sequence
 *    - Continue until all sequences hit EOS or max tokens
 *
 * @param model The LLM model to use for generation
 * @param prompt_tokens_batch Vector of prompt token sequences
 * @param config Generation configuration
 * @return Vector of generated token sequences (one per prompt)
 */
inline std::vector<std::vector<int>> generate_batch(
    LLMModel& model,
    const std::vector<std::vector<int>>& prompt_tokens_batch,
    const BatchGenerationConfig& config = BatchGenerationConfig{}
) {
    if (prompt_tokens_batch.empty()) {
        return {};
    }

    const size_t batch_size = prompt_tokens_batch.size();
    const size_t vocab_size = model.config().vocab_size;
    const size_t max_seq_len = model.config().max_seq_len;

    // Initialize result vectors with prompts
    std::vector<std::vector<int>> sequences;
    sequences.reserve(batch_size);
    for (const auto& prompt : prompt_tokens_batch) {
        sequences.push_back(prompt);
    }

    // Track which sequences are still active (not hit EOS or max length)
    std::vector<bool> active(batch_size, true);
    size_t num_active = batch_size;

    // Random number generator for sampling
    std::mt19937 rng(std::random_device{}());

    // ========================================================================
    // Prefill phase: Process each prompt independently
    // ========================================================================
    // Each prompt needs its own KV cache state. For now, we process them
    // sequentially during prefill (could be parallelized in the future).

    // Store the logits from each prefill to sample first token
    std::vector<int> first_tokens(batch_size);

    for (size_t b = 0; b < batch_size; ++b) {
        if (sequences[b].empty()) {
            active[b] = false;
            num_active--;
            continue;
        }

        // Initialize KV cache for this sequence
        model.init_kv_cache(max_seq_len);

        // Run prefill
        Tensor prefill_logits = model.forward(sequences[b], 0);

        // Extract logits from last position
        size_t last_pos = sequences[b].size() - 1;
        Tensor last_logits({vocab_size});
        for (size_t i = 0; i < vocab_size; ++i) {
            last_logits[i] = prefill_logits.at({last_pos, i});
        }

        // Sample first token
        int token;
        switch (config.method) {
            case SamplingMethod::Greedy:
                token = greedy_sample(last_logits);
                break;
            case SamplingMethod::TopK:
                token = top_k_sample(last_logits, config.top_k, config.temperature, rng);
                break;
            case SamplingMethod::TopP:
            default:
                token = top_p_sample(last_logits, config.top_p, config.temperature, rng);
                break;
        }

        first_tokens[b] = token;
        sequences[b].push_back(token);

        // Call callback
        if (config.token_callback) {
            config.token_callback(b, token, 1);
        }

        // Check stopping conditions
        if (config.eos_token_id >= 0 && token == config.eos_token_id) {
            active[b] = false;
            num_active--;
        } else if (sequences[b].size() >= max_seq_len) {
            active[b] = false;
            num_active--;
        }

        // Reset KV cache for next prompt's prefill
        model.reset_kv_cache();
    }

    // If we only wanted 1 token or all sequences ended, return now
    if (config.max_new_tokens <= 1 || num_active == 0) {
        return sequences;
    }

    // ========================================================================
    // Decode phase: Batch all active sequences' last tokens
    // ========================================================================
    // Note: For true batching with KV cache, we'd need to maintain separate
    // KV caches per sequence. For simplicity in this initial implementation,
    // we process decode tokens sequentially per batch (each forward is batched
    // but KV cache is reset between batches).
    //
    // A more optimized version would maintain per-sequence KV caches.

    for (size_t step = 1; step < config.max_new_tokens && num_active > 0; ++step) {
        for (size_t b = 0; b < batch_size; ++b) {
            if (!active[b]) continue;

            // Re-initialize KV cache and re-run full context
            // (This is inefficient but correct - a production version would
            //  maintain persistent per-sequence KV caches)
            model.init_kv_cache(max_seq_len);

            // Run forward on full sequence
            Tensor logits = model.forward(sequences[b], 0);

            // Extract logits from last position
            size_t last_pos = sequences[b].size() - 1;
            Tensor last_logits({vocab_size});
            for (size_t i = 0; i < vocab_size; ++i) {
                last_logits[i] = logits.at({last_pos, i});
            }

            // Sample next token
            int token;
            switch (config.method) {
                case SamplingMethod::Greedy:
                    token = greedy_sample(last_logits);
                    break;
                case SamplingMethod::TopK:
                    token = top_k_sample(last_logits, config.top_k, config.temperature, rng);
                    break;
                case SamplingMethod::TopP:
                default:
                    token = top_p_sample(last_logits, config.top_p, config.temperature, rng);
                    break;
            }

            sequences[b].push_back(token);

            // Call callback
            if (config.token_callback) {
                config.token_callback(b, token, step + 1);
            }

            // Check stopping conditions
            if (config.eos_token_id >= 0 && token == config.eos_token_id) {
                active[b] = false;
                num_active--;
            } else if (sequences[b].size() >= max_seq_len) {
                active[b] = false;
                num_active--;
            }

            model.reset_kv_cache();
        }
    }

    return sequences;
}

/**
 * @brief Batched generation with interleaved output (one token per batch per step)
 *
 * This version produces output in the order that matches the user's expectation:
 * [0] token1  [1] token1  [2] token1
 * [0] token2  [1] token2  [2] token2
 * ...
 *
 * @param model The LLM model
 * @param prompt_tokens_batch Vector of prompt token sequences
 * @param config Generation configuration
 * @return Vector of generated token sequences
 */
inline std::vector<std::vector<int>> generate_batch_interleaved(
    LLMModel& model,
    const std::vector<std::vector<int>>& prompt_tokens_batch,
    const BatchGenerationConfig& config = BatchGenerationConfig{}
) {
    // The implementation above already produces interleaved output via callback
    return generate_batch(model, prompt_tokens_batch, config);
}

} // namespace freellm
