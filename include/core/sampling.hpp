#pragma once

#include "core/tensor.hpp"
#include "kernels/tensor_ops.hpp"
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

namespace freellm {

/**
 * @brief Sampling strategies for next token prediction
 *
 * After running the model forward pass, we get logits: [vocab_size]
 * These are raw scores for each possible next token.
 *
 * Sampling methods convert logits → next token ID
 *
 * Different strategies trade off:
 * - **Quality**: How good/coherent is the generated text?
 * - **Diversity**: How varied/creative is the output?
 * - **Determinism**: Same input → same output?
 *
 * Common approaches:
 * 1. **Greedy**: Always pick highest probability (deterministic, can be repetitive)
 * 2. **Top-K**: Sample from K highest probabilities (more diverse)
 * 3. **Top-P (Nucleus)**: Sample from smallest set with cumulative prob ≥ P (best quality)
 * 4. **Temperature**: Scale logits to control randomness (affects all methods)
 */

/**
 * @brief Apply temperature scaling to logits
 *
 * Temperature controls the "sharpness" of the probability distribution:
 *
 * - **T = 1.0**: No change (standard distribution)
 * - **T < 1.0** (e.g., 0.7): Sharper distribution (more confident, less random)
 *   - Probability mass concentrates on high-probability tokens
 *   - More deterministic, more focused output
 *   - Good for factual/precise generation
 *
 * - **T > 1.0** (e.g., 1.2): Softer distribution (less confident, more random)
 *   - Probability mass spreads to lower-probability tokens
 *   - More diverse, more creative output
 *   - Can become incoherent if too high
 *
 * Formula: scaled_logits[i] = logits[i] / temperature
 *
 * Example:
 *   Original logits: [2.0, 1.0, 0.5]
 *   After softmax:   [0.62, 0.23, 0.15]  (T=1.0)
 *
 *   T=0.5 (sharper): [0.76, 0.18, 0.06]  ← Higher contrast
 *   T=2.0 (softer):  [0.47, 0.29, 0.24]  ← More uniform
 *
 * @param logits Input logits (unnormalized scores)
 * @param temperature Temperature value (> 0)
 * @return Temperature-scaled logits
 */
inline Tensor apply_temperature(const Tensor& logits, float temperature) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }

    // Create copy and scale by 1/temperature
    Tensor scaled = logits.copy();
    for (size_t i = 0; i < scaled.size(); ++i) {
        scaled[i] /= temperature;
    }

    return scaled;
}

/**
 * @brief Find index of maximum value (argmax)
 *
 * Returns the position of the largest value in the tensor.
 *
 * @param values Input tensor (1D)
 * @return Index of maximum value
 */
inline size_t argmax(const Tensor& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot find argmax of empty tensor");
    }

    const float* data = values.data();
    size_t max_idx = 0;
    float max_val = data[0];

    for (size_t i = 1; i < values.size(); ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }

    return max_idx;
}

/**
 * @brief Greedy sampling - always pick the highest probability token
 *
 * Strategy: argmax(softmax(logits))
 *
 * Pros:
 * - Deterministic (same input → same output)
 * - Fast (no random sampling needed)
 * - Often produces coherent text
 *
 * Cons:
 * - Can be repetitive ("the the the...")
 * - No diversity (always same output)
 * - Gets stuck in loops
 *
 * Use when:
 * - You want deterministic output
 * - Generation quality is good enough
 * - Speed is critical
 *
 * @param logits Unnormalized scores for each token [vocab_size]
 * @return Token ID with highest probability
 */
inline int greedy_sample(const Tensor& logits) {
    // Convert logits to probabilities (not strictly needed for argmax, but clearer)
    // We could just argmax(logits) directly since argmax is invariant to monotonic transforms
    return static_cast<int>(argmax(logits));
}

/**
 * @brief Top-K sampling - sample from K most likely tokens
 *
 * Strategy:
 * 1. Find K tokens with highest logits
 * 2. Zero out all other tokens
 * 3. Apply softmax to get probabilities
 * 4. Sample from this filtered distribution
 *
 * Pros:
 * - More diverse than greedy
 * - Controls maximum diversity (K parameter)
 * - Prevents very unlikely tokens
 *
 * Cons:
 * - Fixed K can be too restrictive or too permissive
 * - Doesn't adapt to confidence level
 *
 * Typical values: K=40-100
 *
 * Use when:
 * - You want controlled diversity
 * - Generation is too repetitive
 *
 * @param logits Unnormalized scores [vocab_size]
 * @param k Number of top tokens to consider
 * @param temperature Temperature for scaling (default: 1.0)
 * @param rng Random number generator
 * @return Sampled token ID
 */
inline int top_k_sample(
    const Tensor& logits,
    int k,
    float temperature = 1.0f,
    std::mt19937& rng = []() -> std::mt19937& {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }()
) {
    if (k <= 0) {
        throw std::invalid_argument("K must be positive");
    }

    size_t vocab_size = logits.size();
    k = std::min(k, static_cast<int>(vocab_size));

    // Apply temperature
    Tensor scaled_logits = apply_temperature(logits, temperature);

    // Create vector of (logit, index) pairs for sorting
    std::vector<std::pair<float, int>> logit_idx_pairs;
    logit_idx_pairs.reserve(vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
        logit_idx_pairs.emplace_back(scaled_logits[i], static_cast<int>(i));
    }

    // Partial sort to get top K (heap-based, O(n + k log k))
    std::partial_sort(
        logit_idx_pairs.begin(),
        logit_idx_pairs.begin() + k,
        logit_idx_pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Compute softmax probabilities for top K
    // First, find max for numerical stability
    float max_logit = logit_idx_pairs[0].first;

    // Compute exp(logit - max) and sum
    std::vector<float> probs(k);
    float sum_exp = 0.0f;
    for (int i = 0; i < k; ++i) {
        float exp_val = std::exp(logit_idx_pairs[i].first - max_logit);
        probs[i] = exp_val;
        sum_exp += exp_val;
    }

    // Normalize to get probabilities
    for (int i = 0; i < k; ++i) {
        probs[i] /= sum_exp;
    }

    // Sample from categorical distribution
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int selected_idx = dist(rng);

    return logit_idx_pairs[selected_idx].second;
}

/**
 * @brief Top-P (Nucleus) sampling - sample from smallest set with cumulative prob ≥ P
 *
 * Strategy:
 * 1. Sort tokens by probability (descending)
 * 2. Find smallest set where cumulative probability ≥ P
 * 3. Sample from this "nucleus" of tokens
 *
 * Pros:
 * - Adapts to model confidence (high confidence → fewer tokens, low confidence → more tokens)
 * - Better quality than top-K (more principled)
 * - Industry standard (used in GPT-3, LLaMA)
 *
 * Cons:
 * - Slightly more complex than top-K
 * - Still need to tune P parameter
 *
 * Typical values: P=0.9-0.95
 *
 * Example:
 *   Sorted probs: [0.4, 0.3, 0.15, 0.1, 0.05]
 *   P=0.9: Keep [0.4, 0.3, 0.15, 0.1] (sum=0.95 ≥ 0.9)
 *   Resample from just these 4 tokens
 *
 * Intuition: "Consider tokens that together make up 90% of probability mass"
 *
 * Use when:
 * - You want high-quality generation
 * - Model confidence varies across predictions
 * - Used in production LLMs
 *
 * @param logits Unnormalized scores [vocab_size]
 * @param p Cumulative probability threshold (0 < p <= 1)
 * @param temperature Temperature for scaling (default: 1.0)
 * @param rng Random number generator
 * @return Sampled token ID
 */
inline int top_p_sample(
    const Tensor& logits,
    float p,
    float temperature = 1.0f,
    std::mt19937& rng = []() -> std::mt19937& {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }()
) {
    if (p <= 0.0f || p > 1.0f) {
        throw std::invalid_argument("P must be in range (0, 1]");
    }

    size_t vocab_size = logits.size();

    // Apply temperature
    Tensor scaled_logits = apply_temperature(logits, temperature);

    // Compute softmax probabilities
    // First, find max for numerical stability
    float max_logit = scaled_logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, scaled_logits[i]);
    }

    // Compute exp and sum
    std::vector<float> exp_vals(vocab_size);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        exp_vals[i] = std::exp(scaled_logits[i] - max_logit);
        sum_exp += exp_vals[i];
    }

    // Normalize to probabilities and create (prob, index) pairs
    std::vector<std::pair<float, int>> prob_idx_pairs;
    prob_idx_pairs.reserve(vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
        float prob = exp_vals[i] / sum_exp;
        prob_idx_pairs.emplace_back(prob, static_cast<int>(i));
    }

    // Sort by probability (descending)
    std::sort(prob_idx_pairs.begin(), prob_idx_pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Find nucleus: smallest set with cumulative prob ≥ p
    float cumulative_prob = 0.0f;
    size_t nucleus_size = 0;

    for (size_t i = 0; i < vocab_size; ++i) {
        cumulative_prob += prob_idx_pairs[i].first;
        nucleus_size++;

        if (cumulative_prob >= p) {
            break;
        }
    }

    // Extract probabilities for nucleus (already normalized)
    std::vector<float> nucleus_probs;
    nucleus_probs.reserve(nucleus_size);

    float nucleus_sum = 0.0f;
    for (size_t i = 0; i < nucleus_size; ++i) {
        nucleus_probs.push_back(prob_idx_pairs[i].first);
        nucleus_sum += prob_idx_pairs[i].first;
    }

    // Renormalize nucleus probabilities (they should sum to ~p, we want sum=1)
    for (float& prob : nucleus_probs) {
        prob /= nucleus_sum;
    }

    // Sample from nucleus
    std::discrete_distribution<int> dist(nucleus_probs.begin(), nucleus_probs.end());
    int selected_idx = dist(rng);

    return prob_idx_pairs[selected_idx].second;
}

} // namespace freellm
