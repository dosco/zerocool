#pragma once

#include "core/tensor.hpp"
#include "core/layers/feed_forward.hpp"
#include "kernels/tensor_ops.hpp"
#include "kernels/quantiz/quant_linear.hpp"
#include "infra/thread_pool.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace freellm {

/**
 * @brief Mixture-of-Experts (MoE) Layer
 *
 * Replaces the standard FeedForward layer in MoE models (like Mixtral, Qwen-MoE, OLMoE).
 *
 * Mechanism:
 * 1. Gating/Router: Decides which experts to use for each token.
 *    logits = x @ gate_weight
 *    probs = softmax(top_k(logits))
 * 2. Experts: A collection of FeedForward networks.
 * 3. Aggregation: Weighted sum of expert outputs.
 *    output = sum(expert_output * prob)
 */
class MoELayer {
public:
    /**
     * @brief Construct MoE Layer
     *
     * @param d_model Model dimension
     * @param d_ff Expert hidden dimension
     * @param num_experts Total number of experts
     * @param num_experts_per_token Number of active experts per token (top-k)
     * @param use_swiglu Whether experts use SwiGLU
     */
    MoELayer(size_t d_model, size_t d_ff, size_t num_experts, size_t num_experts_per_token, bool norm_topk_prob, bool use_swiglu = true)
        : d_model_(d_model)
        , d_ff_(d_ff)
        , num_experts_(num_experts)
        , num_experts_per_token_(num_experts_per_token)
        , norm_topk_prob_(norm_topk_prob)
    {
        // Initialize Gating Network
        // Maps input to expert logits: [d_model, num_experts]
        gate_weight_ = Tensor({d_model, num_experts});

        // Initialize Experts
        experts_.reserve(num_experts);
        for (size_t i = 0; i < num_experts; ++i) {
            experts_.emplace_back(d_model, d_ff, use_swiglu);
        }
    }

    /**
     * @brief Forward pass
     *
     * @param x Input tensor [seq_len, d_model]
     * @param debug Enable debug output (default: false)
     * @return Output tensor [seq_len, d_model]
     */
    Tensor forward(const Tensor& x, bool debug = false) {
        size_t seq_len = x.shape()[0];
        debug = debug || debug_;  // Enable debug if either flag is set

        // Debug: Input statistics
        if (debug && seq_len > 0) {
            float input_min = x.data()[0], input_max = x.data()[0], input_sum = 0.0f;
            for (size_t i = 0; i < x.size(); ++i) {
                input_min = std::min(input_min, x.data()[i]);
                input_max = std::max(input_max, x.data()[i]);
                input_sum += x.data()[i];
            }
            std::cout << "[MoE Debug] Input stats: min=" << input_min
                      << " max=" << input_max << " mean=" << (input_sum / x.size())
                      << " shape=[" << seq_len << ", " << d_model_ << "]" << std::endl;
        }

        // 1. Compute Gating Logits
        // input: [seq_len, d_model]
        // gate: [d_model, num_experts]
        // logits: [seq_len, num_experts]
        Tensor logits = quant::linear_forward(x, gate_weight_, gate_weight_quant_);

        // Debug: Gate logits statistics
        if (debug && logits.size() > 0) {
            float logit_min = logits.data()[0], logit_max = logits.data()[0], logit_sum = 0.0f;
            for (size_t i = 0; i < logits.size(); ++i) {
                logit_min = std::min(logit_min, logits.data()[i]);
                logit_max = std::max(logit_max, logits.data()[i]);
                logit_sum += logits.data()[i];
            }
            std::cout << "[MoE Debug] Gate logits: min=" << logit_min
                      << " max=" << logit_max << " mean=" << (logit_sum / logits.size())
                      << " shape=[" << seq_len << ", " << num_experts_ << "]" << std::endl;

            // Check for NaN/Inf
            bool has_nan = false, has_inf = false;
            for (size_t i = 0; i < logits.size(); ++i) {
                if (std::isnan(logits.data()[i])) has_nan = true;
                if (std::isinf(logits.data()[i])) has_inf = true;
            }
            if (has_nan) std::cout << "[MoE Debug] WARNING: Gate logits contain NaN!" << std::endl;
            if (has_inf) std::cout << "[MoE Debug] WARNING: Gate logits contain Inf!" << std::endl;
        }

        // 2. Select Top-K Experts

        // Output tensors
        Tensor output = Tensor({seq_len, d_model_}, 0.0f); // Zero init for accumulation

        // For each token...
        for (size_t t = 0; t < seq_len; ++t) {
            // Get logits for this token
            const float* token_logits = logits.data() + t * num_experts_;

            // Debug: Raw gate logits before softmax (for first token)
            if (debug && t == 0) {
                std::cout << "[MoE Debug] Raw gate logits (token 0, first 10 experts): ";
                for (size_t i = 0; i < std::min(size_t(10), num_experts_); ++i) {
                    std::cout << token_logits[i] << " ";
                }
                std::cout << "\n";
            }

            // Compute Softmax over ALL experts
            std::vector<float> all_probs(num_experts_);
            float max_val = -1e9;
            for(size_t i=0; i<num_experts_; ++i) if(token_logits[i] > max_val) max_val = token_logits[i];

            float sum_exp = 0.0f;
            for(size_t i=0; i<num_experts_; ++i) {
                all_probs[i] = std::exp(token_logits[i] - max_val);
                sum_exp += all_probs[i];
            }
            for(size_t i=0; i<num_experts_; ++i) all_probs[i] /= sum_exp;

            // Find Top-K indices and values from PROBS
            std::vector<std::pair<float, size_t>> scores(num_experts_);
            for (size_t i = 0; i < num_experts_; ++i) {
                scores[i] = {all_probs[i], i};
            }

            // Partial sort to get top-k
            std::partial_sort(scores.begin(), scores.begin() + num_experts_per_token_, scores.end(),
                              std::greater<std::pair<float, size_t>>());

            std::vector<float> probs(num_experts_per_token_);
            std::vector<size_t> indices(num_experts_per_token_);

            // Collect Top-K
            float topk_sum = 0.0f;
            for (size_t k = 0; k < num_experts_per_token_; ++k) {
                probs[k] = scores[k].first;
                indices[k] = scores[k].second;
                topk_sum += probs[k];
            }

            // Normalize Top-K if requested (with divide-by-zero protection)
            if (norm_topk_prob_ && topk_sum > 0.0f) {
                 for (size_t k = 0; k < num_experts_per_token_; ++k) {
                     probs[k] /= topk_sum;
                 }
            }

            // Debug: Show expert selection for first few tokens
            if (debug && t < 3) {
                std::cout << "[MoE Debug] Token " << t << " experts: ";
                for (size_t k = 0; k < num_experts_per_token_; ++k) {
                    std::cout << indices[k] << "(p=" << probs[k] << ") ";
                }
                std::cout << std::endl;
            }

            // 3. Process with selected experts
            Tensor token_input({1, d_model_});
            std::memcpy(token_input.data(), x.data() + t * d_model_, d_model_ * sizeof(float));

            for (size_t k = 0; k < num_experts_per_token_; ++k) {
                size_t expert_idx = indices[k];
                float weight = probs[k];

                // Run expert
                Tensor expert_out = experts_[expert_idx].forward(token_input);

                // Debug: Expert output statistics for first token
                if (debug && t == 0 && k == 0) {
                    float out_min = expert_out.data()[0], out_max = expert_out.data()[0], out_sum = 0.0f;
                    for (size_t i = 0; i < expert_out.size(); ++i) {
                        out_min = std::min(out_min, expert_out.data()[i]);
                        out_max = std::max(out_max, expert_out.data()[i]);
                        out_sum += expert_out.data()[i];
                    }
                    std::cout << "[MoE Debug] Expert " << expert_idx << " output: min=" << out_min
                              << " max=" << out_max << " mean=" << (out_sum / expert_out.size()) << std::endl;
                }

                // Accumulate: output += weight * expert_out
                float* out_row = output.data() + t * d_model_;
                const float* exp_out_data = expert_out.data();

                for (size_t d = 0; d < d_model_; ++d) {
                    out_row[d] += weight * exp_out_data[d];
                }
            }
        }

        // Debug: Output statistics
        if (debug && output.size() > 0) {
            float out_min = output.data()[0], out_max = output.data()[0], out_sum = 0.0f;
            for (size_t i = 0; i < output.size(); ++i) {
                out_min = std::min(out_min, output.data()[i]);
                out_max = std::max(out_max, output.data()[i]);
                out_sum += output.data()[i];
            }
            std::cout << "[MoE Debug] Final output: min=" << out_min
                      << " max=" << out_max << " mean=" << (out_sum / output.size()) << std::endl;
        }

        return output;
    }

    // Accessors
    Tensor& gate_weight() { return gate_weight_; }
    FeedForward& expert(size_t i) { return experts_[i]; }

    // Checkers
    size_t num_experts() const { return num_experts_; }

    // Debug mode
    void set_debug(bool debug) { debug_ = debug; }
    bool debug() const { return debug_; }

    void set_quantized_gate(QuantizedTensor tensor) { gate_weight_quant_ = std::move(tensor); }
    const std::optional<QuantizedTensor>& quantized_gate() const { return gate_weight_quant_; }

private:
    size_t d_model_;
    size_t d_ff_;
    size_t num_experts_;
    size_t num_experts_per_token_;
    bool norm_topk_prob_;
    bool debug_ = false;

    Tensor gate_weight_; // [d_model, num_experts]
    std::optional<QuantizedTensor> gate_weight_quant_;

    std::vector<FeedForward> experts_;
};

} // namespace freellm
