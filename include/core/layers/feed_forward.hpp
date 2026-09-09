#pragma once

#include "core/tensor.hpp"
#include "kernels/tensor_ops.hpp"
#include "kernels/quantiz/quant_linear.hpp"
#include "infra/thread_pool.hpp"
#include <memory>
#include <optional>

namespace freellm {

/**
 * @brief Feed-Forward Network (FFN) used in transformer blocks
 *
 * TinyLLaMA uses SwiGLU activation (a gated variant of SiLU):
 *
 *   SwiGLU(x) = (SiLU(W_gate @ x) ⊙ W_up @ x) @ W_down
 *
 * Where ⊙ is element-wise multiplication (gating).
 *
 * Architecture:
 *   Input [d_model] →  W_gate [d_ff] → SiLU  ──┐
 *                  └→  W_up [d_ff]  ──────────→ ⊙ → W_down [d_model] → Output
 *
 * Why SwiGLU?
 * - Gating mechanism allows model to control information flow
 * - Better performance than plain FFN in LLaMA models
 * - Two parallel transformations with gating
 * - Used in LLaMA, PaLM, and derivatives
 *
 * Purpose:
 * - After attention captures relationships between tokens, FFN processes each position
 * - FFN is applied identically at each position (position-wise)
 * - The expansion (d_model → d_ff → d_model) allows the model to learn complex transformations
 */
class FeedForward {
public:
    /**
     * @brief Construct feed-forward network with SwiGLU
     *
     * @param d_model Model dimension (input and output size)
     * @param d_ff Hidden dimension (expansion size)
     * @param use_swiglu Whether to use SwiGLU (true) or plain SiLU (false)
     */
    FeedForward(size_t d_model, size_t d_ff, bool use_swiglu = true)
        : d_model_(d_model), d_ff_(d_ff), use_swiglu_(use_swiglu) {

        // Initialize weight matrices
        if (use_swiglu) {
            // SwiGLU requires three weight matrices
            W_gate_ = Tensor({d_model, d_ff});  // Gate projection
            W_up_ = Tensor({d_model, d_ff});    // Up projection
            W_down_ = Tensor({d_ff, d_model});  // Down projection
        } else {
            // Plain FFN uses two weight matrices
            W_up_ = Tensor({d_model, d_ff});
            W_down_ = Tensor({d_ff, d_model});
        }
    }

    /**
     * @brief Forward pass through FFN
     *
     * SwiGLU Pipeline:
     *   1. gate = x @ W_gate:  [seq_len, d_model] @ [d_model, d_ff] -> [seq_len, d_ff]
     *   2. up = x @ W_up:      [seq_len, d_model] @ [d_model, d_ff] -> [seq_len, d_ff]
     *   3. gated = SiLU(gate) ⊙ up  (element-wise multiplication)
     *   4. output = gated @ W_down: [seq_len, d_ff] @ [d_ff, d_model] -> [seq_len, d_model]
     *
     * Shape invariant: output shape = input shape = [seq_len, d_model]
     *
     * @param x Input tensor with shape [seq_len, d_model]
     * @return Output tensor with shape [seq_len, d_model]
     */
    Tensor forward(const Tensor& x) {
        // Validate input dimensions
        if (x.ndim() != 2 || x.shape()[1] != d_model_) {
            throw std::invalid_argument("FFN: input must be [seq_len, d_model]");
        }

        if (use_swiglu_) {
            // SwiGLU: Gated FFN with SiLU activation

            // Launch Gate and Up projections in parallel
            auto future_gate = ThreadPool::instance().enqueue([&] {
                Tensor gate = quant::linear_forward(x, W_gate_, W_gate_quant_);
                return ops::silu(gate);
            });

            auto future_up = ThreadPool::instance().enqueue([&] {
                return quant::linear_forward(x, W_up_, W_up_quant_);
            });

            // Wait for results
            Tensor gate_activated = future_gate.get();
            Tensor up = future_up.get();

            // Step 3: Gating (element-wise multiplication)
            // SiLU(gate) ⊙ up -> [seq_len, d_ff]
            Tensor gated = ops::multiply(gate_activated, up);

            // Step 4: Down projection
            // gated @ W_down -> [seq_len, d_model]
            Tensor output = quant::linear_forward(gated, W_down_, W_down_quant_);

            return output;

        } else {
            // Plain FFN with SiLU (fallback)

            // Step 1: Up projection
            Tensor hidden = quant::linear_forward(x, W_up_, W_up_quant_);

            // Step 2: SiLU activation
            Tensor activated = ops::silu(hidden);

            // Step 3: Down projection
            Tensor output = quant::linear_forward(activated, W_down_, W_down_quant_);

            return output;
        }
    }

    // Accessors for weight loading
    Tensor& W_gate() { return W_gate_; }
    Tensor& W_up() { return W_up_; }
    Tensor& W_down() { return W_down_; }
    const Tensor& W_gate() const { return W_gate_; }
    const Tensor& W_up() const { return W_up_; }
    const Tensor& W_down() const { return W_down_; }

    void set_quantized_W_gate(QuantizedTensor tensor) { W_gate_quant_ = std::move(tensor); }
    void set_quantized_W_up(QuantizedTensor tensor) { W_up_quant_ = std::move(tensor); }
    void set_quantized_W_down(QuantizedTensor tensor) { W_down_quant_ = std::move(tensor); }

    const std::optional<QuantizedTensor>& quantized_W_gate() const { return W_gate_quant_; }
    const std::optional<QuantizedTensor>& quantized_W_up() const { return W_up_quant_; }
    const std::optional<QuantizedTensor>& quantized_W_down() const { return W_down_quant_; }

private:
    size_t d_model_;   // Model dimension (input/output size)
    [[maybe_unused]] size_t d_ff_;      // Feed-forward hidden dimension (expansion size)
    bool use_swiglu_;  // Whether to use SwiGLU or plain SiLU

    Tensor W_gate_;  // Gate projection (SwiGLU only): [d_model, d_ff]
    Tensor W_up_;    // Up projection: [d_model, d_ff]
    Tensor W_down_;  // Down projection: [d_ff, d_model]

    std::optional<QuantizedTensor> W_gate_quant_;
    std::optional<QuantizedTensor> W_up_quant_;
    std::optional<QuantizedTensor> W_down_quant_;
};

} // namespace freellm
