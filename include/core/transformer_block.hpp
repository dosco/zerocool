#pragma once

#include "core/tensor.hpp"
#include "kernels/tensor_ops.hpp"
#include "kernels/quantiz/quant_linear.hpp"
#include "core/layers/attention.hpp"
#include "core/paged_kv_cache.hpp"
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

/**
 * @brief Transformer Decoder Block (used in LLaMA and GPT)
 *
 * A transformer block is the fundamental building unit of modern LLMs.
 * Each block consists of:
 *   1. Multi-head self-attention (captures relationships between tokens)
 *   2. Feed-forward network (processes each position independently)
 *   3. Residual connections (helps with gradient flow during training)
 *   4. Layer normalizations (stabilizes activations)
 *
 * Architecture (Pre-Norm style, used in modern LLMs):
 *
 *   Input x
 *     ↓
 *   RMSNorm ────────┐
 *     ↓             │
 *   Attention       │  ← Self-attention block
 *     ↓             │
 *   (+residual) ←───┘
 *     ↓
 *   RMSNorm ────────┐
 *     ↓             │
 *   FeedForward     │  ← FFN block
 *     ↓             │
 *   (+residual) ←───┘
 *     ↓
 *   Output
 *
 * Why Pre-Norm? (norm before attention/FFN)
 * - Better training stability
 * - Allows training very deep models (100+ layers)
 * - Used in modern models (LLaMA, GPT-3+)
 * - Alternative: Post-Norm (norm after, used in original Transformer, GPT-2)
 *
 * Why Residual Connections?
 * - Allows gradients to flow directly through the network
 * - Prevents vanishing gradients in deep networks
 * - Lets each block learn small "refinements" to the input
 * - Essential for training 20+ layer models
 */
class TransformerBlock {
public:
    /**
     * @brief Construct a transformer block with GQA support
     *
     * @param d_model Model dimension (embedding size)
     * @param d_ff Feed-forward hidden dimension
     * @param n_heads Number of query attention heads
     * @param n_kv_heads Number of key/value heads (for GQA). If 0, defaults to n_heads (MHA)
     * @param norm_eps Epsilon for RMSNorm (numerical stability)
     * @param use_rope Whether to use Rotary Position Embeddings
     * @param rope_theta RoPE base frequency
     * @param max_seq_len Maximum sequence length (for RoPE cache)
     */
    TransformerBlock(
        size_t d_model,
        size_t d_ff,
        size_t n_heads,
        size_t n_kv_heads = 0,  // 0 means use n_heads (MHA mode)
        float norm_eps = 1e-6f,
        bool use_rope = true,
        float rope_theta = 10000.0f,
        size_t max_seq_len = 2048
    )
        : d_model_(d_model)
        , d_ff_(d_ff)
        , n_heads_(n_heads)
        , n_kv_heads_(n_kv_heads == 0 ? n_heads : n_kv_heads)
        , norm_eps_(norm_eps)
    {
        // Initialize attention mechanism with GQA support
        attention_ = std::make_unique<MultiHeadAttention>(
            n_heads, d_model, n_kv_heads_, use_rope, rope_theta, max_seq_len
        );

        // Initialize feed-forward network with SwiGLU (used in TinyLLaMA)
        ffn_ = std::make_unique<FeedForward>(d_model, d_ff, true);  // true = use SwiGLU

        // Initialize RMSNorm weights (one for attention, one for FFN)
        // RMSNorm learns a scale parameter for each dimension
        // Initialize to 1.0 (identity scaling)
        attn_norm_weight_ = Tensor({d_model}, 1.0f);
        ffn_norm_weight_ = Tensor({d_model}, 1.0f);
    }

    /**
     * @brief Forward pass through transformer block
     *
     * Implements the complete pre-norm transformer block with residual connections.
     *
     * Detailed pipeline:
     *   1. norm1 = RMSNorm(x)             - Normalize input
     *   2. attn_out = Attention(norm1)    - Self-attention (with optional KV cache)
     *   3. x = x + attn_out               - Residual connection #1
     *   4. norm2 = RMSNorm(x)             - Normalize again
     *   5. ffn_out = FFN(norm2)           - Feed-forward network
     *   6. x = x + ffn_out                - Residual connection #2
     *   7. return x
     *
     * Shape invariant: output shape = input shape = [seq_len, d_model]
     *
     * @param x Input tensor with shape [seq_len, d_model]
     * @param position_offset Position offset for RoPE (used during generation)
     * @param cache Optional KV cache for this layer. Passed through to attention.
     * @return Output tensor with shape [seq_len, d_model]
     */
    Tensor forward(const Tensor& x, size_t position_offset = 0, PagedKVCache* cache = nullptr) {
        // Validate input shape
        if (x.ndim() != 2 || x.shape()[1] != d_model_) {
            throw std::invalid_argument("TransformerBlock: input must be [seq_len, d_model]");
        }

        // ======================================================================
        // ATTENTION BLOCK (with pre-norm and residual)
        // ======================================================================

        // Step 1: Pre-normalization before attention
        // RMSNorm normalizes the input to have unit RMS (root mean square)
        // This stabilizes the inputs to attention, preventing exploding/vanishing activations
        // Shape: [seq_len, d_model] -> [seq_len, d_model]
        Tensor attn_norm_input = ops::rms_norm(x, attn_norm_weight_, norm_eps_);

        // Step 2: Multi-head self-attention (with optional KV cache)
        // Attention allows each position to gather information from other positions
        // "Self-attention" means Q, K, V all come from the same input
        // When cache is provided, only K/V for new tokens are computed
        // Shape: [seq_len, d_model] -> [seq_len, d_model]
        Tensor attn_output = attention_->forward(attn_norm_input, position_offset, cache);

        // Step 3: Residual connection (add input back to attention output)
        // Why? Allows the model to learn "refinements" rather than full transformations
        // Also helps gradients flow through the network during training
        // Shape: [seq_len, d_model] + [seq_len, d_model] -> [seq_len, d_model]
        Tensor x_after_attn = ops::add(x, attn_output);  // x = x + attention(norm(x))

        // ======================================================================
        // FEED-FORWARD BLOCK (with pre-norm and residual)
        // ======================================================================

        // Step 4: Pre-normalization before FFN
        // Normalize the output from attention block before feeding to FFN
        // Shape: [seq_len, d_model] -> [seq_len, d_model]
        Tensor ffn_norm_input = ops::rms_norm(x_after_attn, ffn_norm_weight_, norm_eps_);

        // Step 5: Feed-forward network
        // FFN processes each position independently (no cross-position interaction)
        // This is where most of the model's parameters are (d_model * d_ff * 2)
        // Shape: [seq_len, d_model] -> [seq_len, d_model]
        Tensor ffn_output = ffn_->forward(ffn_norm_input);

        // Step 6: Residual connection (add input back to FFN output)
        // Second residual connection - again helps with gradient flow
        // Shape: [seq_len, d_model] + [seq_len, d_model] -> [seq_len, d_model]
        Tensor output = ops::add(x_after_attn, ffn_output);  // x = x + ffn(norm(x))

        // Final output has the same shape as input: [seq_len, d_model]
        // The block has "refined" the representations through attention and FFN
        return output;
    }

    // Accessors for weight loading/inspection
    MultiHeadAttention& attention() { return *attention_; }
    FeedForward& ffn() { return *ffn_; }
    Tensor& attn_norm_weight() { return attn_norm_weight_; }
    Tensor& ffn_norm_weight() { return ffn_norm_weight_; }

    const MultiHeadAttention& attention() const { return *attention_; }
    const FeedForward& ffn() const { return *ffn_; }
    const Tensor& attn_norm_weight() const { return attn_norm_weight_; }
    const Tensor& ffn_norm_weight() const { return ffn_norm_weight_; }

private:
    size_t d_model_;     // Model dimension
    [[maybe_unused]] size_t d_ff_;        // Feed-forward dimension
    [[maybe_unused]] size_t n_heads_;     // Number of query attention heads
    size_t n_kv_heads_;  // Number of key/value heads (for GQA)
    float norm_eps_;     // RMSNorm epsilon

    std::unique_ptr<MultiHeadAttention> attention_;  // Multi-head attention module
    std::unique_ptr<FeedForward> ffn_;               // Feed-forward network

    Tensor attn_norm_weight_;  // RMSNorm weight before attention: [d_model]
    Tensor ffn_norm_weight_;   // RMSNorm weight before FFN: [d_model]
};

} // namespace freellm
