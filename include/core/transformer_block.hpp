#pragma once

#include "core/tensor.hpp"
#include "kernels/tensor_ops.hpp"
#include "kernels/quantiz/quant_linear.hpp"
#include "core/layers/attention.hpp"
#include "core/paged_kv_cache.hpp"
#include "infra/thread_pool.hpp"
#include <memory>
#include <optional>

#include "core/layers/feed_forward.hpp"
#include "core/layers/moe_layer.hpp"

namespace freellm {

// FeedForward class moved to core/layers/feed_forward.hpp

/**
 * @brief Transformer Decoder Block (used in LLaMA and GPT)
 *
 * A transformer block is the fundamental building unit of modern LLMs.
 * Each block consists of:
 *   1. Multi-head self-attention (captures relationships between tokens)
 *   2. Feed-forward network (processes each position independently) OR MoE Layer
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
 *   FFN / MoE       │  ← FFN or MoE block
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
     * @param num_experts Number of experts (0 for dense)
     * @param num_experts_per_token Number of active experts per token
     */
    TransformerBlock(
        size_t d_model,
        size_t d_ff,
        size_t n_heads,
        size_t n_kv_heads = 0,  // 0 means use n_heads (MHA mode)
        float norm_eps = 1e-6f,
        bool use_rope = true,
        float rope_theta = 10000.0f,
        size_t max_seq_len = 2048,
        size_t num_experts = 0,
        size_t num_experts_per_token = 0,
        bool norm_topk_prob = true
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

        if (num_experts > 0) {
            // Initialize MoE Layer
            moe_layer_ = std::make_unique<MoELayer>(
                d_model, d_ff, num_experts, num_experts_per_token, norm_topk_prob, true // use_swiglu=true
            );
        } else {
            // Initialize feed-forward network with SwiGLU (used in TinyLLaMA)
            ffn_ = std::make_unique<FeedForward>(d_model, d_ff, true);  // true = use SwiGLU
        }

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
        // FEED-FORWARD / MoE BLOCK (with pre-norm and residual)
        // ======================================================================

        // Step 4: Pre-normalization before FFN
        // Normalize the output from attention block before feeding to FFN
        // Shape: [seq_len, d_model] -> [seq_len, d_model]
        Tensor ffn_norm_input = ops::rms_norm(x_after_attn, ffn_norm_weight_, norm_eps_);

        // Step 5: Feed-forward network OR MoE
        // Shape: [seq_len, d_model] -> [seq_len, d_model]
        Tensor ffn_output;
        if (moe_layer_) {
            ffn_output = moe_layer_->forward(ffn_norm_input);
        } else {
            ffn_output = ffn_->forward(ffn_norm_input);
        }

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
    FeedForward* ffn() { return ffn_.get(); }
    MoELayer* moe_layer() { return moe_layer_.get(); }
    
    Tensor& attn_norm_weight() { return attn_norm_weight_; }
    Tensor& ffn_norm_weight() { return ffn_norm_weight_; }

    const MultiHeadAttention& attention() const { return *attention_; }
    const FeedForward* ffn() const { return ffn_.get(); }
    const MoELayer* moe_layer() const { return moe_layer_.get(); }
    
    const Tensor& attn_norm_weight() const { return attn_norm_weight_; }
    const Tensor& ffn_norm_weight() const { return ffn_norm_weight_; }

private:
    size_t d_model_;     // Model dimension
    [[maybe_unused]] size_t d_ff_;        // Feed-forward dimension
    [[maybe_unused]] size_t n_heads_;     // Number of query attention heads
    size_t n_kv_heads_;  // Number of key/value heads (for GQA)
    float norm_eps_;     // RMSNorm epsilon

    std::unique_ptr<MultiHeadAttention> attention_;  // Multi-head attention module
    std::unique_ptr<FeedForward> ffn_;               // Feed-forward network (Dense)
    std::unique_ptr<MoELayer> moe_layer_;            // Mixture-of-Experts Layer (Sparse)

    Tensor attn_norm_weight_;  // RMSNorm weight before attention: [d_model]
    Tensor ffn_norm_weight_;   // RMSNorm weight before FFN: [d_model]


};


} // namespace freellm
