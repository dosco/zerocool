#pragma once

#include "core/tensor.hpp"
#include "kernels/rope.hpp"
#include "kernels/quantiz/quant_linear.hpp"
#include "kernels/attention/scaled_dot_product.hpp"
#include "kernels/attention/paged_attention.hpp"
#include "core/paged_kv_cache.hpp"
#include <memory>
#include <optional>
#include <cmath>

namespace freellm {

/**
 * @brief Multi-Head Attention layer
 *
 * Multi-head attention allows the model to jointly attend to information
 * from different representation subspaces at different positions.
 *
 * The layer applies linear transformations to create Q, K, V, performs
 * attention in parallel across multiple heads, and combines the results.
 *
 * DESIGN NOTE: This implementation processes a SINGLE sequence at a time.
 * There is NO batch dimension - shapes are [seq_len, d_model] instead of
 * [batch_size, seq_len, d_model]. This is intentional for inference:
 *
 * - During autoregressive generation, you generate one token at a time
 * - Processing sequences one-by-one avoids padding and masking complexity
 * - More memory-efficient for CPU inference
 * - Simpler code without batch coordination
 *
 * If you need batch processing (e.g., for training or parallel inference),
 * you would need to:
 * 1. Add batch dimension: [batch_size, seq_len, d_model]
 * 2. Add attention masks for padding
 * 3. Handle variable-length sequences
 * 4. Update all loops to iterate over batches
 */
class MultiHeadAttention {
public:
    /**
     * @brief Construct multi-head attention layer with GQA support
     *
     * @param n_heads Number of query attention heads
     * @param d_model Total model dimension
     * @param n_kv_heads Number of key/value heads (for GQA). If 0, defaults to n_heads (MHA)
     * @param use_rope Whether to use Rotary Position Embeddings
     * @param rope_theta RoPE base frequency (default: 10000.0 for LLaMA)
     * @param max_seq_len Maximum sequence length (for RoPE cache)
     */
    MultiHeadAttention(
        size_t n_heads,
        size_t d_model,
        size_t n_kv_heads = 0,  // 0 means use n_heads (MHA mode)
        bool use_rope = false,
        float rope_theta = 10000.0f,
        size_t max_seq_len = 2048
    )
        : n_heads_(n_heads)
        , n_kv_heads_(n_kv_heads == 0 ? n_heads : n_kv_heads)  // Default to MHA
        , d_model_(d_model)
        , head_dim_(d_model / n_heads)
        , use_rope_(use_rope)
    {
        if (d_model % n_heads != 0) {
            throw std::invalid_argument("d_model must be divisible by n_heads");
        }
        if (n_heads % n_kv_heads_ != 0) {
            throw std::invalid_argument("n_heads must be divisible by n_kv_heads (for GQA)");
        }

        // Initialize weight matrices (will be loaded from file in practice)
        // For GQA: W_k and W_v have fewer heads than W_q
        W_q_ = Tensor({d_model, n_heads * head_dim_});       // [d_model, d_model]
        W_k_ = Tensor({d_model, n_kv_heads_ * head_dim_});   // [d_model, n_kv_heads * head_dim]
        W_v_ = Tensor({d_model, n_kv_heads_ * head_dim_});   // [d_model, n_kv_heads * head_dim]
        W_o_ = Tensor({d_model, d_model});

        if (use_rope) {
            rope_cache_ = std::make_unique<RoPECache>(head_dim_, max_seq_len, rope_theta);
        }
    }

    /**
     * @brief Forward pass through multi-head attention
     *
     * Forward pass pipeline:
     * 1. Linear projections: x -> Q, K, V (learned transformations)
     * 2. Reshape: split d_model into n_heads × head_dim
     * 3. Apply RoPE (if enabled): add positional information to Q, K
     * 4. Scaled dot-product attention: compute attention for each head
     * 5. Reshape: concatenate heads back to d_model
     * 6. Output projection: final learned transformation
     *
     * PAGED KV CACHE SUPPORT:
     * -----------------------
     * When cache != nullptr, the function uses Paged KV cache for efficient generation:
     * - Computes K/V only for NEW tokens (seq_len positions)
     * - Updates paged cache with new K/V values (allocating blocks if needed)
     * - Uses paged_scaled_dot_product_attention to read from non-contiguous memory
     *
     * @param x Input tensor with shape [seq_len, d_model]
     * @param position_offset Position offset for RoPE
     * @param cache Optional Paged KV cache pointer.
     * @return Output tensor with shape [seq_len, d_model]
     */
    Tensor forward(const Tensor& x, size_t position_offset = 0, PagedKVCache* cache = nullptr) {
        // Validate input shape: must be 2D [seq_len, d_model]
        if (x.ndim() != 2) {
            throw std::invalid_argument("MultiHeadAttention: input must be 2D [seq_len, d_model]");
        }

        size_t seq_len = x.shape()[0];

        // Step 1: Linear projections to create Query, Key, Value
        Tensor Q = quant::linear_forward(x, W_q_, W_q_quant_);

        // Step 2: Reshape Q to separate heads
        Q.reshape({seq_len, n_heads_, head_dim_});

        // Step 3: Apply RoPE to Q
        if (use_rope_ && rope_cache_) {
            Q = rope_cache_->apply(Q, position_offset);
        }

        Tensor output;

        if (cache != nullptr) {
            // PAGED KV CACHE PATH
            // -------------------
            
            // Compute K/V for NEW tokens only
            Tensor K_new = quant::linear_forward(x, W_k_, W_k_quant_);
            Tensor V_new = quant::linear_forward(x, W_v_, W_v_quant_);

            // Reshape to separate heads
            K_new.reshape({seq_len, n_kv_heads_, head_dim_});
            V_new.reshape({seq_len, n_kv_heads_, head_dim_});

            // Apply RoPE to K_new
            if (use_rope_ && rope_cache_) {
                K_new = rope_cache_->apply(K_new, position_offset);
            }

            // Update paged cache with new K/V values
            cache->update(K_new, V_new);

            // Compute attention using paged kernel
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
            Tensor attn_output = paged_scaled_dot_product_attention(Q, *cache, scale, n_kv_heads_, position_offset);
            
            // Reshape back to combine all heads
            attn_output.reshape({seq_len, d_model_});
            
            // Output projection
            output = quant::linear_forward(attn_output, W_o_, W_o_quant_);

        } else {
            // STANDARD PATH (No Cache)
            // ------------------------
            // Used for initial prompt processing if not using cache immediately, or testing.
            // Note: In a real paged system, we might want to always use the cache even for prompt.
            // But for now, let's keep the non-cached path for flexibility/testing.
            
            Tensor K = quant::linear_forward(x, W_k_, W_k_quant_);
            Tensor V = quant::linear_forward(x, W_v_, W_v_quant_);

            K.reshape({seq_len, n_kv_heads_, head_dim_});
            V.reshape({seq_len, n_kv_heads_, head_dim_});

            if (use_rope_ && rope_cache_) {
                K = rope_cache_->apply(K, position_offset);
            }

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
            Tensor attn_output = scaled_dot_product_attention(Q, K, V, scale, n_kv_heads_, position_offset);

            attn_output.reshape({seq_len, d_model_});
            output = quant::linear_forward(attn_output, W_o_, W_o_quant_);
        }

        return output;
    }

    // Accessors for weight matrices (for loading from file)
    Tensor& W_q() { return W_q_; }
    Tensor& W_k() { return W_k_; }
    Tensor& W_v() { return W_v_; }
    Tensor& W_o() { return W_o_; }

    const Tensor& W_q() const { return W_q_; }
    const Tensor& W_k() const { return W_k_; }
    const Tensor& W_v() const { return W_v_; }
    const Tensor& W_o() const { return W_o_; }

    void set_quantized_W_q(QuantizedTensor tensor) { W_q_quant_ = std::move(tensor); }
    void set_quantized_W_k(QuantizedTensor tensor) { W_k_quant_ = std::move(tensor); }
    void set_quantized_W_v(QuantizedTensor tensor) { W_v_quant_ = std::move(tensor); }
    void set_quantized_W_o(QuantizedTensor tensor) { W_o_quant_ = std::move(tensor); }

    const std::optional<QuantizedTensor>& quantized_W_q() const { return W_q_quant_; }
    const std::optional<QuantizedTensor>& quantized_W_k() const { return W_k_quant_; }
    const std::optional<QuantizedTensor>& quantized_W_v() const { return W_v_quant_; }
    const std::optional<QuantizedTensor>& quantized_W_o() const { return W_o_quant_; }

    // Getters
    size_t n_heads() const { return n_heads_; }
    size_t n_kv_heads() const { return n_kv_heads_; }
    size_t d_model() const { return d_model_; }
    size_t head_dim() const { return head_dim_; }
    bool uses_rope() const { return use_rope_; }

private:
    size_t n_heads_;       // Number of query heads
    size_t n_kv_heads_;    // Number of key/value heads (for GQA)
    size_t d_model_;
    size_t head_dim_;
    bool use_rope_;

    Tensor W_q_;  // Query projection
    Tensor W_k_;  // Key projection
    Tensor W_v_;  // Value projection
    Tensor W_o_;  // Output projection

    std::optional<QuantizedTensor> W_q_quant_;
    std::optional<QuantizedTensor> W_k_quant_;
    std::optional<QuantizedTensor> W_v_quant_;
    std::optional<QuantizedTensor> W_o_quant_;

    std::unique_ptr<RoPECache> rope_cache_;
};

} // namespace freellm
