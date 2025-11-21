#pragma once

#include "core/tensor.hpp"
#include "kernels/rope.hpp"
#include "kernels/quantiz/quant_linear.hpp"
#include "kernels/attention/scaled_dot_product.hpp"
#include "core/kv_cache.hpp"
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
     * KV CACHE SUPPORT:
     * -----------------
     * When cache != nullptr, the function uses KV cache for efficient generation:
     * - Computes K/V only for NEW tokens (seq_len positions)
     * - Retrieves cached K/V from previous tokens
     * - Concatenates: [cached_K; new_K] and [cached_V; new_V]
     * - Updates cache with new K/V values
     * - Attention uses full sequence length (cached + new)
     *
     * This avoids recomputing K/V for all previous tokens, giving 10-100x speedup!
     *
     * When cache == nullptr, uses original behavior (no caching).
     *
     * @param x Input tensor with shape [seq_len, d_model]
     *          NOTE: No batch dimension - single sequence only
     *          With cache: typically seq_len=1 (one new token)
     *          Without cache: seq_len=full sequence
     * @param position_offset Position offset for RoPE (used in incremental generation)
     *                        When generating token-by-token, this tracks position in sequence
     * @param cache Optional KV cache pointer. If provided, uses cached K/V values.
     * @return Output tensor with shape [seq_len, d_model]
     */
    Tensor forward(const Tensor& x, size_t position_offset = 0, KVCache* cache = nullptr) {
        // Validate input shape: must be 2D [seq_len, d_model]
        // This enforces single-sequence processing (no batch dimension)
        if (x.ndim() != 2) {
            throw std::invalid_argument("MultiHeadAttention: input must be 2D [seq_len, d_model]");
        }

        size_t seq_len = x.shape()[0];  // Sequence length (can vary)

        // Step 1: Linear projections to create Query, Key, Value
        // Each projection is a learned matrix: W_q, W_k, W_v: [d_model, d_model]
        // x @ W_q -> Q: [seq_len, d_model]
        // These projections allow the model to learn what to attend to
        Tensor Q = quant::linear_forward(x, W_q_, W_q_quant_);  // Query: what am I looking for?

        // Step 2: Reshape Q to separate heads
        // Q: [seq_len, d_model] -> [seq_len, n_heads, head_dim]
        Q.reshape({seq_len, n_heads_, head_dim_});

        // Step 3: Apply Rotary Position Embeddings (RoPE) to Q if enabled
        // RoPE encodes positional information directly into Q and K by rotating them
        // This is more efficient than adding positional embeddings
        // position_offset is used during incremental generation (token-by-token)
        // to correctly position new tokens in the sequence
        if (use_rope_ && rope_cache_) {
            Q = rope_cache_->apply(Q, position_offset);
        }

        // Step 4: Compute K and V (with or without cache)
        // ================================================
        Tensor K;
        Tensor V;

        if (cache != nullptr) {
            // KV CACHE PATH: Only compute K/V for NEW tokens
            // -----------------------------------------------
            // During generation, we only need to compute K/V for the new token(s).
            // Previous K/V values are retrieved from cache.
            //
            // Example: Generating 4th token
            //   - Input: x [1, d_model] (just the new token)
            //   - Cache contains: K[0:3], V[0:3] (positions 0, 1, 2)
            //   - Compute: K[3], V[3] (just position 3)
            //   - Concatenate: K_full = [K[0:3]; K[3]] -> [4, n_kv_heads, head_dim]
            //   - Update cache with K[3], V[3]
            //
            // This is the KEY optimization: O(1) K/V computation instead of O(N)!

            // Compute K/V for NEW tokens only
            Tensor K_new = quant::linear_forward(x, W_k_, W_k_quant_);  // [seq_len, n_kv_heads * head_dim]
            Tensor V_new = quant::linear_forward(x, W_v_, W_v_quant_);  // [seq_len, n_kv_heads * head_dim]

            // Reshape to separate heads
            K_new.reshape({seq_len, n_kv_heads_, head_dim_});
            V_new.reshape({seq_len, n_kv_heads_, head_dim_});

            // Apply RoPE to K_new (with correct position offset!)
            if (use_rope_ && rope_cache_) {
                K_new = rope_cache_->apply(K_new, position_offset);
            }

            // Update cache with new K/V values
            cache->update(K_new, V_new);

            // Retrieve FULL K/V (cached + new)
            // After update, cache contains all positions: 0 to (current_length - 1)
            K = cache->get_keys();    // [current_length, n_kv_heads, head_dim]
            V = cache->get_values();  // [current_length, n_kv_heads, head_dim]

            // Note: seq_len_k (used in attention) is now cache->current_length()
            // This is typically much larger than seq_len (which is often 1 for new token)
        } else {
            // NON-CACHED PATH: Compute K/V for entire sequence (original behavior)
            // ---------------------------------------------------------------------
            // This path is used for:
            // 1. Initial prompt processing (no cache yet)
            // 2. Testing/debugging (to verify cache correctness)
            // 3. Single forward passes (not autoregressive generation)

            K = quant::linear_forward(x, W_k_, W_k_quant_);  // Key: what can I match against?
            V = quant::linear_forward(x, W_v_, W_v_quant_);  // Value: what information do I provide?

            // Reshape to separate heads
            // K: [seq_len, n_kv_heads * head_dim] -> [seq_len, n_kv_heads, head_dim]
            // V: [seq_len, n_kv_heads * head_dim] -> [seq_len, n_kv_heads, head_dim]
            //
            // For GQA: K and V have fewer heads than Q
            // Example: d_model=2048, n_heads=32, n_kv_heads=4, head_dim=64
            //   Q: [seq_len, 32, 64]
            //   K: [seq_len, 4, 64]
            //   V: [seq_len, 4, 64]
            K.reshape({seq_len, n_kv_heads_, head_dim_});
            V.reshape({seq_len, n_kv_heads_, head_dim_});

            // Apply RoPE to K
            if (use_rope_ && rope_cache_) {
                K = rope_cache_->apply(K, position_offset);
            }
        }

        // Step 5: Compute scaled dot-product attention for all heads
        // This is the core attention mechanism - see scaled_dot_product_attention()
        // Scale factor prevents attention scores from becoming too large
        // For GQA: pass n_kv_heads so attention function knows to share K/V heads
        //
        // Note: When using cache, K/V have shape [cache_length, n_kv_heads, head_dim]
        //       where cache_length can be much larger than seq_len (Q's length)
        //       The attention function handles different seq_len_q and seq_len_k correctly
        //
        // IMPORTANT: Pass position_offset for correct causal masking with KV cache!
        // This tells the attention function the absolute position of Q[0] in the sequence.
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        Tensor attn_output = scaled_dot_product_attention(Q, K, V, scale, n_kv_heads_, position_offset);
        // Output shape: [seq_len, n_heads, head_dim]
        // (Note: output has n_heads, not n_kv_heads - all Q heads produce output)

        // Step 6: Reshape back to combine all heads
        // Concatenate all head outputs: [seq_len, n_heads, head_dim] -> [seq_len, d_model]
        // This merges the parallel attention heads back into a single representation
        attn_output.reshape({seq_len, d_model_});

        // Step 7: Output projection (learned linear transformation)
        // W_o: [d_model, d_model] projects the combined head outputs
        // This final projection allows the model to learn how to combine information
        // from different attention heads
        Tensor output = quant::linear_forward(attn_output, W_o_, W_o_quant_);
        // Final output: [seq_len, d_model]

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
