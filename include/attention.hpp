#pragma once

#include "tensor.hpp"
#include "tensor_ops.hpp"
#include "rope.hpp"
#include "quantiz/quant_linear.hpp"
#include "kv_cache.hpp"
#include "thread_pool.hpp"
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>
#include <optional>

// AVX2 intrinsics for vectorized causal masking
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace freellm {

/**
 * @brief Scaled dot-product attention with support for Grouped Query Attention (GQA)
 *
 * attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * This is the core operation in transformer models.
 *
 * Supports both Multi-Head Attention (MHA) and Grouped Query Attention (GQA):
 * - MHA: n_heads == n_kv_heads (default, all heads independent)
 * - GQA: n_heads > n_kv_heads (K/V heads shared across multiple Q heads)
 *
 * Example GQA: 32 Q heads with 4 K/V heads means each K/V head is shared
 * by 8 Q heads. This reduces memory for KV cache by ~3x.
 *
 * NOTE: This implementation does NOT include a batch dimension. It is designed
 * for single-sequence inference (autoregressive generation), which is common
 * for LLM inference engines. This design choice:
 * - Simplifies the code (no padding/masking needed)
 * - Is memory-efficient for CPU inference
 * - Matches the typical use case: generating one token at a time for one prompt
 *
 * Shape convention:
 *   Q: [seq_len_q, n_heads, head_dim]      - Query heads
 *   K: [seq_len_k, n_kv_heads, head_dim]   - Key heads (can be < n_heads)
 *   V: [seq_len_k, n_kv_heads, head_dim]   - Value heads (can be < n_heads)
 *
 * Compare to batched implementations (e.g., PyTorch):
 *   - Batched: [batch_size, seq_len, n_heads, head_dim]
 *   - This:    [seq_len, n_heads, head_dim]  (batch_size=1 implicit)
 *
 * @param Q Query tensor with shape [seq_len_q, n_heads, head_dim]
 * @param K Key tensor with shape [seq_len_k, n_kv_heads, head_dim]
 * @param V Value tensor with shape [seq_len_k, n_kv_heads, head_dim]
 * @param scale Scaling factor (usually 1/sqrt(head_dim)) to prevent large dot products
 * @param n_kv_heads Number of K/V heads (for GQA validation)
 * @param q_offset Absolute position offset for Q (for causal masking with KV cache)
 * @return Attention output with shape [seq_len_q, n_heads, head_dim]
 */
inline Tensor scaled_dot_product_attention(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    float scale,
    size_t n_kv_heads,
    size_t q_offset = 0
) {
    // Validate tensor dimensions (must be 3D: [seq_len, n_heads, head_dim])
    // No batch dimension - this is single-sequence inference
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3) {
        throw std::invalid_argument("attention: Q, K, V must be 3D tensors");
    }

    // Extract dimensions from Q tensor
    // Shape: [seq_len_q, n_heads, head_dim]
    size_t seq_len_q = Q.shape()[0];  // Number of query positions
    size_t n_heads = Q.shape()[1];    // Number of Q heads (can be > n_kv_heads for GQA)
    size_t head_dim = Q.shape()[2];   // Dimension of each head

    // K and V can have different sequence length (for cross-attention)
    // and different head count (for GQA)
    size_t seq_len_k = K.shape()[0];  // Number of key/value positions

    // Validate K and V shapes (they must match n_kv_heads, not n_heads for GQA)
    if (K.shape()[1] != n_kv_heads || K.shape()[2] != head_dim) {
        throw std::invalid_argument(
            "attention: K shape mismatch. Expected [" +
            std::to_string(seq_len_k) + ", " + std::to_string(n_kv_heads) +
            ", " + std::to_string(head_dim) + "], got [" +
            std::to_string(K.shape()[0]) + ", " + std::to_string(K.shape()[1]) +
            ", " + std::to_string(K.shape()[2]) + "]"
        );
    }
    if (V.shape()[0] != seq_len_k || V.shape()[1] != n_kv_heads || V.shape()[2] != head_dim) {
        throw std::invalid_argument(
            "attention: V shape mismatch. Expected [" +
            std::to_string(seq_len_k) + ", " + std::to_string(n_kv_heads) +
            ", " + std::to_string(head_dim) + "], got [" +
            std::to_string(V.shape()[0]) + ", " + std::to_string(V.shape()[1]) +
            ", " + std::to_string(V.shape()[2]) + "]"
        );
    }

    // Calculate how many Q heads share each K/V head (for GQA)
    // For MHA: num_queries_per_kv = 1 (each Q head has its own K/V head)
    // For GQA: num_queries_per_kv > 1 (K/V heads are shared)
    size_t num_queries_per_kv = n_heads / n_kv_heads;

    // Allocate output tensor: [seq_len_q, n_heads, head_dim]
    // Each head is computed independently and stored in the output
    Tensor output({seq_len_q, n_heads, head_dim});

    // Process each attention head independently IN PARALLEL
    // Uses ThreadPool to distribute heads across available cores
    ThreadPool::instance().parallel_for(0, n_heads, [&](size_t q_head) {
        // For GQA: calculate which K/V head this Q head uses
        // For MHA: kv_head == q_head (num_queries_per_kv == 1)
        size_t kv_head = q_head / num_queries_per_kv;

        // Extract Q, K, V slices for this specific head
        // After extraction, we have 2D tensors: [seq_len, head_dim]
        // This makes the matrix operations simpler
        Tensor Q_h({seq_len_q, head_dim});  // Queries for Q head q_head
        Tensor K_h({seq_len_k, head_dim});  // Keys for K/V head kv_head (shared!)
        Tensor V_h({seq_len_k, head_dim});  // Values for K/V head kv_head (shared!)

        // Copy data for this Q head from the 3D tensor to 2D working tensors
        // Q[i, q_head, j] -> Q_h[i, j] for all positions i and dimensions j
        for (size_t i = 0; i < seq_len_q; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                Q_h[i * head_dim + j] = Q.at({i, q_head, j});
            }
        }

        // Copy K and V from the shared K/V head (this is the key difference for GQA!)
        // K[i, kv_head, j] -> K_h[i, j] (note: using kv_head, not q_head)
        // V[i, kv_head, j] -> V_h[i, j]
        for (size_t i = 0; i < seq_len_k; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                K_h[i * head_dim + j] = K.at({i, kv_head, j});
                V_h[i * head_dim + j] = V.at({i, kv_head, j});
            }
        }

        // Step 1: Compute attention scores (similarity between queries and keys)
        // Q @ K^T gives us a matrix where scores[i, j] = similarity(Q[i], K[j])
        // Shape: Q_h [seq_len_q, head_dim] @ K_T [head_dim, seq_len_k]
        //      = scores [seq_len_q, seq_len_k]
        // Each row i contains how much position i attends to each position j
        Tensor K_T = ops::transpose(K_h);  // Transpose K: [head_dim, seq_len_k]
        Tensor scores = ops::matmul_naive(Q_h, K_T);  // [seq_len_q, seq_len_k]

        // Step 2: Scale scores to prevent extreme values
        // Without scaling, dot products can grow large (especially with large head_dim),
        // causing softmax to saturate and gradients to vanish
        // Scale factor is typically 1/sqrt(head_dim) to normalize variance
        for (size_t i = 0; i < scores.size(); ++i) {
            scores[i] *= scale;
        }

        // Step 2.5: Apply causal masking (CRITICAL for autoregressive generation!)
        // Causal mask ensures position i can only attend to positions 0...i (past/current)
        // This prevents "looking into the future" which would break autoregressive generation
        //
        // Without causal masking:
        //   - Tokens can see future tokens during attention
        //   - Model predictions become inconsistent
        //   - Often leads to degenerate behavior (repeated tokens, incoherence)
        //
        // Implementation: Set scores[i, j] = -inf for all j > (i + q_offset) (future positions)
        // After softmax, these positions will have probability ~0
        //
        // KV CACHE SUPPORT:
        // When using KV cache, Q positions are relative (e.g., [0]) but K positions are absolute (e.g., [0,1,2,3,4]).
        // q_offset tells us the absolute position of Q[0]. For example:
        //   - q_offset=4 means Q[0] is at absolute position 4
        //   - Q[0] should see K positions [0,1,2,3,4], mask positions [5,6,7,...]
        //   - Without q_offset, we'd incorrectly mask based on relative position (mask [1,2,3,4])
        //
        // We use two implementations:
        //   1. AVX2 SIMD (8x faster): Vectorized masking using 256-bit registers
        //   2. Scalar fallback: Portable std::fill_n for non-AVX2 systems

#if defined(__AVX2__)
        // AVX2 vectorized implementation: process 8 floats at a time
        const __m256 neg_inf_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

        for (size_t i = 0; i < seq_len_q; ++i) {
            size_t absolute_pos = i + q_offset;  // Absolute position in sequence
            size_t start_mask = absolute_pos + 1;  // Mask positions > absolute_pos

            if (start_mask < seq_len_k) {
                float* row_ptr = &scores[i * seq_len_k + start_mask];
                size_t num_future = seq_len_k - start_mask;
                size_t j = 0;

                // Process 8 floats at a time with AVX2
                constexpr size_t simd_width = 8;
                for (; j + simd_width <= num_future; j += simd_width) {
                    _mm256_storeu_ps(row_ptr + j, neg_inf_vec);
                }

                // Handle remaining elements (< 8 floats)
                for (; j < num_future; ++j) {
                    row_ptr[j] = -std::numeric_limits<float>::infinity();
                }
            }
        }
#else
        // Scalar fallback: use std::fill_n (still quite fast, uses memset internally)
        for (size_t i = 0; i < seq_len_q; ++i) {
            size_t absolute_pos = i + q_offset;  // Absolute position in sequence
            size_t start_mask = absolute_pos + 1;  // Mask positions > absolute_pos

            if (start_mask < seq_len_k) {
                float* row_ptr = &scores[i * seq_len_k + start_mask];
                size_t num_future = seq_len_k - start_mask;
                std::fill_n(row_ptr, num_future, -std::numeric_limits<float>::infinity());
            }
        }
#endif

        // Step 3: Apply softmax to convert scores to attention weights (probabilities)
        // Softmax is applied row-wise: each query position gets a probability distribution
        // over all key positions. Higher scores -> higher attention weights
        // Shape: attn_weights [seq_len_q, seq_len_k]
        //        attn_weights[i, j] = probability that position i attends to position j
        Tensor attn_weights = ops::softmax(scores);

        // Step 4: Weighted sum of values using attention weights
        // attn_weights @ V_h computes: for each query position i,
        //   output[i] = sum_j(attn_weights[i, j] * V[j])
        // This is the final attention output - a weighted combination of values
        // Shape: attn_weights [seq_len_q, seq_len_k] @ V_h [seq_len_k, head_dim]
        //      = output_h [seq_len_q, head_dim]
        Tensor output_h = ops::matmul_naive(attn_weights, V_h);

        // Step 5: Copy the computed head output back into the 3D output tensor
        // output_h[i, j] -> output[i, q_head, j]
        // All Q heads will be concatenated along the head dimension
        for (size_t i = 0; i < seq_len_q; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                output.at({i, q_head, j}) = output_h[i * head_dim + j];
            }
        }
    });

    // Return combined output from all heads: [seq_len_q, n_heads, head_dim]
    return output;
}

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
