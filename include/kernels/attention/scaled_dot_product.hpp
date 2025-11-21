#pragma once

#include "core/tensor.hpp"
#include "kernels/tensor_ops.hpp"
#include "kernels/attention/masking.hpp"
#include "infra/thread_pool.hpp"
#include <cmath>
#include <algorithm>

namespace freellm {

/**
 * @brief Process a single attention head (helper for scaled_dot_product_attention)
 * 
 * Computes attention for a specific query head, handling GQA and causal masking.
 */
inline void process_attention_head(
    size_t q_head,
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& output,
    float scale,
    size_t num_queries_per_kv,
    size_t seq_len_q,
    size_t seq_len_k,
    size_t head_dim,
    size_t q_offset
) {
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
    Tensor scores = ops::matmul(Q_h, K_T);  // [seq_len_q, seq_len_k]

    // Step 2: Scale scores to prevent extreme values
    // Without scaling, dot products can grow large (especially with large head_dim),
    // causing softmax to saturate and gradients to vanish
    // Scale factor is typically 1/sqrt(head_dim) to normalize variance
    for (size_t i = 0; i < scores.size(); ++i) {
        scores[i] *= scale;
    }

    // Step 2.5: Apply causal masking (CRITICAL for autoregressive generation!)
    apply_causal_mask(scores, seq_len_q, seq_len_k, q_offset);

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
    Tensor output_h = ops::matmul(attn_weights, V_h);

    // Step 5: Copy the computed head output back into the 3D output tensor
    // output_h[i, j] -> output[i, q_head, j]
    // All Q heads will be concatenated along the head dimension
    for (size_t i = 0; i < seq_len_q; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
            output.at({i, q_head, j}) = output_h[i * head_dim + j];
        }
    }
}

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
        process_attention_head(
            q_head, Q, K, V, output, scale, num_queries_per_kv,
            seq_len_q, seq_len_k, head_dim, q_offset
        );
    });

    // Return combined output from all heads: [seq_len_q, n_heads, head_dim]
    return output;
}

} // namespace freellm
