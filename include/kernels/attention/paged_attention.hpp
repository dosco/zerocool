#pragma once

#include "core/tensor.hpp"
#include "kernels/tensor_ops.hpp"
#include "kernels/attention/masking.hpp"
#include "core/paged_kv_cache.hpp"
#include "infra/thread_pool.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace freellm {

/**
 * @brief Process a single attention head with Paged KV Cache
 */
inline void process_paged_attention_head(
    size_t q_head,
    const Tensor& Q,
    const std::vector<int32_t>& block_table,
    KVCacheManager& kv_manager,
    Tensor& output,
    float scale,
    size_t num_queries_per_kv,
    size_t seq_len_q,
    size_t current_seq_len, // Total sequence length (cached + new)
    size_t head_dim,
    size_t q_offset
) {
    size_t kv_head = q_head / num_queries_per_kv;
    size_t block_size = kv_manager.config().block_size;

    // Extract Q for this head: [seq_len_q, head_dim]
    Tensor Q_h({seq_len_q, head_dim});
    for (size_t i = 0; i < seq_len_q; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
            Q_h[i * head_dim + j] = Q.at({i, q_head, j});
        }
    }

    // Reconstruct K_h and V_h for this head from paged memory
    Tensor K_h({current_seq_len, head_dim});
    Tensor V_h({current_seq_len, head_dim});

    // Gather K and V from blocks
    for (size_t i = 0; i < current_seq_len; ++i) {
        size_t logical_block_idx = i / block_size;
        size_t block_offset = i % block_size;
        int32_t physical_block_id = block_table[logical_block_idx];

        // Access global K/V tensors
        const Tensor& global_K = kv_manager.global_keys();
        const Tensor& global_V = kv_manager.global_values();

        for (size_t j = 0; j < head_dim; ++j) {
            // Copy K
            K_h[i * head_dim + j] = global_K.at({static_cast<size_t>(physical_block_id), block_offset, kv_head, j});
            // Copy V
            V_h[i * head_dim + j] = global_V.at({static_cast<size_t>(physical_block_id), block_offset, kv_head, j});
        }
    }

    // Step 1: Scores = Q @ K^T
    Tensor K_T = ops::transpose(K_h);
    Tensor scores = ops::matmul(Q_h, K_T); // [seq_len_q, current_seq_len]

    // Step 2: Scale
    for (size_t i = 0; i < scores.size(); ++i) {
        scores[i] *= scale;
    }

    // Step 3: Causal Masking
    apply_causal_mask(scores, seq_len_q, current_seq_len, q_offset);

    // Step 4: Softmax
    Tensor attn_weights = ops::softmax(scores);

    // Step 5: Output = Weights @ V
    Tensor output_h = ops::matmul(attn_weights, V_h);

    // Step 6: Copy back to output tensor
    for (size_t i = 0; i < seq_len_q; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
            output.at({i, q_head, j}) = output_h[i * head_dim + j];
        }
    }
}

/**
 * @brief Scaled dot-product attention using Paged KV Cache
 */
inline Tensor paged_scaled_dot_product_attention(
    const Tensor& Q,
    PagedKVCache& cache,
    float scale,
    size_t n_kv_heads,
    size_t q_offset = 0
) {
    // Validate input
    if (Q.ndim() != 3) {
        throw std::invalid_argument("paged_attention: Q must be 3D");
    }

    size_t seq_len_q = Q.shape()[0];
    size_t n_heads = Q.shape()[1];
    size_t head_dim = Q.shape()[2];
    
    size_t current_seq_len = cache.current_length(); 
    const auto& block_table = cache.block_table();
    KVCacheManager* manager = cache.manager();
    
    if (!manager) {
        throw std::runtime_error("paged_attention: Cache has no manager");
    }

    size_t num_queries_per_kv = n_heads / n_kv_heads;
    Tensor output({seq_len_q, n_heads, head_dim});

    // Process heads in parallel
    ThreadPool::instance().parallel_for(0, n_heads, [&](size_t q_head) {
        process_paged_attention_head(
            q_head, Q, block_table, *manager, output, scale, num_queries_per_kv,
            seq_len_q, current_seq_len, head_dim, q_offset
        );
    });

    return output;
}

} // namespace freellm
