#pragma once

#include "tensor.hpp"
#include <cstddef>
#include <stdexcept>

namespace freellm {

/*
 * KV Cache: Key-Value Cache for Transformer Attention
 * =====================================================
 *
 * WHAT IS KV CACHE?
 * -----------------
 * During autoregressive text generation, transformers generate tokens one at a time.
 * For each new token, we compute attention with ALL previous tokens in the sequence.
 *
 * WITHOUT KV cache:
 *   Step 1: Process token 1, compute K₁, V₁
 *   Step 2: Process tokens 1-2, RE-compute K₁, V₁, K₂, V₂
 *   Step 3: Process tokens 1-3, RE-compute K₁, V₁, K₂, V₂, K₃, V₃
 *   ...
 *   Complexity: O(N²) where N = sequence length
 *
 * WITH KV cache:
 *   Step 1: Process token 1, compute K₁, V₁ → cache them
 *   Step 2: Process token 2, compute ONLY K₂, V₂ → append to cache
 *   Step 3: Process token 3, compute ONLY K₃, V₃ → append to cache
 *   ...
 *   Complexity: O(N)
 *
 * This gives 10-100x speedup for generation!
 *
 * WHY IT WORKS:
 * -------------
 * Key insight: In transformer attention, the Key and Value matrices for previous
 * tokens NEVER change. Only the Query for the new token changes.
 *
 *   Attention(Q, K, V) = softmax(Q·K^T / √d) · V
 *
 * For the new token at position t:
 *   - Q_t depends only on token t (NEW - must compute)
 *   - K_t depends only on token t (NEW - must compute)
 *   - V_t depends only on token t (NEW - must compute)
 *   - K₁...K_{t-1} were already computed (CACHED - reuse!)
 *   - V₁...V_{t-1} were already computed (CACHED - reuse!)
 *
 * GROUPED QUERY ATTENTION (GQA):
 * -------------------------------
 * This implementation supports GQA, where multiple query heads share fewer KV heads.
 * Example: TinyLLaMA has 32 query heads but only 4 KV heads (8:1 ratio).
 *
 * Cache shape: [max_seq_len, n_kv_heads, head_dim]
 * NOT [max_seq_len, n_heads, head_dim] - this is critical!
 *
 * MEMORY USAGE:
 * -------------
 * Per layer: max_seq_len × n_kv_heads × head_dim × sizeof(float) × 2 (K + V)
 * Example (TinyLLaMA, max_seq_len=2048):
 *   2048 × 4 × 64 × 4 bytes × 2 = 4.2 MB per layer
 *   22 layers × 4.2 MB = ~92 MB total (acceptable!)
 *
 * USAGE EXAMPLE:
 * --------------
 * // Create cache for one attention layer
 * KVCache cache(max_seq_len=2048, n_kv_heads=4, head_dim=64);
 *
 * // Generation loop
 * cache.reset();  // Clear before new sequence
 * for (int step = 0; step < max_tokens; ++step) {
 *     // Compute K, V for ONLY the new token
 *     Tensor K_new = compute_keys(new_token);    // [1, n_kv_heads, head_dim]
 *     Tensor V_new = compute_values(new_token);  // [1, n_kv_heads, head_dim]
 *
 *     // Update cache
 *     cache.update(K_new, V_new);
 *
 *     // Get all keys/values (cached + new)
 *     Tensor K_all = cache.get_keys();    // [current_len, n_kv_heads, head_dim]
 *     Tensor V_all = cache.get_values();  // [current_len, n_kv_heads, head_dim]
 *
 *     // Compute attention using K_all, V_all
 *     ...
 * }
 */

class KVCache {
public:
    /*
     * Constructor: Initialize KV cache storage
     *
     * Parameters:
     *   max_seq_len: Maximum sequence length (cache capacity)
     *   n_kv_heads: Number of KV heads (for GQA - NOT n_heads!)
     *   head_dim: Dimension of each attention head
     *
     * Allocates:
     *   cached_keys_: [max_seq_len, n_kv_heads, head_dim]
     *   cached_values_: [max_seq_len, n_kv_heads, head_dim]
     */
    KVCache(size_t max_seq_len, size_t n_kv_heads, size_t head_dim)
        : max_seq_len_(max_seq_len),
          n_kv_heads_(n_kv_heads),
          head_dim_(head_dim),
          current_len_(0),
          cached_keys_({max_seq_len, n_kv_heads, head_dim}),
          cached_values_({max_seq_len, n_kv_heads, head_dim})
    {
        // Tensors are already initialized with correct shape in initializer list
    }

    /*
     * Reset cache to empty state
     *
     * Call this before processing a new prompt/sequence.
     * Does NOT deallocate memory, just resets the length counter.
     */
    void reset() {
        current_len_ = 0;
    }

    /*
     * Update cache with new keys and values
     *
     * Appends new K/V tensors to the cache at position current_len_.
     *
     * Parameters:
     *   new_keys: [seq_len_new, n_kv_heads, head_dim] - Keys for new tokens
     *   new_values: [seq_len_new, n_kv_heads, head_dim] - Values for new tokens
     *
     * Typical usage: seq_len_new = 1 (one new token at a time)
     * But can handle multiple tokens (e.g., prompt processing)
     *
     * Shape transformations:
     *   Input:  new_keys [seq_len_new, n_kv_heads, head_dim]
     *   Cache:  cached_keys_ [max_seq_len, n_kv_heads, head_dim]
     *   Action: Copy new_keys into cached_keys_[current_len_:current_len_+seq_len_new, :, :]
     *   Result: current_len_ += seq_len_new
     */
    void update(const Tensor& new_keys, const Tensor& new_values) {
        const auto& k_shape = new_keys.shape();
        const auto& v_shape = new_values.shape();

        // Validate shapes
        if (k_shape.size() != 3 || v_shape.size() != 3) {
            throw std::runtime_error("KVCache::update: Keys and Values must be 3D tensors");
        }

        size_t seq_len_new = k_shape[0];
        size_t kv_heads_k = k_shape[1];
        size_t head_dim_k = k_shape[2];
        size_t kv_heads_v = v_shape[1];
        size_t head_dim_v = v_shape[2];

        if (kv_heads_k != n_kv_heads_ || head_dim_k != head_dim_) {
            throw std::runtime_error(
                "KVCache::update: Key shape mismatch. Expected [*, " +
                std::to_string(n_kv_heads_) + ", " + std::to_string(head_dim_) +
                "], got [" + std::to_string(seq_len_new) + ", " +
                std::to_string(kv_heads_k) + ", " + std::to_string(head_dim_k) + "]"
            );
        }

        if (kv_heads_v != n_kv_heads_ || head_dim_v != head_dim_) {
            throw std::runtime_error(
                "KVCache::update: Value shape mismatch. Expected [*, " +
                std::to_string(n_kv_heads_) + ", " + std::to_string(head_dim_) +
                "], got [" + std::to_string(seq_len_new) + ", " +
                std::to_string(kv_heads_v) + ", " + std::to_string(head_dim_v) + "]"
            );
        }

        if (v_shape[0] != seq_len_new) {
            throw std::runtime_error("KVCache::update: Keys and Values must have same seq_len");
        }

        if (current_len_ + seq_len_new > max_seq_len_) {
            throw std::runtime_error(
                "KVCache::update: Cache overflow. Current length: " +
                std::to_string(current_len_) + ", adding: " +
                std::to_string(seq_len_new) + ", max: " +
                std::to_string(max_seq_len_)
            );
        }

        // Copy new keys and values into cache
        // We need to copy data manually since we're appending to specific positions
        const float* k_data = new_keys.data();
        const float* v_data = new_values.data();
        float* cached_k_data = cached_keys_.data();
        float* cached_v_data = cached_values_.data();

        // Calculate offsets for cache insertion
        // Cache layout: [max_seq_len, n_kv_heads, head_dim]
        // Offset to position current_len_: current_len_ * n_kv_heads_ * head_dim_
        size_t offset = current_len_ * n_kv_heads_ * head_dim_;
        size_t copy_size = seq_len_new * n_kv_heads_ * head_dim_;

        // Copy keys
        std::copy(k_data, k_data + copy_size, cached_k_data + offset);

        // Copy values
        std::copy(v_data, v_data + copy_size, cached_v_data + offset);

        // Update length
        current_len_ += seq_len_new;
    }

    /*
     * Get all cached keys up to current position
     *
     * Returns: Tensor [current_len_, n_kv_heads, head_dim]
     *
     * This is a VIEW into the cache (shares underlying data).
     * Only the first current_len_ positions contain valid data.
     *
     * Shape transformation:
     *   Cache:  [max_seq_len, n_kv_heads, head_dim]
     *   Return: [current_len_, n_kv_heads, head_dim]
     */
    Tensor get_keys() const {
        if (current_len_ == 0) {
            // Return empty tensor with correct shape
            return Tensor({0, n_kv_heads_, head_dim_});
        }

        // Create a new tensor with the cached data up to current_len_
        // We need to copy only the valid portion
        size_t valid_size = current_len_ * n_kv_heads_ * head_dim_;
        Tensor result({current_len_, n_kv_heads_, head_dim_});

        const float* cached_data = cached_keys_.data();
        float* result_data = result.data();
        std::copy(cached_data, cached_data + valid_size, result_data);

        return result;
    }

    /*
     * Get all cached values up to current position
     *
     * Returns: Tensor [current_len_, n_kv_heads, head_dim]
     *
     * This is a VIEW into the cache (shares underlying data).
     * Only the first current_len_ positions contain valid data.
     *
     * Shape transformation:
     *   Cache:  [max_seq_len, n_kv_heads, head_dim]
     *   Return: [current_len_, n_kv_heads, head_dim]
     */
    Tensor get_values() const {
        if (current_len_ == 0) {
            // Return empty tensor with correct shape
            return Tensor({0, n_kv_heads_, head_dim_});
        }

        // Create a new tensor with the cached data up to current_len_
        size_t valid_size = current_len_ * n_kv_heads_ * head_dim_;
        Tensor result({current_len_, n_kv_heads_, head_dim_});

        const float* cached_data = cached_values_.data();
        float* result_data = result.data();
        std::copy(cached_data, cached_data + valid_size, result_data);

        return result;
    }

    /*
     * Get current cache length (number of tokens cached)
     */
    size_t current_length() const {
        return current_len_;
    }

    /*
     * Get maximum cache capacity
     */
    size_t max_length() const {
        return max_seq_len_;
    }

    /*
     * Get number of KV heads
     */
    size_t n_kv_heads() const {
        return n_kv_heads_;
    }

    /*
     * Get head dimension
     */
    size_t head_dim() const {
        return head_dim_;
    }

    /*
     * Get memory usage in bytes
     *
     * Returns total memory allocated for K and V caches.
     */
    size_t memory_bytes() const {
        return 2 * max_seq_len_ * n_kv_heads_ * head_dim_ * sizeof(float);
    }

private:
    size_t max_seq_len_;   // Maximum sequence length (cache capacity)
    size_t n_kv_heads_;    // Number of KV heads (for GQA)
    size_t head_dim_;      // Dimension of each head
    size_t current_len_;   // Current number of cached positions

    // Storage for cached keys and values
    // Shape: [max_seq_len, n_kv_heads, head_dim]
    Tensor cached_keys_;
    Tensor cached_values_;
};

} // namespace freellm
