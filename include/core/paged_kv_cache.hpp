#pragma once

#include "core/tensor.hpp"
#include <vector>
#include <list>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <cmath>
#include <algorithm>

namespace freellm {

struct KVCacheConfig {
    size_t block_size = 16;      // Number of tokens per block
    size_t max_num_blocks = 1024; // Maximum number of blocks in the pool
    size_t n_kv_heads = 0;       // Number of KV heads
    size_t head_dim = 0;         // Dimension of each head
};

/**
 * @brief Manages the global pool of KV cache blocks
 * 
 * This class allocates a large contiguous memory pool for keys and values.
 * It divides this pool into fixed-size blocks and manages their allocation/deallocation.
 */
class KVCacheManager {
public:
    KVCacheManager(const KVCacheConfig& config) : config_(config) {
        // Calculate total size for pre-allocation
        // Shape: [max_num_blocks, block_size, n_kv_heads, head_dim]
        // size_t total_elements = config.max_num_blocks * config.block_size * config.n_kv_heads * config.head_dim;
        
        // Allocate global storage
        global_keys_ = Tensor({config.max_num_blocks, config.block_size, config.n_kv_heads, config.head_dim});
        global_values_ = Tensor({config.max_num_blocks, config.block_size, config.n_kv_heads, config.head_dim});

        // Initialize free list with all block indices
        for (int32_t i = 0; i < static_cast<int32_t>(config.max_num_blocks); ++i) {
            free_blocks_.push_back(i);
        }
    }

    // Allocate a new block
    int32_t allocate_block() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_blocks_.empty()) {
            throw std::runtime_error("KVCacheManager: Out of memory (no free blocks available)");
        }
        int32_t block_id = free_blocks_.front();
        free_blocks_.pop_front();
        return block_id;
    }

    // Free a block and return it to the pool
    void free_block(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (block_id < 0 || static_cast<size_t>(block_id) >= config_.max_num_blocks) {
            throw std::invalid_argument("KVCacheManager: Invalid block_id to free");
        }
        free_blocks_.push_back(block_id);
    }

    // Getters for global tensors
    Tensor& global_keys() { return global_keys_; }
    Tensor& global_values() { return global_values_; }
    const Tensor& global_keys() const { return global_keys_; }
    const Tensor& global_values() const { return global_values_; }
    
    const KVCacheConfig& config() const { return config_; }

    size_t free_blocks_count() const {
        return free_blocks_.size();
    }

private:
    KVCacheConfig config_;
    Tensor global_keys_;
    Tensor global_values_;
    std::list<int32_t> free_blocks_;
    std::mutex mutex_;
};

/**
 * @brief Paged KV Cache for a single sequence
 * 
 * Maintains the mapping between logical blocks (sequence position) and physical blocks (memory pool).
 */
class PagedKVCache {
public:
    PagedKVCache(KVCacheManager* manager) 
        : manager_(manager), current_len_(0) {}

    ~PagedKVCache() {
        // Free all allocated blocks when destroyed
        reset();
    }

    // Reset the cache and free all blocks
    void reset() {
        for (int32_t block_id : block_table_) {
            manager_->free_block(block_id);
        }
        block_table_.clear();
        current_len_ = 0;
    }

    // Update cache with new keys and values
    void update(const Tensor& new_keys, const Tensor& new_values) {
        size_t seq_len_new = new_keys.shape()[0];
        size_t block_size = manager_->config().block_size;
        
        // Iterate over new tokens and copy them to blocks
        for (size_t i = 0; i < seq_len_new; ++i) {
            size_t logical_pos = current_len_ + i;
            size_t logical_block_idx = logical_pos / block_size;
            size_t block_offset = logical_pos % block_size;

            // Allocate new block if needed
            if (logical_block_idx >= block_table_.size()) {
                block_table_.push_back(manager_->allocate_block());
            }

            int32_t physical_block_id = block_table_[logical_block_idx];

            // Copy Key
            // Source: new_keys[i]
            // Dest: global_keys_[physical_block_id, block_offset]
            copy_slot(new_keys, i, manager_->global_keys(), physical_block_id, block_offset);

            // Copy Value
            // Source: new_values[i]
            // Dest: global_values_[physical_block_id, block_offset]
            copy_slot(new_values, i, manager_->global_values(), physical_block_id, block_offset);
        }

        current_len_ += seq_len_new;
    }

    const std::vector<int32_t>& block_table() const {
        return block_table_;
    }

    size_t current_length() const {
        return current_len_;
    }

    KVCacheManager* manager() const {
        return manager_;
    }

private:
    void copy_slot(const Tensor& src, size_t src_idx, Tensor& dst, int32_t block_id, size_t block_offset) {
        // src shape: [seq_len_new, n_kv_heads, head_dim]
        // dst shape: [max_blocks, block_size, n_kv_heads, head_dim]
        
        size_t n_kv_heads = manager_->config().n_kv_heads;
        size_t head_dim = manager_->config().head_dim;
        
        const float* src_ptr = src.data() + src_idx * n_kv_heads * head_dim;
        float* dst_ptr = dst.data() + (block_id * manager_->config().block_size + block_offset) * n_kv_heads * head_dim;
        
        std::copy(src_ptr, src_ptr + n_kv_heads * head_dim, dst_ptr);
    }

    KVCacheManager* manager_;
    std::vector<int32_t> block_table_; // Maps logical block index -> physical block ID
    size_t current_len_;
};

} // namespace freellm
