#pragma once

#include "core/tensor.hpp"
#include <vector>
#include <list>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <cmath>
#include <algorithm>
#include "infra/compute_backend.hpp"

namespace freellm {

enum class KvCacheDataType {
    FP32,
    INT8
};

struct KVCacheConfig {
    size_t block_size = 16;      // Number of tokens per block
    size_t max_num_blocks = 1024; // Maximum number of blocks in the pool
    size_t n_kv_heads = 0;       // Number of KV heads
    size_t head_dim = 0;         // Dimension of each head
    KvCacheDataType data_type = KvCacheDataType::FP32; // Data type (FP32 or INT8)
};

/**
 * @brief Manages the global pool of KV cache blocks
 * 
 * This class allocates a large contiguous memory pool for keys and values.
 * It divides this pool into fixed-size blocks and manages their allocation/deallocation.
 */
class KVCacheManager {
public:
    KVCacheManager(const KVCacheConfig& config, infra::ComputeBackend* backend = nullptr) 
        : config_(config), backend_(backend) {
        
        if (backend_) {
             size_t total_elements = config.max_num_blocks * config.block_size * config.n_kv_heads * config.head_dim;
             
             if (config.data_type == KvCacheDataType::INT8) {
                 // Allocate INT8 buffers
                 size_t total_bytes = total_elements * sizeof(uint8_t); // or int8_t
                 global_keys_int8_device_ = backend_->allocate(total_bytes, infra::DType::INT8);
                 global_values_int8_device_ = backend_->allocate(total_bytes, infra::DType::INT8);
                 
                 // Allocate Scale buffers (One float per token (slot) per head)
                 // Layout: [max_num_blocks, block_size, n_kv_heads] to align with data layout logic
                 // Actually, data layout is [block, block_size, kv_head, head_dim] usually flattened.
                 // Scales: [max_num_blocks, block_size, n_kv_heads]
                 size_t total_scales = config.max_num_blocks * config.block_size * config.n_kv_heads;
                 size_t scale_bytes = total_scales * sizeof(float);
                 global_k_scales_device_ = backend_->allocate(scale_bytes, infra::DType::FLOAT32);
                 global_v_scales_device_ = backend_->allocate(scale_bytes, infra::DType::FLOAT32);

             } else {
                 // FP32
                 size_t total_bytes = total_elements * sizeof(float);
                 global_keys_device_ = backend_->allocate(total_bytes, infra::DType::FLOAT32);
                 global_values_device_ = backend_->allocate(total_bytes, infra::DType::FLOAT32);
             }


        } else {
            // CPU Support fallback (currently assumes FP32 for simplicity or TODO)
            if (config.data_type == KvCacheDataType::INT8) {
                 throw std::runtime_error("INT8 KV Cache currently only supported on GPU Backend");
            }
            // Allocate global storage (CPU)
            global_keys_ = Tensor({config.max_num_blocks, config.block_size, config.n_kv_heads, config.head_dim});
            global_values_ = Tensor({config.max_num_blocks, config.block_size, config.n_kv_heads, config.head_dim});
        }

        // Initialize free list with all block indices
        for (int32_t i = 0; i < static_cast<int32_t>(config.max_num_blocks); ++i) {
            free_blocks_.push_back(i);
        }

        if (backend_) {
            // Helper to alloc scalar
            auto alloc_u32 = [&](uint32_t val) {
                auto buf = backend_->allocate(sizeof(uint32_t), infra::DType::INT32);
                backend_->copy_to_device(buf.get(), 0, &val, sizeof(uint32_t));
                return buf;
            };
            scalar_head_dim_ = alloc_u32((uint32_t)config.head_dim);
            scalar_n_kv_heads_ = alloc_u32((uint32_t)config.n_kv_heads);
            scalar_block_size_ = alloc_u32((uint32_t)config.block_size);
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

    // Getters for global tensors (FP32)
    Tensor& global_keys() { return global_keys_; }
    Tensor& global_values() { return global_values_; }
    const Tensor& global_keys() const { return global_keys_; }
    const Tensor& global_values() const { return global_values_; }
    
    // Getters for Device Buffers (FP32)
    infra::DeviceBuffer* global_keys_device() { return global_keys_device_.get(); }
    infra::DeviceBuffer* global_values_device() { return global_values_device_.get(); }

    // Getters for Device Buffers (INT8)
    infra::DeviceBuffer* global_keys_int8_device() { return global_keys_int8_device_.get(); }
    infra::DeviceBuffer* global_values_int8_device() { return global_values_int8_device_.get(); }
    infra::DeviceBuffer* global_k_scales_device() { return global_k_scales_device_.get(); }
    infra::DeviceBuffer* global_v_scales_device() { return global_v_scales_device_.get(); }
    
    // Scalar Getters
    infra::DeviceBuffer* scalar_head_dim() { return scalar_head_dim_.get(); }
    infra::DeviceBuffer* scalar_n_kv_heads() { return scalar_n_kv_heads_.get(); }
    infra::DeviceBuffer* scalar_block_size() { return scalar_block_size_.get(); }

    const KVCacheConfig& config() const { return config_; }

    size_t free_blocks_count() const {
        return free_blocks_.size();
    }

    infra::ComputeBackend* backend() const { return backend_; }

private:
    KVCacheConfig config_;
    Tensor global_keys_;
    Tensor global_values_;
    std::list<int32_t> free_blocks_;
    std::mutex mutex_;
    
    infra::ComputeBackend* backend_;
    
    // FP32 Buffers
    std::unique_ptr<infra::DeviceBuffer> global_keys_device_;
    std::unique_ptr<infra::DeviceBuffer> global_values_device_;
    
    // INT8 Buffers + Scales
    std::unique_ptr<infra::DeviceBuffer> global_keys_int8_device_;
    std::unique_ptr<infra::DeviceBuffer> global_values_int8_device_;
    std::unique_ptr<infra::DeviceBuffer> global_k_scales_device_;
    std::unique_ptr<infra::DeviceBuffer> global_v_scales_device_;

    // Scalars
    std::unique_ptr<infra::DeviceBuffer> scalar_head_dim_;
    std::unique_ptr<infra::DeviceBuffer> scalar_n_kv_heads_;
    std::unique_ptr<infra::DeviceBuffer> scalar_block_size_;
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

            if (manager_->backend()) {
                // Copy Key to Device
                copy_slot_device(new_keys, i, manager_->global_keys_device(), physical_block_id, block_offset);
                // Copy Value to Device
                copy_slot_device(new_values, i, manager_->global_values_device(), physical_block_id, block_offset);
            } else {
                // Copy Key (CPU)
                copy_slot(new_keys, i, manager_->global_keys(), physical_block_id, block_offset);
                // Copy Value (CPU)
                copy_slot(new_values, i, manager_->global_values(), physical_block_id, block_offset);
            }
        }

        current_len_ += seq_len_new;
    }

    // Update cache with new keys and values from DeviceBuffer
    void update(infra::DeviceBuffer* new_keys, infra::DeviceBuffer* new_values, size_t seq_len, size_t src_base_offset_bytes = 0) {
        size_t block_size = manager_->config().block_size;
        
        // Pre-calculate physical destinations for all tokens
        // For INT8, we need to upload these relative offsets to GPU.
        // For FP32, we loop on CPU.
        
        if (manager_->config().data_type == KvCacheDataType::INT8) {
            std::vector<int32_t> offsets;
            offsets.reserve(seq_len);

            for (size_t i = 0; i < seq_len; ++i) {
                size_t logical_pos = current_len_ + i;
                size_t logical_block_idx = logical_pos / block_size;
                size_t block_offset = logical_pos % block_size;

                if (logical_block_idx >= block_table_.size()) {
                    block_table_.push_back(manager_->allocate_block());
                }

                int32_t physical_block_id = block_table_[logical_block_idx];
                
                // Calculate slot index in global pool
                // Global layout: [total_slots, n_kv_heads, head_dim]
                // Slot Index = physical_block_id * block_size + block_offset
                offsets.push_back(physical_block_id * block_size + block_offset);
            }
            
            // Upload offsets to scratch buffer
            size_t bytes = offsets.size() * sizeof(int32_t);
            if (!offsets_scratch_ || offsets_scratch_->size_bytes() < bytes) {
                // Resize (Double capacity to avoid frequent realloc)
                size_t new_cap = std::max(bytes * 2, (size_t)1024);
                offsets_scratch_ = manager_->backend()->allocate(new_cap, infra::DType::INT32);
            }
            manager_->backend()->copy_to_device(offsets_scratch_.get(), offsets.data(), bytes);
            
            // Create scalar for src_elem_offset
            // src_base_offset_bytes MUST be float aligned.
            uint32_t src_elem_offset = src_base_offset_bytes / sizeof(float);
            auto scalar_src_offset = manager_->backend()->allocate(sizeof(uint32_t), infra::DType::INT32);
            manager_->backend()->copy_to_device(scalar_src_offset.get(), 0, &src_elem_offset, sizeof(uint32_t));
            
            // Launch Kernels for Quantization
            // Grid: (seq_len * 32, n_kv_heads, 1) threads total (Metal dispatchThreads)
            // Block: (32, 1, 1) - 32 threads per threadgroup (SIMD width)
            infra::KernelConfig grid;
            grid.grid = infra::Dim3(seq_len * 32, manager_->config().n_kv_heads, 1);
            grid.block = infra::Dim3(32, 1, 1);
            
            // Keys
            manager_->backend()->execute_kernel("quantize_store_kv",
                {new_keys, manager_->global_keys_int8_device(), manager_->global_k_scales_device(), offsets_scratch_.get(),
                 manager_->scalar_head_dim(), manager_->scalar_n_kv_heads(), scalar_src_offset.get()},
                {}, grid);
                
            // Values
            manager_->backend()->execute_kernel("quantize_store_kv",
                {new_values, manager_->global_values_int8_device(), manager_->global_v_scales_device(), offsets_scratch_.get(),
                 manager_->scalar_head_dim(), manager_->scalar_n_kv_heads(), scalar_src_offset.get()},
                {}, grid);
                
            current_len_ += seq_len;
            return;
        }

        // FP32 Loop
        for (size_t i = 0; i < seq_len; ++i) {
            size_t logical_pos = current_len_ + i;
            size_t logical_block_idx = logical_pos / block_size;
            size_t block_offset = logical_pos % block_size;

            if (logical_block_idx >= block_table_.size()) {
                block_table_.push_back(manager_->allocate_block());
            }

            int32_t physical_block_id = block_table_[logical_block_idx];

            if (manager_->backend()) {
                // Device to Device copy
                copy_slot_device_to_device(new_keys, i, manager_->global_keys_device(), physical_block_id, block_offset, src_base_offset_bytes);
                copy_slot_device_to_device(new_values, i, manager_->global_values_device(), physical_block_id, block_offset, src_base_offset_bytes);
            } else {
                throw std::runtime_error("Cannot update from DeviceBuffer without backend");
            }
        }
        current_len_ += seq_len;
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

    void copy_slot_device(const Tensor& src, size_t src_idx, infra::DeviceBuffer* dst, int32_t block_id, size_t block_offset) {
        size_t n_kv_heads = manager_->config().n_kv_heads;
        size_t head_dim = manager_->config().head_dim;
        size_t slot_size_bytes = n_kv_heads * head_dim * sizeof(float);
        
        const float* src_ptr = src.data() + src_idx * n_kv_heads * head_dim;
        
        // Calculate byte offset in destination buffer
        size_t dst_idx = (block_id * manager_->config().block_size + block_offset) * n_kv_heads * head_dim;
        size_t dst_offset_bytes = dst_idx * sizeof(float);
        
        manager_->backend()->copy_to_device(dst, dst_offset_bytes, src_ptr, slot_size_bytes);
    }

    void copy_slot_device_to_device(infra::DeviceBuffer* src, size_t src_idx, infra::DeviceBuffer* dst, int32_t block_id, size_t block_offset, size_t src_base_offset_bytes = 0) {
        size_t n_kv_heads = manager_->config().n_kv_heads;
        size_t head_dim = manager_->config().head_dim;
        size_t slot_size_bytes = n_kv_heads * head_dim * sizeof(float);
        
        size_t src_offset_bytes = src_base_offset_bytes + src_idx * slot_size_bytes;
        
        size_t dst_idx = (block_id * manager_->config().block_size + block_offset) * n_kv_heads * head_dim;
        size_t dst_offset_bytes = dst_idx * sizeof(float);
        
        manager_->backend()->copy_device_to_device(dst, dst_offset_bytes, src, src_offset_bytes, slot_size_bytes);
    }

    KVCacheManager* manager_;
    std::vector<int32_t> block_table_; // Maps logical block index -> physical block ID
    size_t current_len_;
    std::unique_ptr<infra::DeviceBuffer> offsets_scratch_;
};

} // namespace freellm
