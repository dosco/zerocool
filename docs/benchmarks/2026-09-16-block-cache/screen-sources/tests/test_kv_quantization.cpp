#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/paged_kv_cache.hpp"
#include "infra/metal_backend.hpp"
#include <vector>
#include <iostream>
#include <cmath>

using namespace freellm;

TEST_CASE("Int8 KV Cache Quantization") {
    // 1. Setup
    KVCacheConfig config;
    config.block_size = 16;
    config.max_num_blocks = 4;
    config.n_kv_heads = 2; 
    config.head_dim = 64;
    config.data_type = KvCacheDataType::INT8;

    infra::MetalBackend backend(0);
    KVCacheManager manager(config, &backend);
    PagedKVCache cache(&manager);

    // 2. Data
    size_t seq_len = 20; // > block_size
    // Create Float data [seq_len, n_kv_heads, head_dim]
    size_t size = seq_len * config.n_kv_heads * config.head_dim;
    std::vector<float> k_data(size);
    std::vector<float> v_data(size);
    
    // Fill with pattern
    for(size_t i=0; i<size; ++i) {
        k_data[i] = (i % 127) * 0.1f * ((i % 2 == 0) ? 1.0f : -1.0f); // Range -12.7 to 12.7 roughly
        v_data[i] = (i % 50) * 0.5f;
    }
    
    auto k_buf = backend.allocate(size * sizeof(float), infra::DType::FLOAT32);
    auto v_buf = backend.allocate(size * sizeof(float), infra::DType::FLOAT32);
    backend.copy_to_device(k_buf.get(), k_data.data(), size * sizeof(float));
    backend.copy_to_device(v_buf.get(), v_data.data(), size * sizeof(float));

    // 3. Update
    cache.update(k_buf.get(), v_buf.get(), seq_len);
    backend.synchronize();

    // 4. Verification
    // Read back global INT8 and Scales
    // Use config from manager which should match
    size_t total_slots = manager.config().max_num_blocks * manager.config().block_size;
    size_t total_elements = total_slots * config.n_kv_heads * config.head_dim;
    size_t total_scales = total_slots * config.n_kv_heads;

    std::vector<int8_t> k_int8(total_elements);
    std::vector<float> k_scales(total_scales);
    
    backend.copy_to_host(k_scales.data(), manager.global_k_scales_device(), total_scales * sizeof(float));

    const auto& table = cache.block_table();
    
    int errors = 0;
    
    // For each token we inserted
    for(size_t i=0; i<seq_len; ++i) {
        size_t block_idx = i / config.block_size;
        size_t offset = i % config.block_size;
        int32_t phys_block = table[block_idx];
        
        size_t slot_idx = phys_block * config.block_size + offset;
        
        for(size_t h=0; h<config.n_kv_heads; ++h) {
            float stored_scale = k_scales[slot_idx * config.n_kv_heads + h];
            
            // Reconstruct float vector and basic check
            // For THIS head vector, find max
            float local_max = 0.0f;
            for(size_t d=0; d<config.head_dim; ++d) {
                float original = k_data[(i * config.n_kv_heads + h) * config.head_dim + d];
                local_max = std::max(local_max, std::abs(original));
            }
            float expected_scale = local_max / 127.0f;
            if (expected_scale < 1e-8f) expected_scale = 1e-8f;
            
            // Scale check
            CHECK(std::abs(stored_scale - expected_scale) < 1e-4f); 
            
            // Value check
            for(size_t d=0; d<config.head_dim; ++d) {
                float original = k_data[(i * config.n_kv_heads + h) * config.head_dim + d];
                int8_t q_val = k_int8[(slot_idx * config.n_kv_heads + h) * config.head_dim + d];
                
                float dequant = (float)q_val * stored_scale;
                
                // Tolerance: One bucket width
                CHECK(std::abs(dequant - original) <= stored_scale * 1.5f + 1e-4f);
            }
        }
    }
    
    if (errors > 0) {
        std::cout << "Total Errors: " << errors << std::endl;
    }
}
