#include "core/paged_kv_cache.hpp"
#include "kernels/attention/paged_attention.hpp"
#include "kernels/attention/scaled_dot_product.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

using namespace freellm;

void test_manager_allocation() {
    std::cout << "Testing KVCacheManager allocation..." << std::endl;
    
    KVCacheConfig config;
    config.block_size = 16;
    config.max_num_blocks = 10;
    config.n_kv_heads = 2;
    config.head_dim = 64;

    KVCacheManager manager(config);

    assert(manager.free_blocks_count() == 10);

    int32_t b1 = manager.allocate_block();
    assert(manager.free_blocks_count() == 9);
    
    int32_t b2 = manager.allocate_block();
    assert(manager.free_blocks_count() == 8);

    assert(b1 != b2);

    manager.free_block(b1);
    assert(manager.free_blocks_count() == 9);

    manager.free_block(b2);
    assert(manager.free_blocks_count() == 10);

    std::cout << "PASS" << std::endl;
}

void test_paged_cache_update() {
    std::cout << "Testing PagedKVCache update..." << std::endl;

    KVCacheConfig config;
    config.block_size = 4; // Small block size for testing
    config.max_num_blocks = 10;
    config.n_kv_heads = 1;
    config.head_dim = 8;

    KVCacheManager manager(config);
    PagedKVCache cache(&manager);

    // Create dummy data: [seq_len=6, n_kv_heads=1, head_dim=8]
    // Should span 2 blocks (4 + 2)
    Tensor keys({6, 1, 8});
    Tensor values({6, 1, 8});

    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            keys.at({i, 0, j}) = static_cast<float>(i * 10 + j);
            values.at({i, 0, j}) = static_cast<float>(i * 10 + j + 100);
        }
    }

    cache.update(keys, values);

    assert(cache.current_length() == 6);
    assert(cache.block_table().size() == 2);
    assert(manager.free_blocks_count() == 8);

    // Verify data in global memory
    int32_t block0 = cache.block_table()[0];
    int32_t block1 = cache.block_table()[1];

    // Check first block (indices 0-3)
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            float expected_k = static_cast<float>(i * 10 + j);
            float actual_k = manager.global_keys().at({static_cast<size_t>(block0), i, 0, j});
            assert(std::abs(actual_k - expected_k) < 1e-5);
        }
    }

    // Check second block (indices 4-5)
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            float expected_k = static_cast<float>((i + 4) * 10 + j);
            float actual_k = manager.global_keys().at({static_cast<size_t>(block1), i, 0, j});
            assert(std::abs(actual_k - expected_k) < 1e-5);
        }
    }

    std::cout << "PASS" << std::endl;
}

void test_paged_attention_correctness() {
    std::cout << "Testing paged_attention correctness..." << std::endl;

    // Setup: 1 head, head_dim=8, seq_len=6
    size_t n_heads = 1;
    size_t n_kv_heads = 1;
    size_t head_dim = 8;
    size_t seq_len = 6;

    KVCacheConfig config;
    config.block_size = 4;
    config.max_num_blocks = 10;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;

    KVCacheManager manager(config);
    PagedKVCache cache(&manager);

    // Create K, V
    Tensor K({seq_len, n_kv_heads, head_dim});
    Tensor V({seq_len, n_kv_heads, head_dim});
    for (size_t i = 0; i < K.size(); ++i) {
        K.data()[i] = static_cast<float>(i) * 0.1f;
        V.data()[i] = static_cast<float>(i) * 0.2f;
    }

    cache.update(K, V);

    // Create Q [1, n_heads, head_dim] (querying for the next token, or just testing attention)
    // Let's say we are querying for position 5 (last one)
    Tensor Q({1, n_heads, head_dim}, 1.0f); 

    // Reference implementation (standard attention)
    // We need to reshape K/V to match standard attention expectation [seq_len, n_kv_heads, head_dim]
    // (They are already in that shape)
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Standard attention
    // Note: standard attention expects Q [seq_len_q, n_heads, head_dim]
    // K, V [seq_len_k, n_kv_heads, head_dim]
    Tensor expected_output = scaled_dot_product_attention(Q, K, V, scale, n_kv_heads, 0);

    // Paged attention
    Tensor actual_output = paged_scaled_dot_product_attention(Q, cache, scale, n_kv_heads, 0);

    // Compare
    assert(expected_output.shape() == actual_output.shape());
    for (size_t i = 0; i < expected_output.size(); ++i) {
        float diff = std::abs(expected_output.data()[i] - actual_output.data()[i]);
        if (diff > 1e-4) {
            std::cerr << "Mismatch at " << i << ": expected " << expected_output.data()[i] 
                      << ", got " << actual_output.data()[i] << std::endl;
            assert(false);
        }
    }

    std::cout << "PASS" << std::endl;
}

int main() {
    test_manager_allocation();
    test_paged_cache_update();
    test_paged_attention_correctness();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
