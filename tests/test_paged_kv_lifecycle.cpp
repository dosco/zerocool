#include "core/sequence.hpp"
#include "core/paged_kv_cache.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace freellm;

void test_allocation_deallocation() {
    std::cout << "Testing Allocation and Deallocation..." << std::endl;

    // 1. Setup Manager
    KVCacheConfig config;
    config.block_size = 16;
    config.max_num_blocks = 100; // Small pool for testing
    config.n_kv_heads = 4;
    config.head_dim = 64;

    KVCacheManager manager(config);
    
    size_t initial_free = manager.free_blocks_count();
    assert(initial_free == 100);
    std::cout << "  Initial free blocks: " << initial_free << std::endl;

    {
        // 2. Create Sequence (Scope Start)
        std::vector<int> prompt = {1, 2, 3};
        Sequence seq(prompt, &manager);

        // Simulate prefill/generation that triggers allocation
        // We need to manually trigger cache updates since Sequence doesn't do it automatically yet
        // (In real flow, Attention layer does this)
        
        // Create dummy tensors for update
        // Shape: [seq_len, n_kv_heads, head_dim]
        // Let's add 20 tokens. Block size is 16.
        // Should allocate: ceil(20 / 16) = 2 blocks
        size_t n_tokens = 20;
        Tensor k_dummy({n_tokens, config.n_kv_heads, config.head_dim});
        Tensor v_dummy({n_tokens, config.n_kv_heads, config.head_dim});

        seq.kv_cache()->update(k_dummy, v_dummy);

        // Verify allocation
        size_t current_free = manager.free_blocks_count();
        std::cout << "  Free blocks after allocation: " << current_free << std::endl;
        assert(current_free == 98); // Used 2 blocks
        assert(seq.kv_cache()->block_table().size() == 2);

    } // 3. Sequence Destroyed (Scope End)

    // 4. Verify Deallocation
    size_t final_free = manager.free_blocks_count();
    std::cout << "  Free blocks after destruction: " << final_free << std::endl;
    assert(final_free == 100); // All blocks returned

    std::cout << "PASS: Allocation and Deallocation" << std::endl;
}

void test_reuse() {
    std::cout << "\nTesting Reuse..." << std::endl;

    KVCacheConfig config;
    config.block_size = 16;
    config.max_num_blocks = 2; // Very small pool
    config.n_kv_heads = 4;
    config.head_dim = 64;

    KVCacheManager manager(config);

    // Sequence A uses all memory
    {
        Sequence seqA({1}, &manager);
        // Add 32 tokens -> 2 blocks (full pool)
        Tensor k({32, 4, 64});
        Tensor v({32, 4, 64});
        seqA.kv_cache()->update(k, v);
        
        assert(manager.free_blocks_count() == 0);
        std::cout << "  Sequence A used all blocks." << std::endl;
    }
    // seqA destroyed, blocks freed

    assert(manager.free_blocks_count() == 2);

    // Sequence B reuses memory
    {
        Sequence seqB({1}, &manager);
        // Add 16 tokens -> 1 block
        Tensor k({16, 4, 64});
        Tensor v({16, 4, 64});
        seqB.kv_cache()->update(k, v);

        assert(manager.free_blocks_count() == 1);
        std::cout << "  Sequence B successfully reused a block." << std::endl;
    }

    std::cout << "PASS: Reuse" << std::endl;
}

int main() {
    try {
        test_allocation_deallocation();
        test_reuse();
        std::cout << "\nAll tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
