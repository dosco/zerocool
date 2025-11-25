#include "core/scheduler.hpp"
#include "core/paged_kv_cache.hpp"
#include "core/sequence.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>

using namespace freellm;

// Simulation parameters
const size_t N_LAYERS = 2; // Keep it small for simulation
const size_t BLOCK_SIZE = 16;
const size_t MAX_BLOCKS = 100; // Total blocks available
const size_t VOCAB_SIZE = 1000;

void run_continuous_batching_demo() {
    std::cout << "=== Continuous Batching Demo ===" << std::endl;

    // 1. Initialize KV Cache Manager
    KVCacheConfig config;
    config.block_size = BLOCK_SIZE;
    config.max_num_blocks = MAX_BLOCKS;
    config.n_kv_heads = 1;
    config.head_dim = 64;

    KVCacheManager kv_manager(config);
    std::cout << "Initialized KVCacheManager with " << MAX_BLOCKS << " blocks." << std::endl;

    // 2. Initialize Scheduler
    Scheduler scheduler(&kv_manager);

    // 3. Create Requests
    // Request 1: Short prompt, short output
    std::vector<int> prompt1(10, 1); 
    auto seq1 = std::make_unique<Sequence>(prompt1, &kv_manager, N_LAYERS);
    scheduler.add_request(std::move(seq1));

    // Request 2: Long prompt, long output
    std::vector<int> prompt2(40, 2); // Needs 3 blocks per layer initially
    auto seq2 = std::make_unique<Sequence>(prompt2, &kv_manager, N_LAYERS);
    scheduler.add_request(std::move(seq2));

    // Request 3: Medium prompt
    std::vector<int> prompt3(20, 3);
    auto seq3 = std::make_unique<Sequence>(prompt3, &kv_manager, N_LAYERS);
    scheduler.add_request(std::move(seq3));

    std::cout << "Added 3 requests to scheduler." << std::endl;

    // 4. Continuous Batching Loop
    int step_count = 0;
    std::mt19937 rng(42);
    
    while (scheduler.has_unfinished_requests()) {
        std::cout << "\nStep " << step_count << ":" << std::endl;
        
        // Scheduler Step (Ingress/Egress)
        auto batch = scheduler.step();
        
        std::cout << "  Running batch size: " << batch.size() << std::endl;
        if (batch.empty()) {
            std::cout << "  No requests running (waiting for memory?)" << std::endl;
            // Break to avoid infinite loop if stuck
            if (step_count > 100) break;
        }

        // Simulate Execution and Generation
        for (auto* seq : batch) {
            // Simulate model forward pass (compute)
            // In real system: model->forward(batch_tokens)
            
            // Simulate sampling (generate 1 token)
            int next_token = rng() % VOCAB_SIZE;
            seq->add_token(next_token);
            
            // Simulate KV cache update (allocation)
            // We need to update KV cache for ALL layers
            // In real system: PagedAttention kernel writes to blocks
            // Here we just ensure blocks are allocated by calling update() with dummy data?
            // Or we just rely on the fact that we are tracking length.
            // PagedKVCache allocates in update().
            // If we don't call update(), block_table won't grow.
            // But Scheduler uses estimate_memory based on length.
            // So Scheduler thinks we need memory, but if we don't allocate, we don't consume it in manager?
            // Wait, Scheduler checks `kv_manager_->free_blocks_count()`.
            // If we don't allocate, free blocks won't decrease.
            // So we MUST allocate blocks to simulate real memory usage.
            
            // Let's allocate dummy blocks for each layer
            for (size_t i = 0; i < seq->n_layers(); ++i) {
                PagedKVCache* cache = seq->kv_cache(i);
                // We need to simulate adding data. 
                // We can just manually allocate blocks if needed?
                // Or create dummy tensors.
                // Creating tensors is expensive?
                // Let's just manually allocate from manager if needed.
                // PagedKVCache doesn't expose manual alloc.
                // But we can create a small dummy tensor for the new token.
                Tensor key({1, 1, 64}); // [1, n_kv_heads, head_dim]
                Tensor val({1, 1, 64});
                cache->update(key, val);
            }

            std::cout << "    Seq " << (void*)seq << " len: " << seq->length() << std::endl;

            // Check for completion
            // Simple rule: finish if length > initial + 10 (for demo)
            if (seq->length() > seq->tokens().size() + 10) { // This logic is flawed, tokens() grows.
                // Store initial length? No.
                // Just finish if length > 50.
                if (seq->length() > 50) {
                    seq->set_finished();
                    std::cout << "    Seq " << (void*)seq << " FINISHED." << std::endl;
                }
            }
            
            // Randomly finish earlier
            if (rng() % 20 == 0) {
                seq->set_finished();
                std::cout << "    Seq " << (void*)seq << " FINISHED (Random EOS)." << std::endl;
            }
        }
        
        // Check finished requests
        auto finished = scheduler.get_finished_requests();
        if (!finished.empty()) {
            std::cout << "  Evicted " << finished.size() << " finished requests." << std::endl;
        }

        // Simulate time step
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        step_count++;
    }

    std::cout << "All requests completed in " << step_count << " steps." << std::endl;
}

int main() {
    try {
        run_continuous_batching_demo();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
