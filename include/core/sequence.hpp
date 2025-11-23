#pragma once

#include "core/paged_kv_cache.hpp"
#include <vector>
#include <memory>
#include <string>

namespace freellm {

/**
 * @brief Represents a single sequence being generated
 * 
 * This class manages the lifecycle of a sequence, including:
 * - The tokens generated so far
 * - The Paged KV Cache associated with this sequence
 * - Automatic resource management (RAII)
 */
class Sequence {
public:
    /**
     * @brief Create a new sequence
     * 
     * @param prompt_tokens Initial prompt tokens
     * @param kv_manager Global KV cache manager
     */
    Sequence(const std::vector<int>& prompt_tokens, KVCacheManager* kv_manager)
        : tokens_(prompt_tokens) {
        // Allocate a new Paged KV Cache for this sequence
        // The PagedKVCache constructor doesn't allocate blocks yet;
        // blocks are allocated on demand during update()
        kv_cache_ = std::make_unique<PagedKVCache>(kv_manager);
    }

    // Disable copying to prevent double-free or shared ownership confusion
    Sequence(const Sequence&) = delete;
    Sequence& operator=(const Sequence&) = delete;

    // Allow moving
    Sequence(Sequence&&) = default;
    Sequence& operator=(Sequence&&) = default;

    ~Sequence() {
        // kv_cache_ destructor will automatically call reset()
        // and free all allocated blocks back to the manager
    }

    /**
     * @brief Add a new token to the sequence
     * 
     * @param token_id The token ID to append
     */
    void add_token(int token_id) {
        tokens_.push_back(token_id);
    }

    /**
     * @brief Get all tokens in the sequence
     */
    const std::vector<int>& tokens() const {
        return tokens_;
    }

    /**
     * @brief Get the Paged KV Cache for this sequence
     */
    PagedKVCache* kv_cache() {
        return kv_cache_.get();
    }

    /**
     * @brief Get current sequence length
     */
    size_t length() const {
        return tokens_.size();
    }

private:
    std::vector<int> tokens_;
    std::unique_ptr<PagedKVCache> kv_cache_;
};

} // namespace freellm
