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
     * @param n_layers Number of transformer layers
     */
    Sequence(const std::vector<int>& prompt_tokens, KVCacheManager* kv_manager, size_t n_layers)
        : tokens_(prompt_tokens) {
        // Allocate Paged KV Caches for each layer
        kv_caches_.reserve(n_layers);
        for (size_t i = 0; i < n_layers; ++i) {
            kv_caches_.push_back(std::make_unique<PagedKVCache>(kv_manager));
        }
    }

    // Disable copying to prevent double-free or shared ownership confusion
    Sequence(const Sequence&) = delete;
    Sequence& operator=(const Sequence&) = delete;

    // Allow moving
    Sequence(Sequence&&) = default;
    Sequence& operator=(Sequence&&) = default;

    ~Sequence() {
        // kv_caches_ destructors will automatically call reset()
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
     * @brief Get the Paged KV Cache for a specific layer
     */
    PagedKVCache* kv_cache(size_t layer_idx) {
        if (layer_idx >= kv_caches_.size()) {
            throw std::out_of_range("Layer index out of bounds");
        }
        return kv_caches_[layer_idx].get();
    }

    /**
     * @brief Get current sequence length
     */
    size_t length() const {
        return tokens_.size();
    }

    /**
     * @brief Get number of layers
     */
    size_t n_layers() const {
        return kv_caches_.size();
    }

    /**
     * @brief Mark the sequence as finished
     */
    void set_finished() {
        finished_ = true;
    }

    /**
     * @brief Check if the sequence is finished
     */
    bool is_finished() const {
        return finished_;
    }

private:
    std::vector<int> tokens_;
    std::vector<std::unique_ptr<PagedKVCache>> kv_caches_;
    bool finished_ = false;
};

} // namespace freellm
