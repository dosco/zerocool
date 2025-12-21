#pragma once

#include "core/scheduler.hpp"
#include "core/paged_kv_cache.hpp"
#include "infra/compute_backend.hpp"
#include "core/model_config.hpp"
#include "core/scheduler.hpp"
#include "core/paged_kv_cache.hpp"
#include "infra/compute_backend.hpp"
#include "core/model_config.hpp"
// #include "core/tokenizer.hpp" // Decoupled
#include "core/tensor.hpp"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>

namespace freellm {

class GenerationEngine {
public:
    GenerationEngine(const std::string& model_path, 
                     // const std::string& tokenizer_path, // Decoupled
                     const ModelConfig& config,
                     const KVCacheConfig& kv_config,
                     int device_id = 0);

    ~GenerationEngine();

    // Add a new prompt request
    // Add a new prompt request (tokens)
    void add_request(const std::vector<int>& tokens);

    // Run one step of inference (prefill and/or decode)
    // Returns true if there are still active requests
    bool step();

    // Get active sequences (for monitoring)
    const std::vector<Sequence*>& active_sequences() const;

private:
    // Helper to get buffer from weights map
    infra::DeviceBuffer* get_buffer(const std::map<std::string, std::unique_ptr<infra::DeviceBuffer>>& weights, const std::string& name);

    // Initialization
    void load_weights(const std::string& model_path);
    void initialize_buffers();

    // Kernels
    void run_linear_batched(infra::DeviceBuffer* input, const std::string& weight_name, infra::DeviceBuffer* output, 
                           int in_features, int out_features, int batch_size);

    std::unique_ptr<infra::ComputeBackend> backend_;
    std::unique_ptr<KVCacheManager> kv_manager_;
    std::unique_ptr<Scheduler> scheduler_;
    
    ModelConfig config_;
    KVCacheConfig kv_config_;
    // Tokenizer tokenizer_; // Decoupled

    // Weights
    std::map<std::string, std::unique_ptr<infra::DeviceBuffer>> gpu_weights_;
    std::set<std::string> fp32_weight_names_;  // Track which weights are FP32 (lm_head, norms)
    Tensor token_embedding_cpu_;

    // Persistent Buffers (avoid reallocation per step)
    std::unique_ptr<infra::DeviceBuffer> x_;
    std::unique_ptr<infra::DeviceBuffer> residual_;
    std::unique_ptr<infra::DeviceBuffer> x_norm_;
    std::unique_ptr<infra::DeviceBuffer> q_;
    std::unique_ptr<infra::DeviceBuffer> k_cur_;
    std::unique_ptr<infra::DeviceBuffer> v_cur_;
    std::unique_ptr<infra::DeviceBuffer> ffn_gate_;
    std::unique_ptr<infra::DeviceBuffer> ffn_up_;
    std::unique_ptr<infra::DeviceBuffer> ffn_down_;
    std::unique_ptr<infra::DeviceBuffer> logits_; // [max_tokens, vocab_size]
    
    // Scalar buffers for kernel params (Metal constant fix)
    std::unique_ptr<infra::DeviceBuffer> scalar_head_dim_;
    std::unique_ptr<infra::DeviceBuffer> scalar_n_heads_;
    std::unique_ptr<infra::DeviceBuffer> scalar_n_kv_heads_;
    std::unique_ptr<infra::DeviceBuffer> scalar_vocab_size_;
    std::unique_ptr<infra::DeviceBuffer> scalar_feature_constants_; // Reuse for common
    std::unique_ptr<infra::DeviceBuffer> scalar_epsilon_;
    std::unique_ptr<infra::DeviceBuffer> scalar_block_size_;
    std::unique_ptr<infra::DeviceBuffer> scalar_max_num_blocks_;
    std::unique_ptr<infra::DeviceBuffer> scalar_scale_attn_;

    // Metadata Buffers
    std::unique_ptr<infra::DeviceBuffer> block_table_buf_;
    std::unique_ptr<infra::DeviceBuffer> cu_seqlens_buf_;
    std::unique_ptr<infra::DeviceBuffer> positions_buf_;
    std::unique_ptr<infra::DeviceBuffer> cos_buf_;
    std::unique_ptr<infra::DeviceBuffer> sin_buf_;

    // State
    std::map<Sequence*, int> processed_lengths_;
    std::vector<Sequence*> active_sequences_;
};

} // namespace freellm
