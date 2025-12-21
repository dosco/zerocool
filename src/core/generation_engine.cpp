#include "core/generation_engine.hpp"
#include "infra/metal_backend.hpp"
#include "infra/safetensors_loader.hpp"
#include "kernels/quantiz/q4_0/quantize.hpp" // For BlockQ4_0
#include "core/sampling.hpp"
#include <iostream>
#include <print>
#include <cmath>

namespace freellm {



GenerationEngine::GenerationEngine(const std::string& model_path, 
                                   // const std::string& tokenizer_path, 
                                   const ModelConfig& config,
                                   const KVCacheConfig& kv_config,
                                   int device_id)
    : config_(config), kv_config_(kv_config) {
    
    // 1. Initialize Backend
    backend_ = std::make_unique<infra::MetalBackend>(device_id);
    
    // 2. Initialize KV Cache Manager
    kv_manager_ = std::make_unique<KVCacheManager>(kv_config, backend_.get());
    
    // 3. Initialize Scheduler
    scheduler_ = std::make_unique<Scheduler>(kv_manager_.get());
    
    // 4. Load Weights
    load_weights(model_path);
    
    // 5. Initialize Buffers
    initialize_buffers();
    
    // Precompute RoPE
    size_t head_dim = config.d_model / config.n_heads;
    size_t half_head_dim = head_dim / 2;
    std::vector<float> cos_table(config.max_seq_len * half_head_dim);
    std::vector<float> sin_table(config.max_seq_len * half_head_dim);

    for (size_t pos = 0; pos < config.max_seq_len; ++pos) {
        for (size_t i = 0; i < half_head_dim; ++i) {
            double theta = 1.0 / std::pow((double)config_.rope_theta, 2.0 * i / head_dim);
            float angle = pos * theta;
            cos_table[pos * half_head_dim + i] = std::cos(angle);
            sin_table[pos * half_head_dim + i] = std::sin(angle);
        }
    }
    
    cos_buf_ = backend_->allocate(cos_table.size() * sizeof(float), infra::DType::FLOAT32);
    sin_buf_ = backend_->allocate(sin_table.size() * sizeof(float), infra::DType::FLOAT32);
    backend_->copy_to_device(cos_buf_.get(), cos_table.data(), cos_table.size() * sizeof(float));
    backend_->copy_to_device(sin_buf_.get(), sin_table.data(), sin_table.size() * sizeof(float));
}

GenerationEngine::~GenerationEngine() {}

void GenerationEngine::add_request(const std::vector<int>& tokens) {
    // std::vector<int> tokens = tokenizer_.encode(prompt); // Decoupled
    scheduler_->add_request(std::make_unique<Sequence>(tokens, kv_manager_.get(), config_.n_layers));
}

const std::vector<Sequence*>& GenerationEngine::active_sequences() const {
    return active_sequences_;
}

infra::DeviceBuffer* GenerationEngine::get_buffer(const std::map<std::string, std::unique_ptr<infra::DeviceBuffer>>& weights, const std::string& name) {
    auto it = weights.find(name);
    if (it == weights.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second.get();
}

void GenerationEngine::load_weights(const std::string& model_path) {
    // Quantization resolver (default: all Q4_0 except token_embedding/norm/lm_head)
    auto quant_resolver = [](const std::string& name) -> std::optional<freellm::quant::QuantType> {
         // Keep embeddings and norms in FP32 for accuracy
         if (name == "token_embedding" || name.find("norm") != std::string::npos || name == "lm_head") {
             return std::nullopt; // Keep FP32
         }
         return freellm::quant::QuantType::Q4_0;
    };

    auto [fp32_tensors, quant_tensors] = freellm::load_model_from_path(model_path, quant_resolver);
    
    // Extract embedding (keep on CPU)
    if (fp32_tensors.find("token_embedding") != fp32_tensors.end()) {
        token_embedding_cpu_ = std::move(fp32_tensors.at("token_embedding"));
        fp32_tensors.erase("token_embedding");
    } else {
        throw std::runtime_error("token_embedding not found in model weights");
    }

    // Upload FP32 tensors
    for (const auto& [name, tensor] : fp32_tensors) {
        // Skip tensors that are also in quant_tensors.
        // ParallelTensorLoader returns ALL tensors in fp32_tensors, even if they were quantized.
        if (quant_tensors.find(name) != quant_tensors.end()) {
            continue;
        }

        size_t bytes = tensor.size() * sizeof(float);
        auto dev_buf = backend_->allocate(bytes, infra::DType::FLOAT32);
        backend_->copy_to_device(dev_buf.get(), tensor.data(), bytes);
        gpu_weights_[name] = std::move(dev_buf);
        fp32_weight_names_.insert(name);  // Track as FP32
    }
    
    // Upload Quantized tensors
    for (auto& [name, q_tensor] : quant_tensors) {
        size_t bytes = q_tensor.data_size();
        auto dev_buf = backend_->allocate(bytes, infra::DType::INT8);
        backend_->copy_to_device(dev_buf.get(), q_tensor.data(), bytes);
        gpu_weights_[name] = std::move(dev_buf);
    }
}

void GenerationEngine::initialize_buffers() {
    size_t batch_size = 32; // Default max batch size for allocation (can resize if needed, but for now fixed)
    // Actually we only need buffers for CURRENT batch execution, which is limited by VRAM or scheduler batch limit.
    // Let's assume max 1024 tokens processed at once.
    size_t alloc_tokens = 4096;
    
    size_t head_dim = config_.d_model / config_.n_heads;
    
    x_ = backend_->allocate(alloc_tokens * config_.d_model * sizeof(float), infra::DType::FLOAT32);
    residual_ = backend_->allocate(alloc_tokens * config_.d_model * sizeof(float), infra::DType::FLOAT32);
    x_norm_ = backend_->allocate(alloc_tokens * config_.d_model * sizeof(float), infra::DType::FLOAT32);
    
    q_ = backend_->allocate(alloc_tokens * config_.d_model * sizeof(float), infra::DType::FLOAT32);
    k_cur_ = backend_->allocate(alloc_tokens * config_.n_kv_heads * head_dim * sizeof(float), infra::DType::FLOAT32);
    v_cur_ = backend_->allocate(alloc_tokens * config_.n_kv_heads * head_dim * sizeof(float), infra::DType::FLOAT32);
    
    ffn_gate_ = backend_->allocate(alloc_tokens * config_.d_ff * sizeof(float), infra::DType::FLOAT32);
    ffn_up_ = backend_->allocate(alloc_tokens * config_.d_ff * sizeof(float), infra::DType::FLOAT32);
    ffn_down_ = backend_->allocate(alloc_tokens * config_.d_model * sizeof(float), infra::DType::FLOAT32);
    
    logits_ = backend_->allocate(alloc_tokens * config_.vocab_size * sizeof(float), infra::DType::FLOAT32);
    
    block_table_buf_ = backend_->allocate(batch_size * kv_config_.max_num_blocks * sizeof(int32_t), infra::DType::INT32);
    cu_seqlens_buf_ = backend_->allocate((batch_size + 1) * sizeof(int32_t), infra::DType::INT32);
    positions_buf_ = backend_->allocate(alloc_tokens * sizeof(int32_t), infra::DType::INT32);

    // Initialize Scalar Constant Buffers
    auto alloc_u32 = [&](uint32_t val) {
        auto buf = backend_->allocate(sizeof(uint32_t), infra::DType::INT32);
        backend_->copy_to_device(buf.get(), 0, &val, sizeof(uint32_t));
        return buf;
    };
    auto alloc_f32 = [&](float val) {
        auto buf = backend_->allocate(sizeof(float), infra::DType::FLOAT32);
        backend_->copy_to_device(buf.get(), 0, &val, sizeof(float));
        return buf;
    };

    scalar_head_dim_ = alloc_u32((uint32_t)head_dim);
    scalar_n_heads_ = alloc_u32((uint32_t)config_.n_heads);
    scalar_n_kv_heads_ = alloc_u32((uint32_t)config_.n_kv_heads);
    scalar_vocab_size_ = alloc_u32((uint32_t)config_.vocab_size);
    scalar_epsilon_ = alloc_f32(1e-5f);
    scalar_block_size_ = alloc_u32((uint32_t)kv_config_.block_size);
    scalar_max_num_blocks_ = alloc_u32((uint32_t)kv_config_.max_num_blocks);
    
    float scale = 1.0f / std::sqrt((float)head_dim);
    scalar_scale_attn_ = alloc_f32(scale);
}

void GenerationEngine::run_linear_batched(infra::DeviceBuffer* input, const std::string& weight_name, infra::DeviceBuffer* output, 
                       int in_features, int out_features, int batch_size) {
    // Scalar Allocator Helper (todo: dedup)
    auto alloc_u32 = [&](uint32_t val) {
        auto buf = backend_->allocate(sizeof(uint32_t), infra::DType::INT32);
        backend_->copy_to_device(buf.get(), 0, &val, sizeof(uint32_t));
        return buf;
    };

    auto w_buf = get_buffer(gpu_weights_, weight_name);
    
    // Metal dispatchThreads needs total threads, not threadgroups
    // We want out_features threadgroups, each with 32 threads
    // So grid = (out_features * 32, batch_size, 1)
    infra::KernelConfig grid; 
    grid.grid = infra::Dim3(out_features * 32, batch_size, 1); 
    grid.block = infra::Dim3(32, 1, 1);
    
    // Params as buffers
    // Kernel: A(0), B(1), C(2), cols(3), batch(4)
    auto b_cols = alloc_u32((uint32_t)in_features);
    auto b_batch = alloc_u32((uint32_t)batch_size);
    
    // Check if weight is FP32 or quantized
    bool is_fp32 = fp32_weight_names_.count(weight_name) > 0;
    
    if (is_fp32) {
        // Use FP32 GEMM kernel for unquantized weights (lm_head, etc.)
        backend_->execute_kernel("gemm_f32", 
            {w_buf, input, output, b_cols.get(), b_batch.get()}, 
            {}, 
            grid
        );
    } else {
        // Use Q4_0 GEMM kernel for quantized weights
        backend_->execute_kernel("gemm_q4_0", 
            {w_buf, input, output, b_cols.get(), b_batch.get()}, 
            {}, 
            grid
        );
    }
}

bool GenerationEngine::step() {
    auto sequences = scheduler_->step();
    active_sequences_ = sequences;
    if (sequences.empty()) return false;

    // Helper for dynamic scalars to safe buffers
    auto alloc_u32 = [&](uint32_t val) {
        auto buf = backend_->allocate(sizeof(uint32_t), infra::DType::INT32);
        backend_->copy_to_device(buf.get(), 0, &val, sizeof(uint32_t));
        return buf;
    };

    // Separate Prefill and Decode
    std::vector<Sequence*> prefills;
    std::vector<Sequence*> decodes;
    for (auto* seq : sequences) {
        // If not processed yet (or length changed dramatically?), assume prefill.
        // Or track state? processed_lengths_ is used.
        if (processed_lengths_.find(seq) == processed_lengths_.end()) {
            prefills.push_back(seq);
        } else {
            decodes.push_back(seq);
        }
    }
    
    size_t head_dim = config_.d_model / config_.n_heads;
    size_t d_model_bytes = config_.d_model * sizeof(float);

    // 1. Run Prefills (Batched)
    if (!prefills.empty()) {
        int total_prompt_len = 0;
        std::vector<int> cu_seqlens = {0};
        std::vector<int> flat_tokens;
        
        for (auto* seq : prefills) {
            int len = seq->length();
            total_prompt_len += len;
            cu_seqlens.push_back(total_prompt_len);
            const auto& tokens = seq->tokens();
            flat_tokens.insert(flat_tokens.end(), tokens.begin(), tokens.end());
        }

        // Copy Embeddings
        std::vector<float> flat_embeds(total_prompt_len * config_.d_model);
        for (int i = 0; i < total_prompt_len; ++i) {
            float* src = token_embedding_cpu_.data() + flat_tokens[i] * config_.d_model;
            std::memcpy(flat_embeds.data() + i * config_.d_model, src, d_model_bytes);
        }
        backend_->copy_to_device(x_.get(), flat_embeds.data(), flat_embeds.size() * sizeof(float));
        
        // Metadata
        backend_->copy_to_device(cu_seqlens_buf_.get(), cu_seqlens.data(), cu_seqlens.size() * sizeof(int32_t));

        // Layers
        for (size_t l = 0; l < config_.n_layers; ++l) {
            std::string layer_prefix = "layers." + std::to_string(l) + ".";

            backend_->copy_to_device(residual_.get(), x_->raw_ptr(), total_prompt_len * d_model_bytes);

            // RMSNorm
            auto b_dim = alloc_u32((uint32_t)config_.d_model);
            infra::DeviceBuffer* w_norm = get_buffer(gpu_weights_, layer_prefix + "attn_norm_weight");
            infra::KernelConfig grid_norm; 
            grid_norm.grid = infra::Dim3(total_prompt_len, 1, 1); 
            grid_norm.block = infra::Dim3(std::min((int)total_prompt_len, 256), 1, 1);
            
            backend_->execute_kernel("rms_norm", 
                {x_.get(), w_norm, x_norm_.get(), b_dim.get(), scalar_epsilon_.get()}, 
                {}, grid_norm);

            // QKV
            run_linear_batched(x_norm_.get(), layer_prefix + "attn.W_q", q_.get(), config_.d_model, config_.d_model, total_prompt_len);
            run_linear_batched(x_norm_.get(), layer_prefix + "attn.W_k", k_cur_.get(), config_.d_model, config_.n_kv_heads * head_dim, total_prompt_len);
            run_linear_batched(x_norm_.get(), layer_prefix + "attn.W_v", v_cur_.get(), config_.d_model, config_.n_kv_heads * head_dim, total_prompt_len);

            // RoPE Ragged
            std::vector<int32_t> all_positions(total_prompt_len);
            for (size_t s = 0; s < prefills.size(); ++s) {
                for (int t = 0; t < (cu_seqlens[s+1] - cu_seqlens[s]); ++t) {
                    all_positions[cu_seqlens[s] + t] = t;
                }
            }
            backend_->copy_to_device(positions_buf_.get(), all_positions.data(), all_positions.size() * sizeof(int32_t));

            infra::KernelConfig grid_rope; 
            grid_rope.grid = infra::Dim3(head_dim / 2, config_.n_heads, total_prompt_len); 
            grid_rope.block = infra::Dim3(32, 1, 1);
            
            // Inputs: q, cos, sin, pos, out, head_dim, n_heads
            backend_->execute_kernel("rope_ragged", 
                {q_.get(), cos_buf_.get(), sin_buf_.get(), positions_buf_.get(), q_.get(), scalar_head_dim_.get(), scalar_n_heads_.get()}, 
                {}, grid_rope);
            
            infra::KernelConfig grid_rope_k; 
            grid_rope_k.grid = infra::Dim3(head_dim / 2, config_.n_kv_heads, total_prompt_len); 
            grid_rope_k.block = infra::Dim3(32, 1, 1);
            
            backend_->execute_kernel("rope_ragged", 
                {k_cur_.get(), cos_buf_.get(), sin_buf_.get(), positions_buf_.get(), k_cur_.get(), scalar_head_dim_.get(), scalar_n_kv_heads_.get()}, 
                {}, grid_rope_k);

            // KV Cache Update
            for (size_t s = 0; s < prefills.size(); ++s) {
                 // Use offset-aware updates manually or updated method
                 // Assuming PagedKVCache::update supports src_base_offset
                 size_t len = cu_seqlens[s+1] - cu_seqlens[s];
                 size_t offset_bytes = cu_seqlens[s] * config_.n_kv_heads * head_dim * sizeof(float);
                 prefills[s]->kv_cache(l)->update(k_cur_.get(), v_cur_.get(), len, offset_bytes);
            }

            // Attention
            std::vector<int32_t> strided_tables(prefills.size() * kv_config_.max_num_blocks);
            for (size_t s = 0; s < prefills.size(); ++s) {
                const auto& bt = prefills[s]->kv_cache(l)->block_table();
                std::memcpy(strided_tables.data() + s * kv_config_.max_num_blocks, bt.data(), bt.size() * sizeof(int32_t));
            }
            backend_->copy_to_device(block_table_buf_.get(), strided_tables.data(), strided_tables.size() * sizeof(int32_t));

            infra::KernelConfig grid_attn; 
            // MetalBackend uses dispatchThreads. We need n_heads groups.
            // Block size must cover head_dim (usually 64 or 128).
            int attn_block_size = (head_dim + 31) / 32 * 32; // Round up to 32
            grid_attn.grid = infra::Dim3(config_.n_heads * attn_block_size, total_prompt_len, 1); 
            grid_attn.block = infra::Dim3(attn_block_size, 1, 1);
            
            auto b_n_seqs = alloc_u32((uint32_t)prefills.size());
            // Inputs: q, k, v, block_table, cu_seqlens, output, 
            // Params: n_seqs, n_heads, n_kv, hd, scale, bs, max_num
            if (kv_config_.data_type == KvCacheDataType::INT8) {
                 backend_->execute_kernel("gqa_attention_prefill_int8", 
                     {q_.get(), kv_manager_->global_keys_int8_device(), kv_manager_->global_values_int8_device(), 
                      kv_manager_->global_k_scales_device(), kv_manager_->global_v_scales_device(),
                      block_table_buf_.get(), cu_seqlens_buf_.get(), x_norm_.get(),
                      b_n_seqs.get(), scalar_n_heads_.get(), scalar_n_kv_heads_.get(), scalar_head_dim_.get(), scalar_scale_attn_.get(), scalar_block_size_.get(), scalar_max_num_blocks_.get()}, 
                     {}, grid_attn);
            } else {
                 backend_->execute_kernel("gqa_attention_prefill_ragged", 
                     {q_.get(), kv_manager_->global_keys_device(), kv_manager_->global_values_device(), block_table_buf_.get(), cu_seqlens_buf_.get(), x_norm_.get(),
                      b_n_seqs.get(), scalar_n_heads_.get(), scalar_n_kv_heads_.get(), scalar_head_dim_.get(), scalar_scale_attn_.get(), scalar_block_size_.get(), scalar_max_num_blocks_.get()}, 
                     {}, grid_attn
                 );
            }

            // FFN
            run_linear_batched(x_norm_.get(), layer_prefix + "attn.W_o", q_.get(), config_.d_model, config_.d_model, total_prompt_len);
            
            infra::KernelConfig grid_add; grid_add.grid = infra::Dim3(total_prompt_len * config_.d_model, 1, 1); grid_add.block = infra::Dim3(256, 1, 1);
            backend_->execute_kernel("add", {residual_.get(), q_.get()}, {x_.get()}, grid_add);

            backend_->copy_to_device(residual_.get(), x_->raw_ptr(), total_prompt_len * d_model_bytes);
            
            infra::DeviceBuffer* w_norm_ffn = get_buffer(gpu_weights_, layer_prefix + "ffn_norm_weight");
            auto b_dim_ffn = alloc_u32((uint32_t)config_.d_model);
            backend_->execute_kernel("rms_norm", {x_.get(), w_norm_ffn, x_norm_.get(), b_dim_ffn.get(), scalar_epsilon_.get()}, {}, grid_norm);

            run_linear_batched(x_norm_.get(), layer_prefix + "ffn.W_gate", ffn_gate_.get(), config_.d_model, config_.d_ff, total_prompt_len);
            run_linear_batched(x_norm_.get(), layer_prefix + "ffn.W_up", ffn_up_.get(), config_.d_model, config_.d_ff, total_prompt_len);
            
            infra::KernelConfig grid_ffn; grid_ffn.grid = infra::Dim3(total_prompt_len * config_.d_ff, 1, 1); grid_ffn.block = infra::Dim3(256, 1, 1);
            backend_->execute_kernel("silu", {ffn_gate_.get()}, {ffn_gate_.get()}, grid_ffn);
            backend_->execute_kernel("mul", {ffn_gate_.get(), ffn_up_.get()}, {ffn_gate_.get()}, grid_ffn);
            
            run_linear_batched(ffn_gate_.get(), layer_prefix + "ffn.W_down", ffn_down_.get(), config_.d_ff, config_.d_model, total_prompt_len);
            backend_->execute_kernel("add", {x_.get(), ffn_down_.get()}, {x_.get()}, grid_add);
        }
        
        // Sampling
        infra::DeviceBuffer* w_norm_final = get_buffer(gpu_weights_, "final_norm_weight");
        infra::KernelConfig grid_norm; grid_norm.grid = infra::Dim3(total_prompt_len, 1, 1); grid_norm.block = infra::Dim3(std::min((int)total_prompt_len, 256), 1, 1);
        auto b_dim_final = alloc_u32((uint32_t)config_.d_model);
        backend_->execute_kernel("rms_norm", {x_.get(), w_norm_final, x_norm_.get(), b_dim_final.get(), scalar_epsilon_.get()}, {}, grid_norm);
        
        run_linear_batched(x_norm_.get(), "lm_head", logits_.get(), config_.d_model, config_.vocab_size, total_prompt_len);
        
        backend_->synchronize();
        std::vector<float> all_logits(total_prompt_len * config_.vocab_size);
        backend_->copy_to_host(all_logits.data(), logits_.get(), all_logits.size() * sizeof(float));
        
        for (size_t s = 0; s < prefills.size(); ++s) {
            int last_idx = cu_seqlens[s+1] - 1;
            Tensor logits_tensor({(size_t)config_.vocab_size}, all_logits.data() + last_idx * config_.vocab_size);
            int next_token = freellm::top_p_sample(logits_tensor, 0.9f, 0.7f);
            
            // DEBUG: Print top logits
            {
                std::vector<float> l(logits_tensor.data(), logits_tensor.data() + config_.vocab_size);
                std::vector<std::pair<float, int>> pairs;
                for(size_t i=0; i<l.size(); ++i) pairs.push_back({l[i], (int)i});
                std::sort(pairs.rbegin(), pairs.rend());
                
                std::cout << "\n[DEBUG] Top 5 logits for seq " << s << ":\n";
                for(int i=0; i<5; ++i) {
                     std::cout << "  " << pairs[i].second << ": " << pairs[i].first << "\n";
                }
                std::cout << "  Reserved(128255): " << l[128255] << "\n";
                std::cout << "  Chosen: " << next_token << "\n";
            }

            prefills[s]->add_token(next_token);
            processed_lengths_[prefills[s]] = prefills[s]->length();
            
            // Check EOS or Max Length
            // Llama 3 uses: 128001 (<|end_of_text|>), 128008 (<|eom_id|>), 128009 (<|eot_id|>)
            // Also appearing: 128255 (<|reserved_special_token_250|>) acting as EOS
            if (next_token == 128001 || next_token == 128008 || next_token == 128009 || next_token == 128255 ||
                prefills[s]->length() >= config_.max_seq_len) {
                prefills[s]->set_finished();
            }
        }
    }

    // 2. Run Decodes (Batched)
    if (!decodes.empty()) {
        int batch_size = decodes.size();
        std::vector<float> last_token_embeds(batch_size * config_.d_model);
        std::vector<int> context_lens(batch_size);
        std::vector<int32_t> positions(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            auto* seq = decodes[i];
            int token = seq->tokens().back();
            float* src = token_embedding_cpu_.data() + token * config_.d_model;
            std::memcpy(last_token_embeds.data() + i * config_.d_model, src, d_model_bytes);
            context_lens[i] = seq->length(); // length includes new token? No, new token is being processed. 
            // Sequence length includes all tokens added via add_token.
            // When we decode, we are processing the token added in previous step.
            // Pos = length - 1.
            positions[i] = seq->length() - 1; 
        }
        backend_->copy_to_device(x_.get(), last_token_embeds.data(), last_token_embeds.size() * sizeof(float));
        backend_->copy_to_device(positions_buf_.get(), positions.data(), positions.size() * sizeof(int32_t));
        
        // Layers
        for (size_t l = 0; l < config_.n_layers; ++l) {
            std::string layer_prefix = "layers." + std::to_string(l) + ".";
            backend_->copy_to_device(residual_.get(), x_->raw_ptr(), batch_size * d_model_bytes);

            infra::DeviceBuffer* w_norm = get_buffer(gpu_weights_, layer_prefix + "attn_norm_weight");
            infra::KernelConfig grid_norm; grid_norm.grid = infra::Dim3(batch_size, 1, 1); grid_norm.block = infra::Dim3(std::min((int)batch_size, 256), 1, 1);
            
            auto b_dim_dec = alloc_u32((uint32_t)config_.d_model);
            backend_->execute_kernel("rms_norm", {x_.get(), w_norm, x_norm_.get(), b_dim_dec.get(), scalar_epsilon_.get()}, {}, grid_norm);

            run_linear_batched(x_norm_.get(), layer_prefix + "attn.W_q", q_.get(), config_.d_model, config_.d_model, batch_size);
            run_linear_batched(x_norm_.get(), layer_prefix + "attn.W_k", k_cur_.get(), config_.d_model, config_.n_kv_heads * head_dim, batch_size);
            run_linear_batched(x_norm_.get(), layer_prefix + "attn.W_v", v_cur_.get(), config_.d_model, config_.n_kv_heads * head_dim, batch_size);

            infra::KernelConfig grid_rope; 
            grid_rope.grid = infra::Dim3(head_dim / 2, config_.n_heads, batch_size); 
            grid_rope.block = infra::Dim3(32, 1, 1);
            
            backend_->execute_kernel("rope_ragged", 
                {q_.get(), cos_buf_.get(), sin_buf_.get(), positions_buf_.get(), q_.get(), scalar_head_dim_.get(), scalar_n_heads_.get()}, 
                {}, grid_rope);
            
            infra::KernelConfig grid_rope_k; 
            grid_rope_k.grid = infra::Dim3(head_dim / 2, config_.n_kv_heads, batch_size); 
            grid_rope_k.block = infra::Dim3(32, 1, 1);
            
            backend_->execute_kernel("rope_ragged", 
                {k_cur_.get(), cos_buf_.get(), sin_buf_.get(), positions_buf_.get(), k_cur_.get(), scalar_head_dim_.get(), scalar_n_kv_heads_.get()}, 
                {}, grid_rope_k);

            // KV Cache Update
            for (int i = 0; i < batch_size; ++i) {
                decodes[i]->kv_cache(l)->update(k_cur_.get(), v_cur_.get(), 1, i * config_.n_kv_heads * head_dim * sizeof(float));
            }

            // Attention
            std::vector<int32_t> strided_tables(batch_size * kv_config_.max_num_blocks);
            for (int i = 0; i < batch_size; ++i) {
                const auto& bt = decodes[i]->kv_cache(l)->block_table();
                std::memcpy(strided_tables.data() + i * kv_config_.max_num_blocks, bt.data(), bt.size() * sizeof(int32_t));
            }
            backend_->copy_to_device(block_table_buf_.get(), strided_tables.data(), strided_tables.size() * sizeof(int32_t));
            backend_->copy_to_device(cu_seqlens_buf_.get(), context_lens.data(), context_lens.size() * sizeof(int32_t)); // Using buf for context_lens
            
            infra::KernelConfig grid_attn; 
            // MetalBackend uses dispatchThreads. We need n_heads groups.
            int attn_block_size = (head_dim + 31) / 32 * 32;
            grid_attn.grid = infra::Dim3(config_.n_heads * attn_block_size, batch_size, 1); 
            grid_attn.block = infra::Dim3(attn_block_size, 1, 1);
            
            // Params: n_heads, n_kv, hd, scale, bs, max_num
            if (kv_config_.data_type == KvCacheDataType::INT8) {
                 backend_->execute_kernel("paged_attention_int8", 
                     {q_.get(), kv_manager_->global_keys_int8_device(), kv_manager_->global_values_int8_device(), 
                      kv_manager_->global_k_scales_device(), kv_manager_->global_v_scales_device(),
                      block_table_buf_.get(), cu_seqlens_buf_.get(), x_norm_.get(),
                      scalar_n_heads_.get(), scalar_n_kv_heads_.get(), scalar_head_dim_.get(), scalar_scale_attn_.get(), scalar_block_size_.get(), scalar_max_num_blocks_.get()}, 
                     {}, grid_attn);
            } else {
                 backend_->execute_kernel("paged_attention", 
                     {q_.get(), kv_manager_->global_keys_device(), kv_manager_->global_values_device(), block_table_buf_.get(), cu_seqlens_buf_.get(), x_norm_.get(),
                      scalar_n_heads_.get(), scalar_n_kv_heads_.get(), scalar_head_dim_.get(), scalar_scale_attn_.get(), scalar_block_size_.get(), scalar_max_num_blocks_.get()}, 
                     {}, grid_attn
                 );
            }
            
            // FFN
            run_linear_batched(x_norm_.get(), layer_prefix + "attn.W_o", q_.get(), config_.d_model, config_.d_model, batch_size);
            infra::KernelConfig grid_add; grid_add.grid = infra::Dim3(batch_size * config_.d_model, 1, 1); grid_add.block = infra::Dim3(256, 1, 1);
            backend_->execute_kernel("add", {residual_.get(), q_.get()}, {x_.get()}, grid_add);
            
            backend_->copy_to_device(residual_.get(), x_->raw_ptr(), batch_size * d_model_bytes);
            
            infra::DeviceBuffer* w_norm_ffn = get_buffer(gpu_weights_, layer_prefix + "ffn_norm_weight");
            auto b_dim_ffn_dec = alloc_u32((uint32_t)config_.d_model);
            backend_->execute_kernel("rms_norm", {x_.get(), w_norm_ffn, x_norm_.get(), b_dim_ffn_dec.get(), scalar_epsilon_.get()}, {}, grid_norm);
            
            run_linear_batched(x_norm_.get(), layer_prefix + "ffn.W_gate", ffn_gate_.get(), config_.d_model, config_.d_ff, batch_size);
            run_linear_batched(x_norm_.get(), layer_prefix + "ffn.W_up", ffn_up_.get(), config_.d_model, config_.d_ff, batch_size);
            
            infra::KernelConfig grid_ffn; grid_ffn.grid = infra::Dim3(batch_size * config_.d_ff, 1, 1); grid_ffn.block = infra::Dim3(256, 1, 1);
            backend_->execute_kernel("silu", {ffn_gate_.get()}, {ffn_gate_.get()}, grid_ffn);
            backend_->execute_kernel("mul", {ffn_gate_.get(), ffn_up_.get()}, {ffn_gate_.get()}, grid_ffn);
            
            run_linear_batched(ffn_gate_.get(), layer_prefix + "ffn.W_down", ffn_down_.get(), config_.d_ff, config_.d_model, batch_size);
            backend_->execute_kernel("add", {x_.get(), ffn_down_.get()}, {x_.get()}, grid_add);
        }

        // Output
        infra::DeviceBuffer* w_norm_final = get_buffer(gpu_weights_, "final_norm_weight");
        infra::KernelConfig grid_norm; grid_norm.grid = infra::Dim3(batch_size, 1, 1); grid_norm.block = infra::Dim3(std::min((int)batch_size, 256), 1, 1);
        auto b_dim_final_dec = alloc_u32((uint32_t)config_.d_model);
        backend_->execute_kernel("rms_norm", {x_.get(), w_norm_final, x_norm_.get(), b_dim_final_dec.get(), scalar_epsilon_.get()}, {}, grid_norm);
        
        run_linear_batched(x_norm_.get(), "lm_head", logits_.get(), config_.d_model, config_.vocab_size, batch_size);
        
        backend_->synchronize();
        std::vector<float> all_logits(batch_size * config_.vocab_size);
        backend_->copy_to_host(all_logits.data(), logits_.get(), all_logits.size() * sizeof(float));
        
        for (int i = 0; i < batch_size; ++i) {
            Tensor logits_tensor({(size_t)config_.vocab_size}, all_logits.data() + i * config_.vocab_size);
            int next_token = freellm::top_p_sample(logits_tensor, 0.9f, 0.7f);
            decodes[i]->add_token(next_token);
            
            // Check EOS or Max Length (same as prefill)
            // Llama 3 uses: 128001 (<|end_of_text|>), 128008 (<|eom_id|>), 128009 (<|eot_id|>)
            if (next_token == 128001 || next_token == 128008 || next_token == 128009 || next_token == 128255 ||
                decodes[i]->length() >= config_.max_seq_len) {
                decodes[i]->set_finished();
            }
        }
    }
    
    return true;
}

} // namespace freellm
