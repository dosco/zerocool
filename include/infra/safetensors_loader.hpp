#pragma once

#include "core/tensor.hpp"
#include "core/model_config.hpp"
#include "infra/model_loader.hpp"
#include "infra/safetensors.hh"
#include "infra/parallel_tensor_loader.hpp"
#include "infra/quantized_cache.hpp"
#include "kernels/quantiz/types.hpp"
#include "utils/terminal_ui.hpp"
#include "utils/cache_utils.hpp"
#include <string>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <optional>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <nlohmann/json.hpp>

namespace freellm {

/**
 * @brief Convert BF16 (bfloat16) to F32 (float)
 *
 * BF16 is a 16-bit floating point format that truncates F32:
 * - 1 sign bit
 * - 8 exponent bits (same as F32)
 * - 7 mantissa bits (F32 has 23)
 *
 * To convert BF16 to F32, we just shift left by 16 bits.
 */
inline float bf16_to_f32(uint16_t bf16_value) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16_value) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

/**
 * @brief Convert FP16 (half precision) to F32 (float)
 *
 * FP16 format:
 * - 1 sign bit
 * - 5 exponent bits
 * - 10 mantissa bits
 */
inline float fp16_to_f32(uint16_t fp16_value) {
    uint32_t sign = (fp16_value >> 15) & 0x1;
    uint32_t exponent = (fp16_value >> 10) & 0x1F;
    uint32_t mantissa = fp16_value & 0x3FF;

    uint32_t f32_bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            f32_bits = sign << 31;
        } else {
            // Denormalized number
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32_bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        // Infinity or NaN
        f32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        // Normalized number
        f32_bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

/**
 * @brief Map HuggingFace tensor names to our internal names
 *
 * This function handles multiple model architectures:
 * - Dense models (TinyLlama, Llama): Map to internal ffn.W_* names
 * - MoE models (OLMoE, Mixtral): Pass through MoE-specific names
 *
 * The key insight: For MoE models, we keep the original naming structure
 * (mlp.gate.weight, mlp.experts.N.*) so load_moe_weights can find them.
 */
inline std::string map_hf_name_to_freellm(const std::string& hf_name) {
    // Embed tokens
    if (hf_name == "model.embed_tokens.weight") {
        return "token_embedding";
    }

    if (hf_name == "lm_head.weight") {
        return "lm_head";
    }

    // Final norm
    if (hf_name == "model.norm.weight") {
        return "final_norm_weight";
    }

    // Layer-specific weights
    if (hf_name.find("model.layers.") == 0) {
        // Extract layer number
        size_t layer_start = strlen("model.layers.");
        size_t layer_end = hf_name.find('.', layer_start);
        std::string layer_num = hf_name.substr(layer_start, layer_end - layer_start);

        std::string remaining = hf_name.substr(layer_end + 1);

        // Attention weights
        if (remaining.find("self_attn") == 0) {
            if (remaining.find("q_proj.weight") != std::string::npos) {
                return "layers." + layer_num + ".attn.W_q";
            } else if (remaining.find("k_proj.weight") != std::string::npos) {
                return "layers." + layer_num + ".attn.W_k";
            } else if (remaining.find("v_proj.weight") != std::string::npos) {
                return "layers." + layer_num + ".attn.W_v";
            } else if (remaining.find("o_proj.weight") != std::string::npos) {
                return "layers." + layer_num + ".attn.W_o";
            } else if (remaining.find("q_norm.weight") != std::string::npos) {
                // QK-normalization weights (used by OLMoE)
                return "layers." + layer_num + ".attn.q_norm_weight";
            } else if (remaining.find("k_norm.weight") != std::string::npos) {
                return "layers." + layer_num + ".attn.k_norm_weight";
            }
        }

        // MLP weights - check for MoE patterns FIRST
        if (remaining.find("mlp") == 0) {
            // MoE: Router/Gate weight - PASSTHROUGH with simplified prefix
            if (remaining == "mlp.gate.weight") {
                return "layers." + layer_num + ".mlp.gate.weight";
            }
            
            // MoE: Expert weights - PASSTHROUGH with simplified prefix
            // Format: mlp.experts.{idx}.{gate_proj|up_proj|down_proj}.weight
            if (remaining.find("mlp.experts.") == 0) {
                // Pass through the entire mlp.experts.* structure
                return "layers." + layer_num + "." + remaining;
            }
            
            // Dense MLP weights (SwiGLU in TinyLLaMA, Llama)
            if (remaining.find("gate_proj.weight") != std::string::npos) {
                return "layers." + layer_num + ".ffn.W_gate";
            } else if (remaining.find("up_proj.weight") != std::string::npos) {
                return "layers." + layer_num + ".ffn.W_up";
            } else if (remaining.find("down_proj.weight") != std::string::npos) {
                return "layers." + layer_num + ".ffn.W_down";
            }
        }

        // Norms
        if (remaining == "input_layernorm.weight") {
            return "layers." + layer_num + ".attn_norm_weight";
        } else if (remaining == "post_attention_layernorm.weight") {
            return "layers." + layer_num + ".ffn_norm_weight";
        }
    }

    // Unknown mapping - return empty to skip
    return "";
}

/**
 * @brief Detect model architecture from weight names
 *
 * Parses model.safetensors.index.json to auto-detect:
 * - MoE (Mixture of Experts) models
 * - Number of experts
 * - Number of layers
 */
struct ModelArchitectureInfo {
    bool is_moe = false;
    size_t num_experts = 0;
    size_t num_layers = 0;
    std::vector<std::string> weight_names;
};

inline ModelArchitectureInfo detect_architecture_from_index(const std::string& model_dir) {
    namespace fs = std::filesystem;
    ModelArchitectureInfo info;
    
    std::string index_path = model_dir + "/model.safetensors.index.json";
    if (!fs::exists(index_path)) {
        // Single-file model, can't detect
        return info;
    }
    
    std::ifstream f(index_path);
    if (!f.is_open()) return info;
    
    try {
        nlohmann::json j;
        f >> j;
        
        if (!j.contains("weight_map")) return info;
        
        const auto& weight_map = j["weight_map"];
        size_t max_expert_idx = 0;
        size_t max_layer_idx = 0;
        
        for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
            const std::string& name = it.key();
            info.weight_names.push_back(name);
            
            // Detect MoE
            if (name.find("mlp.experts.") != std::string::npos) {
                info.is_moe = true;
                
                // Extract expert index
                size_t pos = name.find("mlp.experts.");
                if (pos != std::string::npos) {
                    pos += strlen("mlp.experts.");
                    size_t end = name.find('.', pos);
                    if (end != std::string::npos) {
                        size_t expert_idx = std::stoul(name.substr(pos, end - pos));
                        max_expert_idx = std::max(max_expert_idx, expert_idx);
                    }
                }
            }
            
            // Detect layer count
            if (name.find("model.layers.") != std::string::npos) {
                size_t pos = strlen("model.layers.");
                size_t end = name.find('.', pos);
                if (end != std::string::npos) {
                    size_t layer_idx = std::stoul(name.substr(pos, end - pos));
                    max_layer_idx = std::max(max_layer_idx, layer_idx);
                }
            }
        }
        
        info.num_experts = max_expert_idx + 1;
        info.num_layers = max_layer_idx + 1;
        
    } catch (...) {
        // JSON parse error
    }
    
    return info;
}

/**
 * @brief Load ModelConfig from HuggingFace config.json
 *
 * Parses config.json to auto-populate model dimensions.
 * Supports Llama, TinyLlama, OLMoE, Mixtral, and other HF-compatible models.
 *
 * @param model_dir Path to the model directory containing config.json
 * @return Populated ModelConfig, or default config if parsing fails
 */
inline ModelConfig load_model_config_from_json(const std::string& model_dir) {
    namespace fs = std::filesystem;
    ModelConfig config;
    
    std::string config_path = model_dir + "/config.json";
    if (!fs::exists(config_path)) {
        throw std::runtime_error("config.json not found in: " + model_dir);
    }
    
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config.json: " + config_path);
    }
    
    try {
        nlohmann::json j;
        f >> j;
        
        // Required fields
        if (j.contains("hidden_size")) {
            config.d_model = j["hidden_size"].get<size_t>();
        }
        if (j.contains("intermediate_size")) {
            config.d_ff = j["intermediate_size"].get<size_t>();
        }
        if (j.contains("num_hidden_layers")) {
            config.n_layers = j["num_hidden_layers"].get<size_t>();
        }
        if (j.contains("num_attention_heads")) {
            config.n_heads = j["num_attention_heads"].get<size_t>();
        }
        if (j.contains("num_key_value_heads")) {
            config.n_kv_heads = j["num_key_value_heads"].get<size_t>();
        } else {
            // Fallback: MHA (n_kv_heads = n_heads)
            config.n_kv_heads = config.n_heads;
        }
        if (j.contains("vocab_size")) {
            config.vocab_size = j["vocab_size"].get<size_t>();
        }
        if (j.contains("max_position_embeddings")) {
            config.max_seq_len = j["max_position_embeddings"].get<size_t>();
        }
        
        // Optional fields with defaults
        if (j.contains("rms_norm_eps")) {
            config.norm_eps = j["rms_norm_eps"].get<float>();
        }
        if (j.contains("rope_theta")) {
            config.rope_theta = j["rope_theta"].get<float>();
        }
        
        // MoE-specific fields
        if (j.contains("num_experts")) {
            config.num_experts = j["num_experts"].get<size_t>();
        }
        if (j.contains("num_experts_per_tok")) {
            config.num_experts_per_token = j["num_experts_per_tok"].get<size_t>();
        } else if (j.contains("num_experts_per_token")) {
            config.num_experts_per_token = j["num_experts_per_token"].get<size_t>();
        }
        
        if (j.contains("norm_topk_prob")) {
             config.norm_topk_prob = j["norm_topk_prob"].get<bool>();
        }

        // Weight tying (common in OLMoE, Gemma, etc.)
        if (j.contains("tie_word_embeddings")) {
            config.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
        }

        // Detect model type for logging
        std::string model_type = "unknown";
        if (j.contains("model_type")) {
            model_type = j["model_type"].get<std::string>();
        }
        
        ui::print_success("Loaded config: " + model_type + 
                         " (d=" + std::to_string(config.d_model) + 
                         ", L=" + std::to_string(config.n_layers) + 
                         ", H=" + std::to_string(config.n_heads) + 
                         ", V=" + std::to_string(config.vocab_size) + ")");
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse config.json: " + std::string(e.what()));
    }
    
    return config;
}

/**
 * @brief Load weights from safetensors file using safetensors-cpp library
 *
 * This uses the professional safetensors-cpp library which handles:
 * - Multiple data types (F32, F16, BF16, INT8, etc.)
 * - Proper JSON parsing
 * - Validation and error handling
 *
 * @param filepath Path to .safetensors file
 * @return WeightMap with loaded tensors
 */
inline WeightMap load_safetensors(const std::string& filepath) {
    ui::print_loading("Loading: " + filepath);

    // Load using safetensors-cpp library
    safetensors::safetensors_t st;
    std::string warn, err;

    bool ret = safetensors::load_from_file(filepath, &st, &warn, &err);
    if (!ret) {
        throw std::runtime_error("Failed to load safetensors: " + err);
    }

    if (!warn.empty()) {
        ui::print_warning(warn);
    }

    // Validate data offsets
    if (!safetensors::validate_data_offsets(st, err)) {
        throw std::runtime_error("Invalid safetensors data offsets: " + err);
    }

    std::ostringstream size_ss;
    size_ss << std::fixed << std::setprecision(2) << (st.storage.size() / (1024.0 * 1024.0 * 1024.0)) << " GB";
    ui::print_status("File size: " + size_ss.str());
    ui::print_status("Found " + std::to_string(st.tensors.size()) + " tensors");

    // Load tensors
    WeightMap weights;
    size_t loaded_count = 0;

    // Iterate through tensors using the ordered_dict API
    const auto& tensor_keys = st.tensors.keys();
    for (size_t i = 0; i < tensor_keys.size(); ++i) {
        const std::string& hf_name = tensor_keys[i];
        safetensors::tensor_t tensor_info;
        if (!st.tensors.at(hf_name, &tensor_info)) {
            continue; // Skip if we can't get the tensor info
        }
        // Map to our internal name
        std::string our_name = map_hf_name_to_freellm(hf_name);
        if (our_name.empty()) {
            // Skip unmapped tensors
            continue;
        }

        // Shape is already vector<size_t> in safetensors-cpp
        const std::vector<size_t>& shape = tensor_info.shape;

        // Create tensor
        Tensor tensor(shape);
        size_t num_elements = tensor.size();

        // Get pointer to raw data
        const uint8_t* raw_data = st.storage.data() + tensor_info.data_offsets[0];

        // Convert based on dtype
        switch (tensor_info.dtype) {
            case safetensors::dtype::kFLOAT32: {
                // Direct copy for F32
                const float* f32_data = reinterpret_cast<const float*>(raw_data);
                std::memcpy(tensor.data(), f32_data, num_elements * sizeof(float));
                break;
            }

            case safetensors::dtype::kBFLOAT16: {
                // Convert BF16 to F32
                const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(raw_data);
                float* f32_data = tensor.data();
                for (size_t i = 0; i < num_elements; ++i) {
                    f32_data[i] = bf16_to_f32(bf16_data[i]);
                }
                break;
            }

            case safetensors::dtype::kFLOAT16: {
                // Convert FP16 to F32
                const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(raw_data);
                float* f32_data = tensor.data();
                for (size_t i = 0; i < num_elements; ++i) {
                    f32_data[i] = fp16_to_f32(fp16_data[i]);
                }
                break;
            }

            default: {
                ui::print_warning("Skipping " + hf_name + " (unsupported dtype)");
                continue;
            }
        }

        weights[our_name] = std::move(tensor);
        loaded_count++;

        if (loaded_count % 50 == 0) {
            ui::print_progress_bar(loaded_count, tensor_keys.size());
        }
    }

    ui::print_success("Loaded " + std::to_string(loaded_count) + " tensors");

    return weights;
}

/**
 * @brief Load weights from safetensors file using parallel pipeline
 *
 * This is the high-performance version that uses:
 * - Memory-mapping (mmap) for zero-copy I/O
 * - Producer-consumer pattern with bounded queue
 * - Thread pool for parallel dtype conversion
 * - Optional parallel quantization
 *
 * Performance characteristics:
 * - I/O and CPU work are overlapped (producer reads while workers convert)
 * - All CPU cores are utilized for data conversion
 * - Bounded queue prevents unbounded memory growth
 * - mmap allows OS to manage paging efficiently
 *
 * This is particularly effective for:
 * - Large models (multi-GB files)
 * - Systems with many CPU cores
 * - Mixed-precision models (BF16, FP16 → F32 conversion)
 * - Quantization workflows (F32 → Q4_K, Q8_0, etc.)
 *
 * @param filepath Path to .safetensors file
 * @param num_threads Number of worker threads (default: hardware concurrency)
 * @return WeightMap with loaded tensors (F32, no quantization)
 */
inline WeightMap load_safetensors_parallel(
    const std::string& filepath,
    size_t num_threads = std::thread::hardware_concurrency()
) {
    ParallelTensorLoader loader(num_threads);

    // Use the map_hf_name_to_freellm function for name mapping
    auto name_mapper = [](const std::string& hf_name) -> std::string {
        return map_hf_name_to_freellm(hf_name);
    };

    // No quantization in this basic loader
    return loader.load(filepath, name_mapper, nullptr);
}

/**
 * @brief Load weights with parallel quantization
 *
 * Extended version that performs quantization in parallel during loading.
 * This is significantly faster than loading F32 and quantizing sequentially.
 *
 * The quantization resolver is called for each tensor to determine its
 * quantization type based on the tensor name.
 *
 * @param filepath Path to .safetensors file
 * @param quant_resolver Function that returns QuantType for each tensor name
 * @param num_threads Number of worker threads (default: hardware concurrency)
 * @return Tuple of (F32 WeightMap, Quantized results map)
 */
inline std::pair<WeightMap, std::unordered_map<std::string, QuantizedTensor>>
load_safetensors_parallel_quantized(
    const std::string& filepath,
    std::function<std::optional<quant::QuantType>(const std::string&)> quant_resolver,
    size_t num_threads = std::thread::hardware_concurrency()
) {
    ParallelTensorLoader loader(num_threads);

    // Use the map_hf_name_to_freellm function for name mapping
    auto name_mapper = [](const std::string& hf_name) -> std::string {
        return map_hf_name_to_freellm(hf_name);
    };

    // Load with quantization
    return loader.load_with_quantization(filepath, name_mapper, quant_resolver);
}

/**
 * @brief Load weights from a file or directory
 * 
 * If filepath is a directory, it loads all .safetensors files in it and merges them.
 * If filepath is a file, it loads that file.
 * 
 * @param path Path to file or directory
 * @param quant_resolver Quantization resolver
 * @param num_threads Num threads
 * @return Merged weights
 */
inline std::pair<WeightMap, std::unordered_map<std::string, QuantizedTensor>>
load_model_from_path(
    const std::string& path,
    std::function<std::optional<quant::QuantType>(const std::string&)> quant_resolver,
    size_t num_threads = std::thread::hardware_concurrency()
) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;

    if (fs::is_directory(path)) {
        ui::print_subsection("Scanning: " + path);
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.path().extension() == ".safetensors") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end()); // Ensure deterministic order
        if (files.empty()) {
            throw std::runtime_error("No .safetensors files found in directory: " + path);
        }
        ui::print_status("Found " + std::to_string(files.size()) + " weight shards");
    } else {
        files.push_back(path);
    }

    // 1. Pre-scan headers to discover tensor names and categorize them
    std::vector<std::string> all_quant_names;
    std::unordered_map<std::string, std::vector<std::string>> file_to_f32_tensors;
    
    ui::print_status("Scanning headers...");
    for(const auto& file : files) {
         safetensors::safetensors_t st;
         std::string w, e;
         // Perform lightweight header parse
         if(!safetensors::load_from_file(file, &st, &w, &e)) continue;
         
         for(const auto& key : st.tensors.keys()) {
             std::string name = map_hf_name_to_freellm(key);
             if(name.empty()) continue;
             
             if(quant_resolver && quant_resolver(name)) {
                 all_quant_names.push_back(name);
             } else {
                 file_to_f32_tensors[file].push_back(name);
             }
         }
    }

    // Generate cache key
    std::string quant_config_hash = "default"; 
    std::string cache_key = cache::generate_quant_cache_key(files[0], quant_config_hash);
    
    WeightMap total_f32;
    std::unordered_map<std::string, QuantizedTensor> total_quant;
    bool cache_loaded = false;

    // 2. Try loading quantized weights from cache
    if (quant_resolver && !all_quant_names.empty()) {
        std::string source_hash = cache::compute_file_hash(files[0]);
        if (cache::is_quant_cache_valid(cache_key, source_hash, quant_config_hash)) {
             ui::print_status("Found valid quantized weight cache");
             ui::print_loading("Loading from cache...");
             
             total_quant = cache::load_quantized_cache(cache_key, all_quant_names);
             
             if (!total_quant.empty()) {
                 ui::print_success("Loaded " + std::to_string(total_quant.size()) + " quantized tensors from cache");
                 cache_loaded = true;
                 
                 // 3. Load remaining F32 tensors from original files
                 ui::print_loading("Loading remaining F32 tensors...");
                 size_t f32_count = 0;
                 
                 for (const auto& [file, names] : file_to_f32_tensors) {
                     if (names.empty()) continue;
                     
                     safetensors::safetensors_t st;
                     std::string w, e;
                     if(!safetensors::load_from_file(file, &st, &w, &e)) continue;
                     
                     for (const auto& name : names) {
                         // Reverse mapping is hard, but we iterate st keys to find matches?
                         // Or we store HF names in file_to_f32_tensors?
                         // We stored 'name' (internal).
                         // We need HF keys to look up in st.
                         // Optimization: Store HF key -> internal name in separate map during scan?
                         // Let's re-scan keys of this file.
                         for (const auto& hf_key : st.tensors.keys()) {
                             if (map_hf_name_to_freellm(hf_key) == name) {
                                  // Found it
                                  safetensors::tensor_t info;
                                  st.tensors.at(hf_key, &info);
                                  
                                  Tensor tensor(info.shape);
                                  const uint8_t* raw = st.storage.data() + info.data_offsets[0];
                                  
                                  // Copy/Convert
                                  if (info.dtype == safetensors::dtype::kFLOAT32) {
                                      std::memcpy(tensor.data(), raw, tensor.size() * sizeof(float));
                                  } else if (info.dtype == safetensors::dtype::kBFLOAT16) {
                                      const uint16_t* ptr = (const uint16_t*)raw;
                                      float* dst = tensor.data();
                                      for(size_t k=0; k<tensor.size(); ++k) dst[k] = bf16_to_f32(ptr[k]);
                                  } else if (info.dtype == safetensors::dtype::kFLOAT16) {
                                      const uint16_t* ptr = (const uint16_t*)raw;
                                      float* dst = tensor.data();
                                      for(size_t k=0; k<tensor.size(); ++k) dst[k] = fp16_to_f32(ptr[k]);
                                  }
                                  
                                  total_f32[name] = std::move(tensor);
                                  f32_count++;
                                  break;
                             }
                         }
                     }
                 }
                 ui::print_success("Loaded " + std::to_string(f32_count) + " F32 tensors");
             }
        }
    }
    
    // 4. Fallback: Full Parallel Load
    if (!cache_loaded) {
        for (size_t i = 0; i < files.size(); ++i) {
            const auto& file = files[i];
            std::string filename = fs::path(file).filename().string();
            ui::print_loading("[" + std::to_string(i + 1) + "/" + std::to_string(files.size()) + "] " + filename);
            auto [f32, quant] = load_safetensors_parallel_quantized(file, quant_resolver, num_threads);
            
            for (auto& [k, v] : f32) total_f32[k] = std::move(v);
            for (auto& [k, v] : quant) total_quant[k] = std::move(v);
        }
        
        ui::print_success("Loaded " + std::to_string(total_f32.size()) + " F32 tensors, " + std::to_string(total_quant.size()) + " quantized");
        
        if (quant_resolver && !total_quant.empty()) {
            ui::print_loading("Caching quantized weights...");
            std::string source_hash = cache::compute_file_hash(files[0]);
            bool saved = cache::save_quantized_cache(cache_key, total_quant, source_hash, quant_config_hash);
            if (saved) {
                ui::print_success("Cached " + std::to_string(total_quant.size()) + " quantized tensors");
            }
        }
    }
    
    return {std::move(total_f32), std::move(total_quant)};
}

} // namespace freellm
