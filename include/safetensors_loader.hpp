#pragma once

#include "tensor.hpp"
#include "model_loader.hpp"
#include "safetensors.hh"
#include "parallel_tensor_loader.hpp"
#include "quantiz/types.hpp"
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <optional>

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
 * @brief Map HuggingFace TinyLLaMA tensor names to our internal names
 *
 * TinyLLaMA uses standard HuggingFace naming convention.
 * We need to map these to our simpler internal names.
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
            }
        }

        // MLP weights (SwiGLU in TinyLLaMA)
        if (remaining.find("mlp") == 0) {
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

    // Unknown mapping
    return "";
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
    std::println("Loading safetensors from: {}", filepath);

    // Load using safetensors-cpp library
    safetensors::safetensors_t st;
    std::string warn, err;

    bool ret = safetensors::load_from_file(filepath, &st, &warn, &err);
    if (!ret) {
        throw std::runtime_error("Failed to load safetensors: " + err);
    }

    if (!warn.empty()) {
        std::println("  Warning: {}", warn);
    }

    // Validate data offsets
    if (!safetensors::validate_data_offsets(st, err)) {
        throw std::runtime_error("Invalid safetensors data offsets: " + err);
    }

    std::println("  File size: {:.2f} GB", st.storage.size() / (1024.0 * 1024.0 * 1024.0));
    std::println("  Found {} tensors", st.tensors.size());

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
                std::println("  Warning: Skipping {} (unsupported dtype)", hf_name);
                continue;
            }
        }

        weights[our_name] = std::move(tensor);
        loaded_count++;

        if (loaded_count % 50 == 0) {
            std::println("  Loaded {} tensors...", loaded_count);
        }
    }

    std::println("  ✓ Successfully loaded {} tensors", loaded_count);

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

} // namespace freellm
