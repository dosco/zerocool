#pragma once

#include "core/tensor.hpp"
#include <string>
#include <unordered_map>
#include <fstream>
#include <memory>

namespace freellm {

/**
 * @brief Model weight storage
 *
 * Maps tensor names to their data.
 * Example names: "layers.0.attn.W_q", "layers.0.norm.weight", etc.
 */
using WeightMap = std::unordered_map<std::string, Tensor>;

/**
 * @brief Simple model loader
 *
 * For now, this provides a basic interface for loading model weights.
 * Future versions will support:
 * - Safetensors format (industry standard)
 * - Memory-mapped files for large models
 * - Lazy loading
 *
 * Current implementation focuses on the interface and basic binary format.
 */
class ModelLoader {
public:
    /**
     * @brief Load weights from a simple binary format
     *
     * Binary format (for testing/prototyping):
     * - Magic number: "FLLM" (4 bytes)
     * - Version: uint32_t
     * - Num tensors: uint32_t
     * - For each tensor:
     *   - Name length: uint32_t
     *   - Name: char[name_length]
     *   - Num dimensions: uint32_t
     *   - Dimensions: uint64_t[num_dims]
     *   - Data: float[product(dimensions)]
     *
     * @param filepath Path to binary weight file
     * @return WeightMap containing all tensors
     */
    static WeightMap load_binary(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        WeightMap weights;

        // Read and verify magic number
        char magic[5] = {0};
        file.read(magic, 4);
        if (std::string(magic) != "FLLM") {
            throw std::runtime_error("Invalid file format (bad magic number)");
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            throw std::runtime_error("Unsupported file version");
        }

        // Read number of tensors
        uint32_t num_tensors;
        file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

        // Read each tensor
        for (uint32_t i = 0; i < num_tensors; ++i) {
            // Read name
            uint32_t name_length;
            file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));

            std::string name(name_length, '\0');
            file.read(&name[0], name_length);

            // Read dimensions
            uint32_t num_dims;
            file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));

            std::vector<size_t> shape(num_dims);
            for (uint32_t d = 0; d < num_dims; ++d) {
                uint64_t dim;
                file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                shape[d] = static_cast<size_t>(dim);
            }

            // Create tensor and read data
            Tensor tensor(shape);
            size_t num_elements = tensor.size();
            file.read(reinterpret_cast<char*>(tensor.data()), num_elements * sizeof(float));

            weights[name] = std::move(tensor);
        }

        return weights;
    }

    /**
     * @brief Save weights to simple binary format
     *
     * @param weights WeightMap to save
     * @param filepath Output file path
     */
    static void save_binary(const WeightMap& weights, const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to create file: " + filepath);
        }

        // Write magic number
        file.write("FLLM", 4);

        // Write version
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // Write number of tensors
        uint32_t num_tensors = static_cast<uint32_t>(weights.size());
        file.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

        // Write each tensor
        for (const auto& [name, tensor] : weights) {
            // Write name
            uint32_t name_length = static_cast<uint32_t>(name.length());
            file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
            file.write(name.c_str(), name_length);

            // Write dimensions
            uint32_t num_dims = static_cast<uint32_t>(tensor.ndim());
            file.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));

            for (size_t dim : tensor.shape()) {
                uint64_t dim64 = static_cast<uint64_t>(dim);
                file.write(reinterpret_cast<const char*>(&dim64), sizeof(dim64));
            }

            // Write data
            file.write(reinterpret_cast<const char*>(tensor.data()),
                      tensor.size() * sizeof(float));
        }
    }

    /**
     * @brief Initialize random weights for testing
     *
     * Creates a WeightMap with randomly initialized tensors
     * matching the TinyLLaMA architecture.
     *
     * @param vocab_size Vocabulary size
     * @param n_layers Number of transformer layers
     * @param d_model Model dimension
     * @param d_ff Feed-forward dimension
     * @return WeightMap with random weights
     */
    static WeightMap create_random_weights(
        size_t vocab_size,
        size_t n_layers,
        size_t d_model,
        size_t d_ff
    ) {
        WeightMap weights;

        // Token embeddings
        weights["embed.weight"] = Tensor({vocab_size, d_model}, 0.01f);

        // For each layer
        for (size_t i = 0; i < n_layers; ++i) {
            std::string prefix = "layers." + std::to_string(i) + ".";

            // Attention weights
            weights[prefix + "attn.W_q"] = Tensor({d_model, d_model}, 0.01f);
            weights[prefix + "attn.W_k"] = Tensor({d_model, d_model}, 0.01f);
            weights[prefix + "attn.W_v"] = Tensor({d_model, d_model}, 0.01f);
            weights[prefix + "attn.W_o"] = Tensor({d_model, d_model}, 0.01f);

            // Attention norm
            weights[prefix + "attn_norm.weight"] = Tensor({d_model}, 1.0f);

            // Feed-forward weights
            weights[prefix + "ffn.W1"] = Tensor({d_model, d_ff}, 0.01f);
            weights[prefix + "ffn.W2"] = Tensor({d_ff, d_model}, 0.01f);

            // FFN norm
            weights[prefix + "ffn_norm.weight"] = Tensor({d_model}, 1.0f);
        }

        // Final norm
        weights["final_norm.weight"] = Tensor({d_model}, 1.0f);

        // Output projection (LM head)
        weights["lm_head.weight"] = Tensor({d_model, vocab_size}, 0.01f);

        return weights;
    }
};

} // namespace freellm
