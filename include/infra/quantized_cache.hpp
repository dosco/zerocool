#pragma once

#include "core/tensor.hpp"
#include "kernels/quantiz/types.hpp"
#include "utils/cache_utils.hpp"
#include "infra/safetensors.hh"
#include <string>
#include <filesystem>
#include <unordered_map>
#include <optional>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace freellm {
namespace cache {

/**
 * @brief Generate cache key for quantized weights
 * 
 * Cache key is based on:
 * - Source file path and hash
 * - Quantization configuration (which tensors, what types)
 * 
 * @param model_path Path to source model file
 * @param quant_config_hash Hash of quantization configuration
 * @return Cache key string
 */
inline std::string generate_quant_cache_key(
    const std::string& model_path,
    const std::string& quant_config_hash
) {
    namespace fs = std::filesystem;
    
    // Get model name from path
    std::string model_name = fs::path(model_path).stem().string();
    
    // Compute hash of source file
    std::string source_hash = compute_file_hash(model_path);
    
    // Combine hashes
    std::string combined = source_hash + quant_config_hash;
    std::string combined_hash = compute_string_hash(combined);
    
    return model_name + "_" + short_hash(combined_hash);
}

/**
 * @brief Get path to quantized weight cache directory
 * 
 * @param cache_key Unique cache key for this model+config
 * @return Path to cache directory
 */
inline std::filesystem::path get_quant_cache_dir(const std::string& cache_key) {
    return ensure_cache_dir("weights") / cache_key;
}

/**
 * @brief Save quantized weights to cache
 * 
 * Saves quantized tensors in safetensors format for fast loading.
 * Also saves metadata with source hash and quantization config.
 * 
 * @param cache_key Unique cache key
 * @param quantized_weights Map of quantized tensors
 * @param source_hash Hash of source model file
 * @param quant_config_hash Hash of quantization config
 * @return true if save successful, false otherwise
 */
inline bool save_quantized_cache(
    const std::string& cache_key,
    const std::unordered_map<std::string, QuantizedTensor>& quantized_weights,
    const std::string& source_hash,
    const std::string& quant_config_hash
) {
    namespace fs = std::filesystem;
    
    try {
        // Create cache directory
        fs::path cache_dir = get_quant_cache_dir(cache_key);
        fs::create_directories(cache_dir);
        
        // Save each quantized tensor as a separate file
        // This allows partial loading if needed
        size_t saved_count = 0;
        for (const auto& [name, qtensor] : quantized_weights) {
            // Create a safe filename from tensor name
            std::string safe_name = name;
            std::replace(safe_name.begin(), safe_name.end(), '.', '_');
            std::replace(safe_name.begin(), safe_name.end(), '/', '_');
            
            fs::path tensor_path = cache_dir / (safe_name + ".qbin");
            
            // Write quantized data
            std::ofstream file(tensor_path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Warning: Failed to save quantized tensor: " << name << std::endl;
                continue;
            }
            
            // Write header: quant_type (4 bytes), data_size (8 bytes)
            uint32_t quant_type_val = static_cast<uint32_t>(qtensor.qtype());
            uint64_t data_size = qtensor.data_size();
            
            file.write(reinterpret_cast<const char*>(&quant_type_val), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&data_size), sizeof(uint64_t));
            
            // Write quantized data
            file.write(reinterpret_cast<const char*>(qtensor.data()), data_size);
            
            // Write shape
            const auto& shape = qtensor.shape();
            uint32_t shape_size = shape.size();
            file.write(reinterpret_cast<const char*>(&shape_size), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(shape.data()), 
                      shape_size * sizeof(size_t));
            
            saved_count++;
        }
        
        // Save metadata
        fs::path metadata_path = cache_dir / "metadata.txt";
        std::ofstream meta_file(metadata_path);
        if (meta_file.is_open()) {
            meta_file << "source_hash=" << source_hash << "\n";
            meta_file << "quant_config_hash=" << quant_config_hash << "\n";
            meta_file << "num_tensors=" << saved_count << "\n";
        }
        
        return saved_count > 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving quantized cache: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Load quantized weights from cache
 * 
 * @param cache_key Unique cache key
 * @param tensor_names List of tensor names to load
 * @return Map of loaded quantized tensors, or empty if cache invalid
 */
inline std::unordered_map<std::string, QuantizedTensor> load_quantized_cache(
    const std::string& cache_key,
    const std::vector<std::string>& tensor_names
) {
    namespace fs = std::filesystem;
    
    std::unordered_map<std::string, QuantizedTensor> result;
    
    try {
        fs::path cache_dir = get_quant_cache_dir(cache_key);
        
        // Check if cache directory exists
        if (!fs::exists(cache_dir)) {
            return result;
        }
        
        // Load each tensor
        for (const auto& name : tensor_names) {
            // Create safe filename
            std::string safe_name = name;
            std::replace(safe_name.begin(), safe_name.end(), '.', '_');
            std::replace(safe_name.begin(), safe_name.end(), '/', '_');
            
            fs::path tensor_path = cache_dir / (safe_name + ".qbin");
            
            if (!fs::exists(tensor_path)) {
                continue; // Skip missing tensors
            }
            
            // Read quantized data
            std::ifstream file(tensor_path, std::ios::binary);
            if (!file.is_open()) {
                continue;
            }
            
            // Read header
            uint32_t quant_type_val;
            uint64_t data_size;
            
            file.read(reinterpret_cast<char*>(&quant_type_val), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
            
            // Skip quantized data for now (we'll read it after creating the tensor)
            file.seekg(data_size, std::ios::cur);
            
            // Read shape
            uint32_t shape_size;
            file.read(reinterpret_cast<char*>(&shape_size), sizeof(uint32_t));
            
            std::vector<size_t> shape(shape_size);
            file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
            
            // Create QuantizedTensor with proper constructor
            quant::QuantType qtype = static_cast<quant::QuantType>(quant_type_val);
            QuantizedTensor qtensor(shape, qtype);
            
            // Now read the data into the tensor
            file.seekg(sizeof(uint32_t) + sizeof(uint64_t), std::ios::beg);
            file.read(reinterpret_cast<char*>(qtensor.data()), data_size);
            
            result[name] = std::move(qtensor);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading quantized cache: " << e.what() << std::endl;
        result.clear();
    }
    
    return result;
}

/**
 * @brief Check if quantized cache is valid
 * 
 * @param cache_key Unique cache key
 * @param source_hash Expected source file hash
 * @param quant_config_hash Expected quant config hash
 * @return true if cache is valid, false otherwise
 */
inline bool is_quant_cache_valid(
    const std::string& cache_key,
    const std::string& source_hash,
    const std::string& quant_config_hash
) {
    namespace fs = std::filesystem;
    
    fs::path cache_dir = get_quant_cache_dir(cache_key);
    fs::path metadata_path = cache_dir / "metadata.txt";
    
    if (!fs::exists(metadata_path)) {
        return false;
    }
    
    // Read metadata
    std::ifstream file(metadata_path);
    if (!file.is_open()) {
        return false;
    }
    
    std::string cached_source_hash;
    std::string cached_quant_hash;
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("source_hash=") == 0) {
            cached_source_hash = line.substr(12);
        } else if (line.find("quant_config_hash=") == 0) {
            cached_quant_hash = line.substr(18);
        }
    }
    
    return cached_source_hash == source_hash && 
           cached_quant_hash == quant_config_hash;
}

} // namespace cache
} // namespace freellm
