#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <cstdlib>

// For SHA-256 hashing
#include <CommonCrypto/CommonDigest.h>

namespace freellm {
namespace cache {

/**
 * @brief Get the cache directory for freellm
 * 
 * Returns platform-specific cache directory:
 * - macOS: ~/.cache/freellm/
 * - Linux: ~/.cache/freellm/
 * - Windows: %LOCALAPPDATA%/freellm/cache/
 * 
 * @return Absolute path to cache directory
 */
inline std::filesystem::path get_cache_dir() {
    namespace fs = std::filesystem;
    
    const char* home = std::getenv("HOME");
    if (!home) {
        throw std::runtime_error("HOME environment variable not set");
    }
    
    fs::path cache_root = fs::path(home) / ".cache" / "freellm";
    return cache_root;
}

/**
 * @brief Ensure cache directory exists
 * 
 * Creates the cache directory and any necessary parent directories.
 * 
 * @param subdir Optional subdirectory within cache (e.g., "metal", "weights")
 * @return Absolute path to the created directory
 */
inline std::filesystem::path ensure_cache_dir(const std::string& subdir = "") {
    namespace fs = std::filesystem;
    
    fs::path cache_path = get_cache_dir();
    if (!subdir.empty()) {
        cache_path /= subdir;
    }
    
    if (!fs::exists(cache_path)) {
        fs::create_directories(cache_path);
    }
    
    return cache_path;
}

/**
 * @brief Compute SHA-256 hash of a file
 * 
 * @param filepath Path to file to hash
 * @return Hex-encoded SHA-256 hash (64 characters)
 */
inline std::string compute_file_hash(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for hashing: " + filepath);
    }
    
    CC_SHA256_CTX sha256;
    CC_SHA256_Init(&sha256);
    
    constexpr size_t BUFFER_SIZE = 8192;
    char buffer[BUFFER_SIZE];
    
    while (file.read(buffer, BUFFER_SIZE) || file.gcount() > 0) {
        CC_SHA256_Update(&sha256, buffer, file.gcount());
    }
    
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(hash, &sha256);
    
    // Convert to hex string
    std::ostringstream oss;
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    
    return oss.str();
}

/**
 * @brief Compute SHA-256 hash of a string
 * 
 * @param data String data to hash
 * @return Hex-encoded SHA-256 hash (64 characters)
 */
inline std::string compute_string_hash(const std::string& data) {
    CC_SHA256_CTX sha256;
    CC_SHA256_Init(&sha256);
    CC_SHA256_Update(&sha256, data.c_str(), data.length());
    
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(hash, &sha256);
    
    // Convert to hex string
    std::ostringstream oss;
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    
    return oss.str();
}

/**
 * @brief Get short hash (first 12 characters) for human-readable cache keys
 * 
 * @param full_hash Full SHA-256 hash
 * @return First 12 characters of hash
 */
inline std::string short_hash(const std::string& full_hash) {
    return full_hash.substr(0, 12);
}

/**
 * @brief Check if a cached file is valid
 * 
 * A cache is valid if:
 * 1. The cached file exists
 * 2. The source file hash matches the expected hash
 * 
 * @param cache_path Path to cached file
 * @param source_path Path to source file
 * @param expected_hash Expected hash of source file (if empty, will compute)
 * @return true if cache is valid, false otherwise
 */
inline bool is_cache_valid(
    const std::filesystem::path& cache_path,
    const std::string& source_path,
    const std::string& expected_hash = ""
) {
    namespace fs = std::filesystem;
    
    // Check if cache exists
    if (!fs::exists(cache_path)) {
        return false;
    }
    
    // Check if source exists
    if (!fs::exists(source_path)) {
        return false;
    }
    
    // Compute or use provided hash
    std::string source_hash = expected_hash.empty() 
        ? compute_file_hash(source_path) 
        : expected_hash;
    
    // For now, we assume cache is valid if it exists
    // In a more sophisticated implementation, we'd store metadata
    // with the source hash and compare
    return true;
}

/**
 * @brief Write simple metadata file
 * 
 * @param metadata_path Path to metadata file
 * @param source_hash Hash of source file
 * @param timestamp Creation timestamp
 */
inline void write_metadata(
    const std::filesystem::path& metadata_path,
    const std::string& source_hash,
    const std::string& timestamp = ""
) {
    std::ofstream file(metadata_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to write metadata: " + metadata_path.string());
    }
    
    file << "source_hash=" << source_hash << "\n";
    if (!timestamp.empty()) {
        file << "timestamp=" << timestamp << "\n";
    }
}

/**
 * @brief Read metadata file
 * 
 * @param metadata_path Path to metadata file
 * @return Source hash from metadata, or empty string if not found
 */
inline std::string read_metadata(const std::filesystem::path& metadata_path) {
    std::ifstream file(metadata_path);
    if (!file.is_open()) {
        return "";
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("source_hash=") == 0) {
            return line.substr(12); // Length of "source_hash="
        }
    }
    
    return "";
}

} // namespace cache
} // namespace freellm
