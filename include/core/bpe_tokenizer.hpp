#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// Forward declarations to avoid exposing implementation details in header
namespace re2 { class RE2; }

#include "core/tokenizer.hpp" // For ITokenizer

namespace freellm {

class BPETokenizer : public ITokenizer {
public:
    BPETokenizer(const std::string& tokenizer_json_path);
    ~BPETokenizer();

    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& ids) override;

    size_t vocab_size() const override { return vocab_.size(); }
    int eos_id() const override { return eos_id_; }
    int bos_id() const { return bos_id_; } // Not in interface yet, but useful

private:
    void load_json(const std::string& path);
    std::vector<std::string> pre_tokenize(const std::string& text);
    std::vector<std::string> byte_encode_chunk(const std::string& text);
    std::vector<int> bpe_encode(const std::string& text);

    // Python's bytes_to_unicode logic
    static const std::unordered_map<unsigned char, std::string>& get_bytes_to_unicode();

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_vocab_;
    std::unordered_map<std::string, int> merges_; // "pair" -> rank
    
    std::unique_ptr<re2::RE2> regex_;
    
    int bos_id_ = -1;
    int eos_id_ = -1;
};

} // namespace freellm
