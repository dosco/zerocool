#include "core/bpe_tokenizer.hpp"
#include <nlohmann/json.hpp>
#include <re2/re2.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <queue>
#include <map>
#include <set>

using json = nlohmann::json;

namespace freellm {

// Standard GPT-2/Llama-3 bytes_to_unicode mapping
const std::unordered_map<unsigned char, std::string>& BPETokenizer::get_bytes_to_unicode() {
    static std::unordered_map<unsigned char, std::string> map;
    static bool initialized = false;
    
    if (!initialized) {
        std::vector<int> bs;
        // Include printable ranges
        for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
        for (int i = 161; i <= 172; ++i) bs.push_back(i);
        for (int i = 174; i <= 255; ++i) bs.push_back(i);
        
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        
        for (size_t i = 0; i < bs.size(); ++i) {
            // Encode as UTF-8 string
            int c = cs[i];
            std::string s;
            if (c < 0x80) {
                s += (char)c;
            } else if (c < 0x800) {
                s += (char)(0xC0 | (c >> 6));
                s += (char)(0x80 | (c & 0x3F));
            } else {
                s += (char)(0xE0 | (c >> 12));
                s += (char)(0x80 | ((c >> 6) & 0x3F));
                s += (char)(0x80 | (c & 0x3F));
            }
            map[(unsigned char)bs[i]] = s;
        }
        initialized = true;
    }
    return map;
}

BPETokenizer::BPETokenizer(const std::string& tokenizer_json_path) {
    load_json(tokenizer_json_path);
}

BPETokenizer::~BPETokenizer() = default;

void BPETokenizer::load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open tokenizer file: " + path);
    }
    
    json data;
    f >> data;
    
    // Load vocab
    auto vocab = data["model"]["vocab"];
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        vocab_[it.key()] = it.value();
        id_to_vocab_[it.value()] = it.key();
    }
    
    // Load merges
    auto merges = data["model"]["merges"];
    int rank = 0;
    for (const auto& merge : merges) {
        std::string merge_str;
        if (merge.is_array()) {
            if (merge.size() >= 2) {
                merge_str = merge[0].get<std::string>() + " " + merge[1].get<std::string>();
            }
        } else {
            merge_str = merge.get<std::string>();
        }
        
        if (!merge_str.empty()) {
            merges_[merge_str] = rank++;
        }
    }
    
    // Parse pre_tokenizer pattern
    std::string fallback_pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    
    if (data.contains("pre_tokenizer") && data["pre_tokenizer"].contains("pretokenizers")) {
        for (const auto& pt : data["pre_tokenizer"]["pretokenizers"]) {
            if (pt.contains("type") && pt["type"] == "Split" && pt.contains("pattern") && pt["pattern"].contains("Regex")) {
                fallback_pattern = pt["pattern"]["Regex"];
                break;
            }
        }
    }
    
    // RE2 does not support look-around assertions like (?!\S).
    // The pattern usually contains `\s+(?!\S)` to handle trailing whitespace.
    // We simplify it by removing `(?!\S)` which is a reasonable approximation.
    size_t pos;
    while ((pos = fallback_pattern.find("(?!\\S)")) != std::string::npos) {
        fallback_pattern.replace(pos, 6, ""); // Remove (?!\S)
    }

    regex_ = std::make_unique<re2::RE2>(fallback_pattern);
    if (!regex_->ok()) {
        throw std::runtime_error("Failed to compile regex: " + fallback_pattern);
    }

    // Special tokens
    if (data.contains("added_tokens")) {
        for (const auto& t : data["added_tokens"]) {
            std::string content = t["content"];
            int id = t["id"];
            // vocab_ might already have it, but added_tokens are definitive
            vocab_[content] = id;
            id_to_vocab_[id] = content;
            
            if (content == "<|begin_of_text|>") bos_id_ = id;
            if (content == "<|end_of_text|>") eos_id_ = id;
        }
    }
}

std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) {
    std::vector<std::string> chunks;
    re2::StringPiece input(text);
    re2::StringPiece match;
    
    while (re2::RE2::FindAndConsume(&input, *regex_, &match)) {
        chunks.push_back(std::string(match));
    }
    // Handle remaining? Usually FindAndConsume covers all if the regex is ".*" or similar, 
    // but tiktoken split regexes are designed to match contiguous blocks. 
    // If anything is left, it's usually whitespace or garbage not matched.
    if (!input.empty()) {
        chunks.push_back(std::string(input));
    }
    return chunks;
}

std::vector<std::string> BPETokenizer::byte_encode_chunk(const std::string& text) {
    std::vector<std::string> tokens;
    const auto& b2u = get_bytes_to_unicode();
    for (unsigned char c : text) {
        tokens.push_back(b2u.at(c));
    }
    return tokens;
}

std::vector<int> BPETokenizer::bpe_encode(const std::string& text) {
    auto tokens = byte_encode_chunk(text);
    if (tokens.empty()) return {};

    while (tokens.size() > 1) {
        long best_rank = -1;
        int best_idx = -1;
        std::string best_pair_str;
        
        // Find best pair
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            std::string pair = tokens[i] + " " + tokens[i+1];
            if (merges_.count(pair)) {
                int rank = merges_.at(pair);
                if (best_idx == -1 || rank < best_rank) {
                    best_rank = rank;
                    best_idx = i;
                    best_pair_str = pair;
                }
            }
        }
        
        if (best_idx == -1) break; // No merges possible logic
        
        // Merge
        std::vector<std::string> new_tokens;
        new_tokens.reserve(tokens.size());
        
        // Copy up to best_idx
        for (int i = 0; i < best_idx; ++i) {
            new_tokens.push_back(tokens[i]);
        }
        // Merge best_idx and best_idx+1
        new_tokens.push_back(tokens[best_idx] + tokens[best_idx+1]);
        // Copy rest
        for (size_t i = best_idx + 2; i < tokens.size(); ++i) {
            new_tokens.push_back(tokens[i]);
        }
        tokens = new_tokens;
    }
    
    std::vector<int> ids;
    for (const auto& t : tokens) {
        if (vocab_.count(t)) {
            ids.push_back(vocab_.at(t));
        } else {
            // Handle unknown? byte fallback usually ensures everything is in vocab as single bytes
            // If not found, it's weird.
        }
    }
    return ids;
}

std::vector<int> BPETokenizer::encode(const std::string& text) {
    // 1. Regex split
    auto chunks = pre_tokenize(text);
    
    std::vector<int> all_ids;
    if (bos_id_ != -1) all_ids.push_back(bos_id_);
    
    for (const auto& chunk : chunks) {
        auto chunk_ids = bpe_encode(chunk);
        all_ids.insert(all_ids.end(), chunk_ids.begin(), chunk_ids.end());
    }
    
    return all_ids;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) {
    std::string text;
    for (int id : ids) {
        if (id_to_vocab_.count(id)) {
            text += id_to_vocab_.at(id);
        }
    }
    
    // Reverse bytes_to_unicode mapping
    static std::unordered_map<std::string, unsigned char> u2b;
    static bool u2b_init = false;
    
    if (!u2b_init) {
        const auto& b2u = get_bytes_to_unicode();
        for (const auto& kv : b2u) {
            u2b[kv.second] = kv.first;
        }
        u2b_init = true;
    }
    
    std::string decoded;
    decoded.reserve(text.size());
    
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];
        int len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        
        if (i + len > text.size()) break; // Should not happen with valid UTF-8
        
        std::string uchar = text.substr(i, len);
        
        if (u2b.count(uchar)) {
            decoded += (char)u2b[uchar];
        } else {
            decoded += uchar;
        }
        
        i += len;
    }
    
    return decoded;
}

} // namespace freellm
