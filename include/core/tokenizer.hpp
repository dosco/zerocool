#pragma once

#include <sentencepiece_processor.h>
#include <string>
#include <vector>
#include <stdexcept>

namespace freellm {

// Abstract Base Class
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& ids) = 0;
    virtual size_t vocab_size() const = 0;
    virtual int eos_id() const = 0;
    // virtual int bos_id() const = 0; // SentencePiece usually implied/configurable, BPE has it explicit.
};

/**
 * @brief SentencePiece Tokenizer wrapper for TinyLLaMA
 */
class SentencePieceTokenizer : public ITokenizer {
public:
    SentencePieceTokenizer(const std::string& model_path) {
        const auto status = processor_.Load(model_path);
        if (!status.ok()) {
            throw std::runtime_error("Failed to load SentencePiece model: " +
                                   std::string(status.message()));
        }
    }

    std::vector<int> encode(const std::string& text) override {
        std::vector<int> ids;
        const auto status = processor_.Encode(text, &ids);
        if (!status.ok()) {
            throw std::runtime_error("Failed to encode text: " +
                                   std::string(status.message()));
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) override {
        std::string text;
        const auto status = processor_.Decode(ids, &text);
        if (!status.ok()) {
            throw std::runtime_error("Failed to decode tokens: " +
                                   std::string(status.message()));
        }
        return text;
    }

    size_t vocab_size() const override {
        return processor_.GetPieceSize();
    }

    int eos_id() const override {
        return processor_.eos_id();
    }

private:
    sentencepiece::SentencePieceProcessor processor_;
};

// Typedef for backward compatibility if needed, but we prefer explicit usage
// using Tokenizer = SentencePieceTokenizer;

} // namespace freellm
