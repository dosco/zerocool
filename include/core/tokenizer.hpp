#pragma once

#include <sentencepiece_processor.h>
#include <string>
#include <vector>
#include <stdexcept>

namespace freellm {

/**
 * @brief SentencePiece Tokenizer wrapper for TinyLLaMA
 *
 * TinyLLaMA uses a SentencePiece BPE tokenizer, which is different from GPT-2 BPE:
 * - Uses ▁ (U+2581) to represent spaces
 * - Uses <0xNN> format for byte fallback tokens
 * - Different merge algorithm
 *
 * This class wraps the SentencePiece library for proper tokenization.
 */
class Tokenizer {
public:
    /**
     * @brief Initialize tokenizer with tokenizer.model file
     *
     * @param model_path Path to tokenizer.model file (SentencePiece format)
     */
    Tokenizer(const std::string& model_path) {
        const auto status = processor_.Load(model_path);
        if (!status.ok()) {
            throw std::runtime_error("Failed to load SentencePiece model: " +
                                   std::string(status.message()));
        }
    }

    /**
     * @brief Encode text to token IDs
     */
    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        const auto status = processor_.Encode(text, &ids);
        if (!status.ok()) {
            throw std::runtime_error("Failed to encode text: " +
                                   std::string(status.message()));
        }
        return ids;
    }

    /**
     * @brief Decode token IDs back to text
     */
    std::string decode(const std::vector<int>& ids) {
        std::string text;
        const auto status = processor_.Decode(ids, &text);
        if (!status.ok()) {
            throw std::runtime_error("Failed to decode tokens: " +
                                   std::string(status.message()));
        }
        return text;
    }

    /**
     * @brief Get vocabulary size
     */
    size_t vocab_size() const {
        return processor_.GetPieceSize();
    }

    /**
     * @brief Get EOS token ID
     */
    int eos_id() const {
        return processor_.eos_id();
    }

private:
    sentencepiece::SentencePieceProcessor processor_;
};

} // namespace freellm
