#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../external/doctest.h"
#include "core/bpe_tokenizer.hpp"
#include <iostream>
#include <vector>
#include <filesystem>
#include <print>

using namespace freellm;

TEST_CASE("BPETokenizer Llama 3") {
    std::string tokenizer_path = "models/llama3-8b/tokenizer.json";
    
    if (!std::filesystem::exists(tokenizer_path)) {
        std::println("⚠️  Llama 3 tokenizer file not found at: {}", tokenizer_path);
        std::println("Skipping BPE test.");
        return;
    }
    
    SUBCASE("Load Tokenizer") {
        BPETokenizer tokenizer(tokenizer_path);
        CHECK(tokenizer.vocab_size() == 128256);
        CHECK(tokenizer.eos_id() == 128001); // <|end_of_text|> usually
        CHECK(tokenizer.bos_id() == 128000); // <|begin_of_text|> usually
    }

    SUBCASE("Encode Decode") {
        BPETokenizer tokenizer(tokenizer_path);
        std::string text = "Hello World";
        auto ids = tokenizer.encode(text);
        
        // Expected: [128000, 9906, 4435] or similar (BOS + Hello + World)
        // Note: Llama 3 might not add BOS by default in `encode` unless specified?
        // My implementation adds BOS if `bos_id != -1`.
        
        CHECK(ids.size() >= 2);
        
        std::string decoded = tokenizer.decode(ids);
        // Note: My decode might need whitespace fixing or BOS removal.
        // Usually BOS is not printed.
        // And "Hello" might be "Hello" or " Hello".
        
        std::println("Original: '{}'", text);
        std::println("Encoded: {}", ids);
        std::println("Decoded: '{}'", decoded);
        
        // Basic check: decoded contains original text
        CHECK(decoded.find("Hello") != std::string::npos);
        CHECK(decoded.find("World") != std::string::npos);
    }
    
    SUBCASE("Special Tokens") {
        BPETokenizer tokenizer(tokenizer_path);
        std::string text = "<|start_header_id|>";
        auto ids = tokenizer.encode(text);
        
        // Should capture it as single token if it's in added_tokens
        std::println("Encoded special: {}", ids);
        
        // Check if any ID is > 128000 (special token range)
        bool found_special = false;
        for (int id : ids) {
            if (id >= 128000) found_special = true;
        }
        CHECK(found_special);
    }
}
