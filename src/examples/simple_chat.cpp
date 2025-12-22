#include "core/generation_engine.hpp"
#include "core/bpe_tokenizer.hpp"
#include "utils/terminal_ui.hpp"
#include <iostream>
#include <print>
#include <thread>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>

using namespace freellm;

int main(int argc, char** argv) {
    // Print the sci-fi banner
    ui::print_banner();
    
    if (argc < 2) {
        ui::print_error("Usage: " + std::string(argv[0]) + " <model_path> [--int8]");
        return 1;
    }
    
    std::string model_path;
    bool use_int8_kv = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--int8") {
            use_int8_kv = true;
        } else {
            model_path = arg;
        }
    }
    
    if (model_path.empty()) {
        ui::print_error("Usage: " + std::string(argv[0]) + " <model_path> [--int8]");
        return 1;
    }

    // Tokenizer Selection
    std::unique_ptr<ITokenizer> tokenizer;
    std::string tokenizer_path;
    ModelConfig config;
    std::string model_name;

    ui::print_section_header("INITIALIZING NEURAL CORE");

    if (model_path.find("Meta-Llama-3-8B") != std::string::npos || model_path.find("Llama-3") != std::string::npos) {
        model_name = "Meta Llama 3 8B";
        ui::print_info("Detected: " + model_name);
        
        config.d_model = 4096;
        config.d_ff = 14336;
        config.n_layers = 32;
        config.n_heads = 32;
        config.n_kv_heads = 8;
        config.vocab_size = 128256;
        config.norm_eps = 1e-5;
        config.max_seq_len = 8192;
        config.rope_theta = 500000.0f;
        
        if (std::filesystem::is_directory(model_path)) {
            tokenizer_path = model_path + "/tokenizer.json";
        } else {
            tokenizer_path = "models/Meta-Llama-3-8B/tokenizer.json"; 
        }
        ui::print_loading("Loading BPE Tokenizer...");
        tokenizer = std::make_unique<BPETokenizer>(tokenizer_path);
        ui::print_success("Tokenizer ready: " + tokenizer_path);
    } else {
        model_name = "TinyLlama 1.1B";
        ui::print_info("Detected: " + model_name);
        
        config.d_model = 2048;
        config.d_ff = 5632;
        config.n_layers = 22;
        config.n_heads = 32;
        config.n_kv_heads = 4;
        config.vocab_size = 32000;
        config.norm_eps = 1e-5;
        config.max_seq_len = 2048;
        config.rope_theta = 10000.0f;
        
        if (std::filesystem::is_directory(model_path)) {
             tokenizer_path = model_path + "/tokenizer.model";
        } else {
             tokenizer_path = "models/tinyllama/tokenizer.model";
        }
        ui::print_loading("Loading SentencePiece Tokenizer...");
        tokenizer = std::make_unique<SentencePieceTokenizer>(tokenizer_path);
        ui::print_success("Tokenizer ready: " + tokenizer_path);
    }

    // Print model config
    ui::print_model_config(model_name, config.d_model, config.n_layers, 
                           config.n_heads, config.vocab_size, config.max_seq_len);

    KVCacheConfig kv_config;
    kv_config.block_size = 16;
    kv_config.max_num_blocks = 16384; 
    kv_config.n_kv_heads = config.n_kv_heads;
    kv_config.head_dim = config.d_model / config.n_heads;
    if (use_int8_kv) {
        kv_config.data_type = KvCacheDataType::INT8;
        ui::print_info("KV Cache: INT8 Quantized");
    } else {
        ui::print_info("KV Cache: FP32");
    }

    // Initialize Engine
    ui::print_section_header("LOADING NEURAL WEIGHTS");
    ui::print_loading("Initializing Metal backend...");
    GenerationEngine engine(model_path, config, kv_config);
    
    // Add requests
    ui::print_section_header("PREPARING INFERENCE BATCH");
    std::vector<std::string> prompts = {
        "The capital of France is",
        "Paris is the capital of",
        "The quick brown fox jumps over"
    };

    for (size_t i = 0; i < prompts.size(); ++i) {
        auto tokens = tokenizer->encode(prompts[i]);
        engine.add_request(tokens);
        std::cout << ui::colors::DIM << "  │ " << ui::colors::RESET 
                  << ui::colors::BRIGHT_CYAN << "[" << i << "] " << ui::colors::RESET
                  << ui::colors::BRIGHT_WHITE << prompts[i] << ui::colors::RESET
                  << ui::colors::DIM << " (" << tokens.size() << " tokens)" << ui::colors::RESET << "\n";
    }
    
    // Inference
    ui::print_inference_header();
    
    // Track printed length for each sequence
    std::vector<size_t> printed_lens(prompts.size(), 0);
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t total_tokens = 0;
    
    // Loop
    while (engine.step()) {
        const auto& sequences = engine.active_sequences();
        for (size_t i = 0; i < sequences.size(); ++i) {
            auto* seq = sequences[i];
            const auto& tokens = seq->tokens();
            if (tokens.size() > printed_lens[i]) {
                size_t new_tokens = tokens.size() - printed_lens[i];
                total_tokens += new_tokens;
                
                std::string text = tokenizer->decode(std::vector<int>(tokens.begin() + printed_lens[i], tokens.end()));
                
                // Debug: Print token IDs
                std::string token_ids_str;
                for (size_t k = printed_lens[i]; k < tokens.size(); ++k) {
                    token_ids_str += std::to_string(tokens[k]) + " ";
                }

                // Clean up the output - replace control characters
                std::string clean_text;
                for (char c : text) {
                    if (c == '\n') clean_text += " ";
                    else if (c >= 32 || c == '\t') clean_text += c;
                }
                
                std::cout << ui::colors::BRIGHT_CYAN << "│ " << ui::colors::RESET
                          << ui::colors::BRIGHT_MAGENTA << "[" << i << "] " << ui::colors::RESET
                          << ui::colors::DIM << "(" << token_ids_str << ") " << ui::colors::RESET
                          << ui::colors::BRIGHT_WHITE << clean_text << ui::colors::RESET << "\n";
                printed_lens[i] = tokens.size();
            }
        }
        std::cout << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    double tokens_per_sec = total_tokens / duration;
    
    ui::print_inference_footer();
    ui::print_inference_complete(tokens_per_sec);
    
    return 0;
}
