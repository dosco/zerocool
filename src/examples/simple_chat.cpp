#include "infra/model_loader.hpp"
#include "core/generation_engine.hpp"
#include "core/paged_kv_cache.hpp"
#include "core/llm_model.hpp"
#include "core/generation.hpp"
#include "core/cpu_batch_generation.hpp"
#include "infra/safetensors_loader.hpp"
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
    bool force_cpu = false;
    bool debug_moe = false;
    bool dump_activations = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--int8") {
            use_int8_kv = true;
        } else if (arg == "--cpu") {
            force_cpu = true;
        } else if (arg == "--debug-moe") {
            debug_moe = true;
        } else if (arg == "--dump-activations") {
            dump_activations = true;
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
    std::string model_name = "Unknown Model";

    ui::print_section_header("INITIALIZING NEURAL CORE");
    
    // Try to auto-load config from config.json
    bool config_loaded = false;
    if (std::filesystem::is_directory(model_path)) {
        std::string config_json_path = model_path + "/config.json";
        if (std::filesystem::exists(config_json_path)) {
            try {
                config = load_model_config_from_json(model_path);
                config_loaded = true;
                
                // Determine model name from config
                if (config.num_experts > 0) {
                    model_name = "MoE Model (" + std::to_string(config.num_experts) + " experts)";
                } else {
                    model_name = "Dense Model";
                }
            } catch (const std::exception& e) {
                ui::print_warning("Failed to load config.json: " + std::string(e.what()));
                ui::print_info("Falling back to path-based detection...");
            }
        }
    }
    
    // Fallback: Path-based detection for tokenizer
    if (std::filesystem::is_directory(model_path)) {
        // Check for BPE tokenizer (tokenizer.json)
        std::string bpe_path = model_path + "/tokenizer.json";
        std::string sp_path = model_path + "/tokenizer.model";
        
        if (std::filesystem::exists(bpe_path)) {
            tokenizer_path = bpe_path;
            ui::print_loading("Loading BPE Tokenizer...");
            tokenizer = std::make_unique<BPETokenizer>(tokenizer_path);
            ui::print_success("Tokenizer ready: " + tokenizer_path);
        } else if (std::filesystem::exists(sp_path)) {
            tokenizer_path = sp_path;
            ui::print_loading("Loading SentencePiece Tokenizer...");
            tokenizer = std::make_unique<SentencePieceTokenizer>(tokenizer_path);
            ui::print_success("Tokenizer ready: " + tokenizer_path);
        } else {
            ui::print_error("No tokenizer found in: " + model_path);
            return 1;
        }
    } else {
        ui::print_error("Model path is not a directory: " + model_path);
        return 1;
    }
    
    // If config wasn't loaded, use fallbacks based on path
    if (!config_loaded) {
        ui::print_warning("No config.json found, using path-based detection");
        
        if (model_path.find("Meta-Llama-3") != std::string::npos || model_path.find("Llama-3") != std::string::npos) {
            model_name = "Meta Llama 3 8B";
            config.d_model = 4096;
            config.d_ff = 14336;
            config.n_layers = 32;
            config.n_heads = 32;
            config.n_kv_heads = 8;
            config.vocab_size = 128256;
            config.max_seq_len = 8192;
            config.rope_theta = 500000.0f;
        } else if (model_path.find("OLMoE") != std::string::npos) {
            model_name = "OLMoE 1B-7B";
            config.d_model = 2048;
            config.d_ff = 1024;
            config.n_layers = 16;
            config.n_heads = 16;
            config.n_kv_heads = 16;
            config.vocab_size = 50304;
            config.max_seq_len = 4096;
            config.num_experts = 64;
            config.num_experts_per_token = 8;
        } else {
            model_name = "TinyLlama 1.1B";
            config.d_model = 2048;
            config.d_ff = 5632;
            config.n_layers = 22;
            config.n_heads = 32;
            config.n_kv_heads = 4;
            config.vocab_size = 32000;
            config.max_seq_len = 2048;
        }
    }

    // Print model config
    ui::print_model_config(model_name, config.d_model, config.n_layers, 
                           config.n_heads, config.vocab_size, config.max_seq_len);

    // Check if forced CPU - use CPU path (MoE models now use Metal by default)
    if (force_cpu) {
        if (config.num_experts > 0) {
            ui::print_info("MoE Model Detected - Using CPU inference path (--cpu flag)");
        } else {
            ui::print_info("Forced CPU inference path for Dense Model");
        }
        ui::print_section_header("LOADING NEURAL WEIGHTS (CPU)");
        
        // Create LLMModel for MoE
        quant::QuantConfig quant_config;
        quant_config.default_type = quant::QuantType::NONE; // Keep FP32 for MoE initially
        
        LLMModel model(config, quant_config);
        
        // Load weights
        ui::print_loading("Loading model weights...");
        
        // Quant resolver: Don't quantize router, keep experts in FP32 for now
        auto quant_resolver = [](const std::string& name) -> std::optional<quant::QuantType> {
            // Keep all MoE weights in FP32 for correctness testing
            if (name.find("mlp.gate.weight") != std::string::npos) return std::nullopt;
            if (name.find("mlp.experts.") != std::string::npos) return std::nullopt;
            if (name == "token_embedding" || name.find("norm") != std::string::npos || name == "lm_head") {
                return std::nullopt;
            }
            // Could quantize attention later
            return std::nullopt;
        };
        
        auto [fp32_weights, quant_weights] = load_model_from_path(model_path, quant_resolver);

        ui::print_success("Loaded " + std::to_string(fp32_weights.size()) + " F32 tensors");

        // DEBUG: Check layer 1 gate right after loading, before load_weights
        {
            auto it = fp32_weights.find("layers.1.mlp.gate.weight");
            if (it != fp32_weights.end()) {
                for (size_t i = 0; i < 5; ++i) std::cout << it->second.data()[i] << " ";
                std::cout << "\n";
            }
        }

        model.load_weights(fp32_weights, nullptr);

        // Enable MoE debug mode if requested
        if (debug_moe) {
            ui::print_info("MoE debug mode enabled");
            for (auto& block : model.blocks()) {
                if (block->moe_layer()) {
                    block->moe_layer()->set_debug(true);
                }
            }
        }

        // Enable activation dumping if requested
        if (dump_activations) {
            ui::print_info("Activation dumping enabled (compare with scripts/compare_hf.py)");
            model.set_dump_activations(true);
            // Also enable attention dumping for layer 0
            if (!model.blocks().empty()) {
                model.blocks()[0]->attention().set_dump(true);
            }
            // Enable MoE debug for layer 0
            if (!model.blocks().empty() && model.blocks()[0]->moe_layer()) {
                model.blocks()[0]->moe_layer()->set_debug(true);
            }
        }

        ui::print_section_header("PREPARING INFERENCE BATCH");

        // Sanity Check: Print some weight statistics
        auto& emb = model.token_embedding();
        for(int k=0; k<5; ++k) std::cout << emb.at({100, (size_t)k}) << " ";
        std::cout << "\n";


        std::vector<std::string> prompts = {
            "The capital of France is",
            "Paris is the capital of",
            "The quick brown fox jumps over"
        };
        
        for (size_t i = 0; i < prompts.size(); ++i) {
            auto tokens = tokenizer->encode(prompts[i]);
            std::cout << ui::colors::DIM << "  │ " << ui::colors::RESET 
                      << ui::colors::BRIGHT_CYAN << "[" << i << "] " << ui::colors::RESET
                      << ui::colors::BRIGHT_WHITE << prompts[i] << ui::colors::RESET
                      << ui::colors::DIM << " (" << tokens.size() << " tokens)" << ui::colors::RESET << "\n";

        }
        
        ui::print_inference_header();

        // Encode all prompts
        std::vector<std::vector<int>> prompt_tokens_batch;
        for (const auto& prompt : prompts) {
            prompt_tokens_batch.push_back(tokenizer->encode(prompt));
        }

        // Track printed output per batch
        std::vector<std::vector<int>> batch_tokens(prompts.size());
        std::vector<size_t> printed_lens(prompts.size(), 0);
        size_t total_tokens = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Configure batched generation
        BatchGenerationConfig batch_config;
        batch_config.method = SamplingMethod::TopP;
        batch_config.top_p = 0.9f;
        batch_config.temperature = 0.7f;
        batch_config.max_new_tokens = 20;

        // Callback for each generated token - produces interleaved output
        batch_config.token_callback = [&](size_t batch_idx, int token, size_t step) {
            (void)step;
            batch_tokens[batch_idx].push_back(token);
            total_tokens++;

            std::string text = tokenizer->decode(batch_tokens[batch_idx]);
            if (text.length() > printed_lens[batch_idx]) {
                std::cout << ui::colors::BRIGHT_CYAN << "│ " << ui::colors::RESET
                          << ui::colors::BRIGHT_MAGENTA << "[" << batch_idx << "] " << ui::colors::RESET
                          << ui::colors::DIM << "(" << token << ") " << ui::colors::RESET
                          << ui::colors::BRIGHT_WHITE << text.substr(printed_lens[batch_idx]) << ui::colors::RESET << "\n";
                printed_lens[batch_idx] = text.length();
            }
        };

        // Run batched generation
        generate_batch(model, prompt_tokens_batch, batch_config);

        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time).count();
        double tokens_per_sec = total_tokens / duration;

        ui::print_inference_footer();
        ui::print_inference_complete(tokens_per_sec);

        return 0;
    }

    // Metal GenerationEngine path (Dense and MoE models)
    if (config.num_experts > 0) {
        ui::print_info("MoE Model Detected - Using Metal GPU inference with " +
                      std::to_string(config.num_experts) + " experts (top-" +
                      std::to_string(config.num_experts_per_token) + ")");
    }

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

