#include <print>
#include <chrono>
#include <format>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <random>

#include "core/model_config.hpp"
#include "infra/safetensors_loader.hpp"
#include "core/llm_model.hpp"
#include "core/generation.hpp"
#include "core/tokenizer.hpp"
#include "kernels/quantiz/quant_config.hpp"

using namespace freellm;

/**
 * @brief Load and test TinyLLaMA model with text generation (Original Test)
 */
void test_load_tinyllama() {
    std::println("\n========================================");
    std::println("Testing Real TinyLLaMA Weight Loading");
    std::println("========================================\n");

    // Check if model file exists
    std::string model_path = "models/tinyllama/model.safetensors";

    if (!std::filesystem::exists(model_path)) {
        std::println("⚠️  TinyLLaMA weights not found at: {}", model_path);
        std::println("\nTo download the model, run:");
        std::println("  bash scripts/download_tinyllama.sh\n");
        std::println("Skipping this test.\n");
        return;
    }

    std::println("✓ Found model file at: {}\n", model_path);

    // Create TinyLLaMA configuration
    ModelConfig config = ModelConfig::tinyllama_1_1b();
    std::println("Creating TinyLLaMA model:");
    std::println("  Vocab size: {}", config.vocab_size);
    std::println("  Layers: {}", config.n_layers);
    std::println("  d_model: {}", config.d_model);
    std::println("  n_heads: {}", config.n_heads);
    std::println("  d_ff: {}", config.d_ff);
    std::println("");

    // Configure quantization
    quant::QuantConfig quant_config;
    quant_config.default_type = quant::QuantType::Q4_K;
    quant_config.attention = quant::QuantType::Q8_0;
    quant_config.feed_forward = quant::QuantType::Q4_K;
    quant_config.lm_head = quant::QuantType::Q4_K;

    std::println("Quantization configuration:");
    std::println("  Default: {}", quant::quant_type_name(quant_config.default_type));
    std::println("  Attention: {}", quant::quant_type_name(quant_config.resolve_attention()));
    std::println("  Feed-forward: {}", quant::quant_type_name(quant_config.resolve_feed_forward()));
    std::println("  LM head: {}", quant::quant_type_name(quant_config.resolve_lm_head()));
    std::println("");

    // Create model with quantization enabled
    LLMModel model(config, quant_config);

    // Load weights from safetensors with parallel quantization
    std::println("Loading weights from safetensors file (with parallel quantization)...");
    auto load_start = std::chrono::high_resolution_clock::now();

    WeightMap weights;
    std::unordered_map<std::string, QuantizedTensor> quantized_weights;

    try {
        auto quant_resolver = [&quant_config](const std::string& tensor_name) -> std::optional<quant::QuantType> {
            if (tensor_name == "token_embedding" || tensor_name == "final_norm_weight") return std::nullopt;
            if (tensor_name == "lm_head") return quant_config.resolve_lm_head();
            if (tensor_name.find(".attn.W_") != std::string::npos) return quant_config.resolve_attention();
            if (tensor_name.find(".ffn.W_") != std::string::npos) return quant_config.resolve_feed_forward();
            if (tensor_name.find("_norm_weight") != std::string::npos) return std::nullopt;
            return std::nullopt;
        };

        auto [f32_weights, quant_weights] = load_safetensors_parallel_quantized(model_path, quant_resolver);
        weights = std::move(f32_weights);
        quantized_weights = std::move(quant_weights);

    } catch (const std::exception& e) {
        std::println(stderr, "Error loading weights: {}", e.what());
        std::println("Falling back to sequential loader...");
        weights = load_safetensors(model_path);
    }

    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);

    std::println("Weight loading + quantization took {} ms\n", load_duration.count());

    model.load_weights(weights, &quantized_weights);

    // Test generation with real weights
    std::println("\n========================================");
    std::println("Testing Generation with Real Weights");
    std::println("========================================\n");

    std::println("Initializing tokenizer...");
    std::string tokenizer_path = "models/tinyllama/tokenizer.model";

    if (!std::filesystem::exists(tokenizer_path)) {
        std::println("⚠️  Tokenizer file not found. Using fallback token IDs.");
        return;
    }

    Tokenizer tokenizer(tokenizer_path);
    std::println("✓ Tokenizer initialized (vocab size: {})\n", tokenizer.vocab_size());

    std::string prompt_text = "Jack and Jill went";
    std::println("Prompt text: \"{}\"", prompt_text);

    std::vector<int> prompt = tokenizer.encode(prompt_text);
    std::println("Encoded to {} tokens: [{}]", prompt.size(), std::format("{}", prompt[0]));
    for (size_t i = 1; i < prompt.size(); ++i) {
        std::print(", {}", prompt[i]);
    }
    std::println("]\n");

    std::println("Generating 10 tokens with top-p sampling (p=0.95, T=0.8)...");
    std::println("(Tokens will be decoded as they are generated)\n");

    auto gen_start = std::chrono::high_resolution_clock::now();

    GenerationConfig gen_config;
    gen_config.method = SamplingMethod::TopP;
    gen_config.top_p = 0.95f;
    gen_config.temperature = 0.8f;
    gen_config.max_new_tokens = 10;
    gen_config.token_callback = [&tokenizer](int token, size_t step, double elapsed_ms) {
        std::string token_text = tokenizer.decode({token});
        std::println("  Token {}: {} \"{}\" ({:.2f}s)", step, token, token_text, elapsed_ms / 1000.0);
    };

    std::vector<int> generated = generate(model, prompt, gen_config);

    auto gen_end = std::chrono::high_resolution_clock::now();
    auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start);

    std::string generated_text = tokenizer.decode(generated);

    std::println("\n========================================");
    std::println("Generated Text");
    std::println("========================================");
    std::println("Prompt: \"{}\"", prompt_text);
    std::println("Output: \"{}\"", generated_text);
    std::println("========================================\n");

    std::println("Generation stats:");
    std::println("  Total tokens: {}", generated.size());
    std::println("  Total time: {} ms", gen_duration.count());
    std::println("  Tokens/sec: {:.1f}", 1000.0 * generated.size() / gen_duration.count());
}

/**
 * @brief Run Interactive Chat Interface (REPL)
 */
void run_repl(LLMModel& model, Tokenizer& tokenizer) {
    std::println("\n========================================");
    std::println("Interactive Chat Interface (REPL)");
    std::println("========================================");
    std::println("Commands:");
    std::println("  /exit   - Quit the REPL");
    std::println("  /reset  - Clear conversation history");
    std::println("========================================\n");

    std::vector<int> history;
    std::string input_line;
    
    while (true) {
        std::print("> ");
        std::cout.flush();

        if (!std::getline(std::cin, input_line)) {
            break; // EOF
        }

        if (input_line.empty()) continue;

        if (input_line == "/exit") {
            std::println("Goodbye!");
            break;
        }

        if (input_line == "/reset") {
            history.clear();
            model.reset_kv_cache();
            std::println("Conversation history cleared.");
            continue;
        }

        std::string user_input = "User: " + input_line + "\nAssistant:";
        std::vector<int> input_tokens = tokenizer.encode(user_input);
        history.insert(history.end(), input_tokens.begin(), input_tokens.end());

        if (history.size() >= model.config().max_seq_len) {
            std::println("⚠️  Context limit reached. Clearing history.");
            history.clear();
            model.reset_kv_cache();
            history = input_tokens;
        }

        GenerationConfig gen_config;
        gen_config.method = SamplingMethod::TopP;
        gen_config.top_p = 0.95f;
        gen_config.temperature = 0.7f;
        gen_config.max_new_tokens = 200;
        
        // Use model's EOS token if available, otherwise fallback to newline
        // TinyLLaMA / LLaMA usually uses ID 2 for EOS (</s>)
        gen_config.eos_token_id = tokenizer.eos_id();
        
        // State for streaming decoder
        // We decode the *new* sequence accumulated so far to handle spaces correctly
        // (SentencePiece treats spaces as part of tokens, e.g. "_word")
        std::vector<int> current_response_tokens;
        size_t printed_length = 0;

        gen_config.token_callback = [&tokenizer, &current_response_tokens, &printed_length](int token, size_t step, double elapsed_ms) {
            (void)step;       // Mark unused
            (void)elapsed_ms; // Mark unused
            
            current_response_tokens.push_back(token);
            std::string full_text = tokenizer.decode(current_response_tokens);
            
            // Print only the new characters
            if (full_text.length() > printed_length) {
                std::string new_text = full_text.substr(printed_length);
                std::print("{}", new_text);
                std::cout.flush();
                printed_length = full_text.length();
            }
        };

        std::vector<int> full_sequence = generate(model, history, gen_config);
        history = full_sequence;
        std::println(""); 
    }
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    std::println("Usage: {} [OPTIONS]", program_name);
    std::println("\nOptions:");
    std::println("  --help, -h    Show this help message");
    std::println("  --test        Run the original Jack and Jill test (non-interactive)");
    std::println("\nDefault behavior (no args):");
    std::println("  Runs Interactive Chat Interface (REPL)");
}

int main(int argc, char* argv[]) {
    std::println("========================================");
    std::println("FreeLLM - Fast CPU LLM Inference Engine");
    std::println("========================================\n");

    try {
        bool show_help = false;
        bool run_test_mode = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                show_help = true;
            } else if (arg == "--test") {
                run_test_mode = true;
            } else {
                std::println(stderr, "Unknown option: {}", arg);
                print_usage(argv[0]);
                return 1;
            }
        }

        if (show_help) {
            print_usage(argv[0]);
            return 0;
        }

        if (run_test_mode) {
            std::println("Running in TEST mode (Jack and Jill test)...");
            test_load_tinyllama();
        } else {
            // Default: Run REPL
            // We need to load the model here to pass it to run_repl
            
            std::string model_path = "models/tinyllama/model.safetensors";
            if (!std::filesystem::exists(model_path)) {
                std::println("⚠️  TinyLLaMA weights not found at: {}", model_path);
                std::println("\nTo download the model, run:");
                std::println("  bash scripts/download_tinyllama.sh\n");
                return 1;
            }

            std::string tokenizer_path = "models/tinyllama/tokenizer.model";
            if (!std::filesystem::exists(tokenizer_path)) {
                std::println("⚠️  Tokenizer file not found at: {}", tokenizer_path);
                std::println("\nTo download the tokenizer, run:");
                std::println("  bash scripts/download_tinyllama.sh\n");
                return 1;
            }

            ModelConfig config = ModelConfig::tinyllama_1_1b();
            
            quant::QuantConfig quant_config;
            quant_config.default_type = quant::QuantType::Q4_K;
            quant_config.attention = quant::QuantType::Q8_0;
            quant_config.feed_forward = quant::QuantType::Q4_K;
            quant_config.lm_head = quant::QuantType::Q4_K;

            std::println("Loading model...");
            LLMModel model(config, quant_config);

            std::println("Loading weights...");
            WeightMap weights;
            std::unordered_map<std::string, QuantizedTensor> quantized_weights;

            auto quant_resolver = [&quant_config](const std::string& tensor_name) -> std::optional<quant::QuantType> {
                if (tensor_name == "token_embedding" || tensor_name == "final_norm_weight") return std::nullopt;
                if (tensor_name == "lm_head") return quant_config.resolve_lm_head();
                if (tensor_name.find(".attn.W_") != std::string::npos) return quant_config.resolve_attention();
                if (tensor_name.find(".ffn.W_") != std::string::npos) return quant_config.resolve_feed_forward();
                return std::nullopt;
            };

            auto [f32_weights, quant_weights] = load_safetensors_parallel_quantized(model_path, quant_resolver);
            weights = std::move(f32_weights);
            quantized_weights = std::move(quant_weights);

            model.load_weights(weights, &quantized_weights);

            std::println("Initializing tokenizer...");
            Tokenizer tokenizer(tokenizer_path);

            run_repl(model, tokenizer);
        }

    } catch (const std::exception& e) {
        std::println(stderr, "\nError: {}", e.what());
        return 1;
    }

    return 0;
}

