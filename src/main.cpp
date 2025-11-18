#include <print>
#include <chrono>
#include <format>
#include <filesystem>
#include "model_config.hpp"
#include "safetensors_loader.hpp"
#include "llm_model.hpp"
#include "generation.hpp"
#include "tokenizer.hpp"
#include "quantiz/quant_config.hpp"

using namespace freellm;

/**
 * @brief Load and test TinyLLaMA model with text generation
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
    // Use Q4_K for most layers (good compression, ~4.5 bits per weight)
    // Use Q8_0 for attention layers (better quality, ~8.5 bits per weight)
    quant::QuantConfig quant_config;
    quant_config.default_type = quant::QuantType::Q4_K;
    quant_config.attention = quant::QuantType::Q8_0;  // Higher quality for attention
    quant_config.feed_forward = quant::QuantType::Q4_K;  // More compression for FFN
    quant_config.lm_head = quant::QuantType::Q4_K;  // Compress output layer too


    std::println("Quantization configuration:");
    std::println("  Default: {}", quant::quant_type_name(quant_config.default_type));
    std::println("  Attention: {}", quant::quant_type_name(quant_config.resolve_attention()));
    std::println("  Feed-forward: {}", quant::quant_type_name(quant_config.resolve_feed_forward()));
    std::println("  LM head: {}", quant::quant_type_name(quant_config.resolve_lm_head()));
    std::println("");

    // Create model with quantization enabled
    LLMModel model(config, quant_config);
    // LLMModel model(config);

    // Load weights from safetensors
    std::println("Loading weights from safetensors file...");
    auto load_start = std::chrono::high_resolution_clock::now();

    WeightMap weights = load_safetensors(model_path);

    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);

    std::println("Weight loading took {} ms\n", load_duration.count());

    // Load weights into model
    model.load_weights(weights);

    // Test generation with real weights
    std::println("\n========================================");
    std::println("Testing Generation with Real Weights");
    std::println("========================================\n");

    // Initialize tokenizer
    std::println("Initializing tokenizer...");
    std::string tokenizer_path = "models/tinyllama/tokenizer.model";

    if (!std::filesystem::exists(tokenizer_path)) {
        std::println("⚠️  Tokenizer file not found. Using fallback token IDs.");
        std::println("\nTo use the tokenizer, run:");
        std::println("  bash scripts/download_tinyllama.sh");
        std::println("\nThis will download tokenizer.model to: {}\n", tokenizer_path);

        // Fallback to hardcoded tokens
        std::vector<int> prompt = {1, 15043};  // BOS token + approximate "Hello"
        std::println("Prompt tokens: [{}]", std::format("{}", prompt[0]));
        for (size_t i = 1; i < prompt.size(); ++i) {
            std::print(", {}", prompt[i]);
        }
        std::println("]");
        std::println("(Note: Without tokenizer, output won't be readable text)\n");

        // Continue with generation using fallback tokens...
        auto gen_start = std::chrono::high_resolution_clock::now();
        GenerationConfig gen_config;
        gen_config.method = SamplingMethod::TopP;
        gen_config.top_p = 0.95f;
        gen_config.temperature = 0.8f;
        gen_config.max_new_tokens = 10;
        gen_config.token_callback = [](int token, size_t step, double elapsed_ms) {
            std::println("  Token {}: {} ({:.2f}s)", step, token, elapsed_ms / 1000.0);
        };
        std::vector<int> generated = generate(model, prompt, gen_config);
        auto gen_end = std::chrono::high_resolution_clock::now();
        auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start);
        std::println("\nGenerated token IDs: [{}]", std::format("{}", generated[0]));
        for (size_t i = 1; i < generated.size(); ++i) {
            std::print(", {}", generated[i]);
        }
        std::println("]");
        std::println("\nGeneration stats:");
        std::println("  Total time: {} ms", gen_duration.count());
        std::println("  Tokens/sec: {:.1f}", 1000.0 * generated.size() / gen_duration.count());
        return;
    }

    Tokenizer tokenizer(tokenizer_path);
    std::println("✓ Tokenizer initialized (vocab size: {})\n", tokenizer.vocab_size());

    // Encode text prompt
    std::string prompt_text = "Jack and Jill went";
    std::println("Prompt text: \"{}\"", prompt_text);

    std::vector<int> prompt = tokenizer.encode(prompt_text);
    std::println("Encoded to {} tokens: [{}]", prompt.size(), std::format("{}", prompt[0]));
    for (size_t i = 1; i < prompt.size(); ++i) {
        std::print(", {}", prompt[i]);
    }
    std::println("]\n");

    // Generate 10 tokens with real-time feedback
    std::println("Generating 10 tokens with top-p sampling (p=0.95, T=0.8)...");
    std::println("(Tokens will be decoded as they are generated)\n");

    auto gen_start = std::chrono::high_resolution_clock::now();

    // Create generation config with callback for real-time printing
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

    // Decode full generated text
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
    std::println("\n✓ Real weight loading, tokenization, and generation working!");
    std::println("✓ Model can now generate readable text from text prompts!");
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    std::println("Usage: {} [OPTIONS]", program_name);
    std::println("\nOptions:");
    std::println("  --help, -h    Show this help message");
    std::println("\nDefault behavior (no args):");
    std::println("  Runs TinyLLaMA model test with text generation");
}

int main(int argc, char* argv[]) {
    std::println("========================================");
    std::println("FreeLLM - Fast CPU LLM Inference Engine");
    std::println("========================================\n");

    try {
        // Parse command line arguments
        bool show_help = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                show_help = true;
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

        // Default: Run TinyLLaMA model test
        std::println("Running TinyLLaMA model test...");
        std::println("(Use ./build/tests/test_main or ./build/tests/test_quantization to run tests)\n");

        test_load_tinyllama();

        std::println("\n========================================");
        std::println("TinyLLaMA test completed!");
        std::println("========================================\n");

        std::println("🎉 You now have a complete LLM inference engine!");
        std::println("\nNext steps:");
        std::println("  1. ✓ Load real TinyLLaMA weights (DONE!)");
        std::println("  2. ✓ Add tokenizer integration (DONE!)");
        std::println("  3. Implement KV cache for 10-100x speedup");
        std::println("  4. ✓ Add quantization (INT8/INT4) support");
        std::println("  5. Further SIMD optimizations\n");

    } catch (const std::exception& e) {
        std::println(stderr, "\nError: {}", e.what());
        return 1;
    }

    return 0;
}
