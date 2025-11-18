# Agent Instructions for FreeLLM

This document contains high-level guidelines for AI coding agents working on the FreeLLM project.

## Project Overview

FreeLLM is a fast CPU-only LLM inference engine written in modern C++23. The project aims to be:
- **Educational**: Code should be easy to follow, learn from, and understand
- **Evolving to production**: Starting simple but with an eye toward performance optimization
- **Modern C++**: Using C++23 features throughout

## Code Philosophy

### 1. Educational First, Performance Second

- **Write clear, well-commented code** that explains what's happening and why
- **Include extensive documentation** in headers explaining concepts (e.g., "What is RoPE?", "Why SwiGLU?")
- **Progressive optimization**: Start with naive implementations, then add optimized versions alongside
  - Keep `_naive` implementations as reference
  - Add optimized versions (e.g., `_avx2`) in separate functions
  - Use runtime dispatch to select best implementation

Example structure:
```cpp
// Naive implementation (educational reference)
Tensor matmul_naive(const Tensor& a, const Tensor& b);

// Optimized implementation (performance)
Tensor matmul_avx2(const Tensor& a, const Tensor& b);

// Smart dispatcher (automatically picks best)
Tensor matmul(const Tensor& a, const Tensor& b);
```

### 2. C++23 Modern Practices

- **Use `std::print` and `std::println`** instead of `std::cout` for all output
- **Use modern C++ features**: `std::format`, structured bindings, concepts, etc.
- **Prefer standard library** over external dependencies when possible
- **Move semantics**: No unnecessary copies, use move constructors/assignment
- **RAII**: Proper resource management, smart pointers where appropriate

### 3. Code Organization

- **Headers in `include/`**: All public APIs and implementations (header-only for simplicity)
- **Source in `src/`**: Main entry point and test code
- **Separate concerns**: One major concept per file
  - `tensor.hpp` - Core tensor class
  - `rope.hpp` - Rotary position embeddings
  - `attention.hpp` - Attention mechanism
  - etc.

### 4. Documentation Standards

Every major class/function should have:
- **Purpose**: What does this do?
- **Architecture**: How does it work? (diagrams in comments welcome)
- **Design decisions**: Why this approach?
- **Shape transformations**: For tensor operations, document input/output shapes
- **Examples**: Show typical usage

Example:
```cpp
/**
 * @brief Rotary Position Embeddings (RoPE)
 *
 * RoPE encodes position information by rotating token embeddings.
 * Instead of adding position embeddings, we rotate the query/key vectors
 * by an angle that depends on their position.
 *
 * Why RoPE?
 * - Better extrapolation to longer sequences
 * - Relative position encoding (naturally handles distance between tokens)
 * - Used in modern LLMs (LLaMA, GPT-NeoX, PaLM)
 *
 * @param seq_len Maximum sequence length
 * @param dim Dimension (must be even)
 */
```

### 5. Target Model

- **Focus on TinyLLaMA 1.1B** as the practical target model
- Keep code general enough to support similar architectures (LLaMA-style models)
- Configuration should match TinyLLaMA by default

### 6. Performance Targets

- **CPU architecture**: x86-64 with AVX2/AVX-512, ARM with NEON
- **Runtime dispatch**: Detect CPU features and use best implementation
- **Memory efficiency**: Use memory mapping, SIMD-aligned allocations
- **Future optimizations** (not yet implemented):
  - KV cache (10-100x speedup for generation)
  - Quantization (INT8/INT4)
  - Flash Attention

### 7. Testing Philosophy

- **Comprehensive test suite** in `src/main.cpp`
- Test each component individually before integration
- Include performance benchmarks (GFLOPS, tokens/sec)
- Real-world validation with actual model weights

### 8. Build System

- **CMake** with C++23 support
- **Compiler flags**: `-mavx2 -mfma` for x86-64 optimizations
- **Always use the build script**: Run `./build.sh` for building (do not run cmake commands directly)
- **The build script** handles compiler selection (GCC 15) and configuration
- **Strict warnings**: Catch issues early

### 9. Dependencies

- **Minimize external dependencies**: Prefer standard library when possible
- **Use CMake FetchContent for third-party libraries**: Never require manual installation
  - All libraries must be automatically downloaded and built via FetchContent
  - Example: RE2, SentencePiece, etc.
  - Ensures ABI compatibility (everything uses same compiler)
  - Makes project self-contained and easy to build
- **System dependencies only for basics**: POSIX APIs for memory mapping, SIMD intrinsics
- **No JSON library**: Write simple parsers for specific formats (e.g., safetensors)

### 10. Git Workflow

- **Descriptive commit messages**: Explain what and why
- **Small, focused commits**: One logical change per commit
- **No emojis** in code or commits unless explicitly requested
- **Include context**: Generated with Claude Code attribution when appropriate

## Current Architecture

```
FreeLLM/
├── include/
│   ├── tensor.hpp              # Core tensor class with SIMD alignment
│   ├── tensor_ops.hpp          # Naive implementations + smart dispatch
│   ├── tensor_ops_simd.hpp     # AVX2/NEON optimized operations
│   ├── cpu_features.hpp        # Runtime CPU feature detection
│   ├── rope.hpp                # Rotary position embeddings
│   ├── attention.hpp           # Multi-head attention mechanism
│   ├── transformer_block.hpp   # FeedForward + TransformerBlock
│   ├── llm_model.hpp           # Complete LLM pipeline
│   ├── model_config.hpp        # TinyLLaMA configuration
│   ├── model_loader.hpp        # Weight loading interface
│   ├── safetensors_loader.hpp  # Safetensors format parser
│   ├── sampling.hpp            # Greedy, Top-K, Top-P sampling
│   └── generation.hpp          # Autoregressive generation loop
├── src/
│   └── main.cpp                # Test suite and examples
├── scripts/
│   └── download_tinyllama.sh   # Download TinyLLaMA weights
└── build.sh                    # Simple build script
```

## Future Roadmap

1. **KV Cache**: Major performance improvement for generation
2. **Tokenizer Integration**: SentencePiece/BPE for text ↔ token conversion
3. **Quantization**: INT8/INT4 support for reduced memory
4. **Flash Attention**: More efficient attention computation
5. **Multi-threading**: Parallelize layer computation
6. **Metal/CUDA backends**: GPU acceleration (while keeping CPU as primary)

## Common Patterns

### Adding a New Operation

1. Write naive implementation with extensive comments
2. Add optimized SIMD version if performance-critical
3. Add dispatcher function that auto-selects best implementation
4. Add test case in main.cpp
5. Document shape transformations and complexity

### Adding a New Model Component

1. Create dedicated header file (e.g., `new_component.hpp`)
2. Include architecture diagram in comments
3. Explain the "why" behind design decisions
4. Add accessor methods for weight loading
5. Integrate into LLMModel if needed
6. Add focused test in main.cpp

### Debugging Performance Issues

1. Add timing measurements (use `std::chrono`)
2. Print GFLOPS or tokens/sec for comparison
3. Profile with perf/Instruments to find bottlenecks
4. Consider SIMD optimization for hot paths
5. Document performance characteristics in comments

## Key Principles Summary

- **Clarity over cleverness**: Code should be understandable
- **Education over optimization**: Unless optimization is the goal
- **Modern C++**: Use C++23 features throughout
- **Practical focus**: TinyLLaMA 1.1B is the target
- **Progressive enhancement**: Naive → Optimized → Production-ready
- **Comprehensive documentation**: Every major concept explained
- **Real-world validation**: Test with actual model weights

## Questions to Ask Before Major Changes

1. Does this make the code easier or harder to understand?
2. Is this optimization premature, or does it address a real bottleneck?
3. Are we maintaining both naive and optimized versions?
4. Have we documented the "why" as well as the "what"?
5. Does this work with TinyLLaMA specifically?
6. Are we following C++23 best practices?

---

When in doubt, prioritize clarity and educational value. Performance optimizations should be added alongside clear implementations, not at their expense.
