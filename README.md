# freellm

A modern C++ project with CMake build system.

## Prerequisites

- CMake 3.20 or higher
- C++23 compatible compiler:
  - GCC 12.1 or later (GCC 15 recommended for full support)
  - Clang 13 or later (Clang 17+ recommended)
  - MSVC 2022 version 17.13 or later with `/std:c++23preview`

## Building

### Quick Build (Recommended)
```bash
# Build the project
./build.sh

# Run the application
./build/bin/freellm
```

### Manual Build
```bash
# Create build directory
mkdir -p build && cd build

# Configure
cmake ..

# Build
cmake --build .

# Run
./bin/freellm
```

## Testing

The project uses [doctest](https://github.com/doctest/doctest) for unit testing.

### Quick Test (Recommended)
```bash
# Build and run all tests
./test.sh
```

### Manual Testing
```bash
# Build and run via CTest
cd build
ctest --output-on-failure

# Or run individual test executables
./build/tests/test_main
./build/tests/test_quantization
```

### Test Options
```bash
# List all available tests
./build/tests/test_main --list-test-cases

# Run a specific test case
./build/tests/test_main --test-case="KV Cache"

# Show detailed output for all tests
./build/tests/test_main --success

# Run tests matching a pattern
./build/tests/test_main --test-case="*Quantiz*"
```

See [tests/README.md](tests/README.md) for more detailed testing documentation.

## Project Structure

```
.
├── CMakeLists.txt          # Main CMake configuration
├── README.md               # This file
├── build.sh                # Build script
├── test.sh                 # Test runner script
├── src/                    # Source files
│   └── main.cpp           # Entry point
├── include/                # Header files
├── tests/                  # Test files (doctest framework)
│   ├── CMakeLists.txt     # Test configuration
│   ├── README.md          # Testing documentation
│   ├── test_main.cpp      # Core functionality tests
│   └── test_quantization.cpp  # Quantization tests
├── external/               # Third-party dependencies
│   └── doctest.h          # doctest testing framework
└── build/                  # Build artifacts (generated)
```

## Features

- C++23 standard (latest ratified C++ standard)
- CMake build system
- Compiler warnings enabled (-Wall -Wextra -Wpedantic -Werror)
- Comprehensive test suite using doctest framework
- Clean project structure
- Easy build and test scripts

## Quantization

The inference engine ships with llama.cpp-style quantization kernels:

- **Q8_0** (8-bit symmetric) for high-accuracy inference
- **Q4_K** (4.5-bit k-quant) for aggressive memory savings
- Scalar reference kernels + AVX2 dispatch for both formats
- Optional quantized storage for attention, feed-forward, and LM head weights

Enable quantization when constructing the model:

```cpp
using namespace freellm;

ModelConfig cfg = ModelConfig::tinyllama_1_1b();
quant::QuantConfig qcfg;
qcfg.default_type = quant::QuantType::Q8_0;   // or Q4_K
qcfg.lm_head = quant::QuantType::Q4_K;        // mix formats per component

LLMModel model(cfg, qcfg);
```

Weights are quantized on load (runtime quantization), and matvec kernels automatically
dispatch to AVX2 implementations when available.

## Development

To add new source files, update the `SOURCES` variable in `CMakeLists.txt`.

To add headers, place them in the `include/` directory.

## License

MIT
