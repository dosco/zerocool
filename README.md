# FreeLLM: Hybrid JIT Inference Engine

FreeLLM is a high-performance, hybrid host-driven JIT inference engine designed for flexibility and speed across diverse hardware backends. It leverages a unified C++ runtime to orchestrate execution while dispatching compute-intensive kernels to specialized backends.

## Key Features

- **Hybrid Architecture**: Host-driven orchestration with JIT-compiled kernels.
- **Multi-Backend Support**:
  - **Metal (macOS)**: Native Objective-C++ backend using Metal Performance Shaders (MPS) and custom MSL kernels.
# FreeLLM: The Hybrid JIT Inference Engine 🚀

**FreeLLM** is a next-generation inference engine designed to bridge the gap between flexibility and raw performance. It features a unique **Hybrid Host-Driven JIT Architecture** that combines the ease of C++ orchestration with the extreme speed of specialized hardware backends.

## ✨ The Magic of Hybrid JIT

Most inference engines are either flexible (but slow) or fast (but rigid). FreeLLM gives you both:

*   **Host-Driven Orchestration**: The complex logic of the LLM (sampling, beam search, tokenization) runs on the CPU, where it's easy to debug and modify.
*   **JIT-Compiled Kernels**: The heavy lifting (Matrix Multiplication, Attention, RoPE) is dispatched to JIT-compiled kernels on the GPU/TPU.
*   **Zero-Overhead Abstraction**: Our `ComputeBackend` interface maps directly to hardware APIs (Metal, CUDA, PJRT) without bloated intermediate layers.

## 🎮 Choose Your Backend

FreeLLM supports multiple backends. Here's how to choose the right one for you:

### 1. CPU Backend (Reference Implementation)
*   **Best for**: Compatibility, debugging, running on any machine.
*   **Executable**: `freellm`
*   **Status**: Fully functional chat interface.
*   **Usage**:
    ```bash
    ./build/bin/freellm
    ```

### 2. Metal Backend (macOS / Apple Silicon) 🍎
*   **Best for**: MacBook Pro, Mac Studio, Mac Mini.
*   **Magic**: Uses raw Metal Performance Shaders (MPS) and custom MSL kernels compiled at runtime!
*   **Executable**: `metal_inference` (Demo)
*   **Status**: High-performance layer benchmarks (Chat integration coming soon).
*   **Usage**:
    ```bash
    ./build/bin/metal_inference
    ```

### 3. CUDA Backend (NVIDIA) 🟩
*   **Best for**: NVIDIA GPUs (Linux/Windows).
*   **Magic**: JIT-compiles Triton kernels directly to PTX.
*   **Status**: Backend implemented, integration in progress.

## 🚀 Quick Start: Chat with TinyLlama

### Step 1: Build
```bash
./build.sh
```

### Step 2: Download Model
```bash
bash scripts/download_tinyllama.sh
```

### Step 3: Run (CPU Chat)
```bash
./build/bin/freellm
```

### Step 4: Run Metal Demo (Layer Benchmark)
Want to see the speed? Run the Metal backend demo:
```bash
./build/bin/metal_inference
```

## 🛠️ Building from Source

### Prerequisites
- CMake 3.20+
- C++23 Compiler (Clang 17+, GCC 15+, MSVC 2022)
- **macOS**: Xcode Command Line Tools (for Metal)
- **NVIDIA**: CUDA Toolkit 12.x (for CUDA)

### Build Options
You can explicitly enable/disable backends:

```bash
mkdir build && cd build
cmake .. \
  -DFREELM_ENABLE_CUDA=ON \  # Force CUDA
  -DFREELM_ENABLE_METAL=ON   # Force Metal
cmake --build .
```

## 🧩 Architecture

FreeLLM uses a layered architecture:

1.  **Core (`src/core`)**: Model definitions (`LLMModel`), Tokenizer, Sampling.
2.  **Infrastructure (`src/infra`)**: The hardware abstraction layer.
    *   `ComputeBackend`: Unified interface.
    *   `MetalBackend`: Direct Metal API calls.
    *   `CUDABackend`: Driver API + Triton.
3.  **Kernels (`kernels/`)**:
    *   **Triton**: Python-based kernels for NVIDIA.
    *   **Metal**: MSL shaders (`kernels/metal/kernels.metal`) for Apple.

## License

MIT
