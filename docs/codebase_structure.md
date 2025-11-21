# Codebase Structure

This document explains the organization of the FreeLLM codebase, designed to separate low-level kernels, infrastructure, and core model logic.

## Directory Layout

The codebase is divided into three main layers, mirrored in both `include/` (headers) and `src/` (implementations):

### 1. Kernels (`kernels/`)
**Purpose**: Contains high-performance, low-level mathematical operations, SIMD implementations, and quantization logic. These components are often hardware-specific or heavily optimized.

*   **Contents**:
    *   `tensor_ops.hpp`, `tensor_ops_simd.hpp`: Basic tensor math operations (add, multiply, dot product) with SIMD optimizations (AVX2, NEON).
    *   `attention.hpp`: Attention mechanisms (e.g., Scaled Dot Product Attention).
    *   `rope.hpp`: Rotary Positional Embeddings implementation.
    *   `quantiz/`: Quantization primitives and kernels (Q4_K, Q8_0, etc.).
    *   `cpu_features.hpp`: CPU feature detection (AVX, NEON, etc.).

### 2. Infrastructure (`infra/`)
**Purpose**: Handles system-level operations, resource management, data loading, and threading. These components support the execution of the model but are not part of the mathematical model itself.

*   **Contents**:
    *   `model_loader.hpp`: Utilities for loading model weights from disk.
    *   `safetensors_loader.hpp`, `safetensors.hh`: Parsers for the SafeTensors format.
    *   `parallel_tensor_loader.hpp`: Multi-threaded tensor loading.
    *   `thread_pool.hpp`: Thread pool for parallel execution.
    *   `bounded_queue.hpp`: Thread-safe queue for task scheduling.

### 3. Core (`core/`)
**Purpose**: Defines the high-level model architecture, data structures, and inference logic. This layer orchestrates the kernels and infrastructure to perform LLM inference.

*   **Contents**:
    *   `tensor.hpp`: The main `Tensor` class definition.
    *   `llm_model.hpp`: The `LLMModel` class, representing the full language model.
    *   `transformer_block.hpp`: Implementation of a single Transformer block.
    *   `kv_cache.hpp`: Key-Value cache management for efficient generation.
    *   `generation.hpp`: High-level text generation loops.
    *   `sampling.hpp`: Token sampling strategies (greedy, top-k, top-p).
    *   `tokenizer.hpp`: Tokenization utilities.
    *   `model_config.hpp`: Configuration structures for model hyperparameters.

## Design Principles

*   **Separation of Concerns**: Kernels focus on math, Infra focuses on system resources, and Core focuses on model logic.
*   **Extensibility**: New hardware backends (e.g., GPU) can be added by implementing new kernels in `kernels/` without changing `core/` logic.
*   **Readability**: Grouping related files helps developers navigate the codebase and understand dependencies.
