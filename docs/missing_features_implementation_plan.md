# Implementation Plan for Missing Inference Engine Features

This document outlines the implementation strategy for key features currently missing from `freellm`. These features are selected based on their impact on inference performance, memory efficiency, and functional capabilities.

## 1. Paged KV Cache

**Status**: Missing (Current implementation uses contiguous memory allocation per layer).
**Impact**: Essential for efficient memory usage and enabling Inflight Batching. Eliminates memory fragmentation.

### Implementation Strategy

1.  **Data Structure Changes**:
    *   Create a `Block` struct representing a fixed-size chunk of tokens (e.g., 16 or 32 tokens).
    *   Create a `BlockTable` mapping: `SequenceID -> List<BlockPointer>`.
    *   Replace `KVCache::cached_keys_` and `cached_values_` (contiguous tensors) with a global `BlockManager` that allocates blocks from a pre-allocated pool.

2.  **Logic Updates**:
    *   **Allocation**: When a sequence needs more space, the `BlockManager` allocates a new block.
    *   **Attention Kernel**: Update `attention.hpp` to read K/V data from non-contiguous blocks.
        *   *Current*: `K[i]` is at `base + i * stride`.
        *   *New*: `K[i]` logic:
            ```cpp
            block_idx = i / block_size;
            offset = i % block_size;
            block_ptr = block_table[block_idx];
            data = block_ptr->data + offset * head_dim;
            ```

3.  **Refactoring**:
    *   Modify `KVCache` class to hold a `std::vector<Block*>` instead of owning a large `Tensor`.

## 2. Quantized KV Cache (INT8/FP8)

**Status**: Missing (Current implementation uses `float`).
**Impact**: Reduces memory usage by 2x-4x, allowing longer context lengths or larger batch sizes.

### Implementation Strategy

1.  **Storage Format**:
    *   Store K and V as `int8_t` (or `fp8` if hardware supports) instead of `float`.
    *   Store a `scale` (float) and `zero_point` (int8/float) per head or per block.

2.  **Quantization Logic**:
    *   In `KVCache::update()`:
        *   Compute min/max of the incoming `new_keys` and `new_values`.
        *   Calculate `scale = (max - min) / 255.0`.
        *   Quantize: `q_val = (val / scale)`.
        *   Store `q_val` and `scale`.

3.  **Dequantization in Attention**:
    *   In `attention.hpp`, before the dot product (or during, if using integer SIMD):
        *   `val = q_val * scale`.
    *   *Optimization*: Use SIMD instructions (AVX2/NEON) to perform dot products on INT8 data directly, accumulating into INT32, then scaling to float.

## 3. Inflight Batching (Continuous Batching)

**Status**: Missing (Current implementation supports single-sequence inference only).
**Impact**: Drastically improves throughput by processing multiple sequences simultaneously, adding new ones as soon as others finish.

### Implementation Strategy

1.  **Batch Manager**:
    *   Create a `Batch` class that holds a list of active `Sequence` objects.
    *   Each `Sequence` tracks its own `request_id`, `token_ids`, `generated_len`, and `status` (WAITING, RUNNING, FINISHED).

2.  **Scheduler**:
    *   Implement a scheduler loop that:
        1.  Evicts finished sequences.
        2.  Adds new sequences from a request queue (up to memory limits).
        3.  Runs one step of the model for all active sequences.

3.  **Model Updates**:
    *   Update `LLMModel::forward` to accept a `Batch` instead of a single `token_ids` vector.
    *   **Input Packing**: Flatten `token_ids` from all sequences into a single 1D tensor `[total_batch_tokens]`.
    *   **Positional Info**: Pass a `position_ids` tensor `[total_batch_tokens]` so RoPE knows the position of each token.
    *   **Attention Masking**: Since sequences are packed, standard causal masking isn't enough. We need a "Block Diagonal" mask or simply rely on the Paged KV Cache logic where each sequence only attends to its own blocks.

## 4. LoRA (Low-Rank Adaptation) Support

**Status**: Missing.
**Impact**: Allows efficient fine-tuning and serving of multiple specialized models sharing the same base weights.

### Implementation Strategy

1.  **LoRA Layer**:
    *   Define a `LoRAAdapter` struct:
        ```cpp
        struct LoRAAdapter {
            Tensor A; // [r, in_dim]
            Tensor B; // [out_dim, r]
            float scaling;
        };
        ```

2.  **Model Integration**:
    *   Add a `std::unordered_map<std::string, LoRAAdapter>` to `LLMModel` or `TransformerBlock`.
    *   Keyed by layer name (e.g., "layers.0.attn.W_q").

3.  **Forward Pass Modification**:
    *   Modify `quant::linear_forward` or create a wrapper `lora_linear_forward`.
    *   Logic: `output = W @ x + (B @ (A @ x)) * scaling`.
    *   *Optimization*: For multiple LoRAs (SLoRA), batch the LoRA computations.

## 5. Multi-modal Support (Vision-Language)

**Status**: Missing.
**Impact**: Enables processing of images alongside text.

### Implementation Strategy

1.  **Vision Encoder**:
    *   Implement or load a Vision Transformer (ViT) (e.g., CLIP, SigLIP).
    *   This requires adding a new module `VisionEncoder` similar to `LLMModel` but for images.

2.  **Projector**:
    *   Implement a Multi-Layer Perceptron (MLP) projector that maps ViT output embeddings to the LLM's embedding space (`d_model`).

3.  **Input Handling**:
    *   Update `LLMModel::forward` to accept `std::optional<Tensor> images`.
    *   If images are present:
        1.  Run `VisionEncoder(images) -> image_features`.
        2.  Run `Projector(image_features) -> image_embeddings`.
        3.  Insert `image_embeddings` into the `hidden_states` at the placeholder token positions (e.g., `<image>`).

## 6. Tensor Parallelism (Multi-GPU)

**Status**: Missing.
**Impact**: Allows running models larger than a single GPU's memory and speeds up inference.

### Implementation Strategy

1.  **Communication Layer**:
    *   Integrate MPI or NCCL for inter-process communication.

2.  **Weight Sharding**:
    *   **Column Parallel**: Split `W_q`, `W_k`, `W_v`, `W_up`, `W_gate` along the output dimension.
    *   **Row Parallel**: Split `W_o`, `W_down` along the input dimension.

3.  **Forward Pass Updates**:
    *   **Column Parallel Layers**: Each rank computes a slice of the output. `AllGather` is needed if the next layer needs full input, but typically we keep it sharded until a Row Parallel layer.
    *   **Row Parallel Layers**: Each rank computes a partial sum. Requires `AllReduce` (sum) to combine results across ranks.

4.  **KV Cache**:
    *   Split KV heads across ranks. Each rank manages a subset of heads.
