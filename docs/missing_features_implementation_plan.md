# Implementation Plan for Missing Inference Engine Features

This document outlines the implementation strategy for key features currently missing from `freellm`. These features are selected based on their impact on inference performance, memory efficiency, and functional capabilities.

## 1. Paged KV Cache
**Status**: **Complete** (Implemented in `include/core/paged_kv_cache.hpp` and `src/core/paged_kv_cache.cpp`).
**Impact**: Enabling Inflight Batching and efficient memory usage.

## 2. Quantized KV Cache (INT8/FP8)
**Status**: **Planned** (Next Priority).
**Impact**: Reduces memory usage by 2x-4x.

## 3. Inflight Batching (Continuous Batching)
**Status**: **Complete** (Implemented in `GenerationEngine` and `Scheduler`).
**Impact**: Drastically improves throughput.

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
