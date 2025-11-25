# Continuous Batching System Design

## Overview
This document outlines the design for a Continuous Batching (CB) system in FreeLLM, leveraging the Paged KV Cache to maximize throughput and resource utilization.

## 1. Foundation: Paged KV Cache & Memory Management

### 1.1 Paged KV Cache
The system uses the existing `PagedKVCache` and `KVCacheManager` classes.
- **Block Allocation**: `KVCacheManager` manages a global pool of fixed-size blocks.
- **Block Table**: `PagedKVCache` maintains a mapping from logical token positions to physical block IDs.
- **On-Demand Allocation**: Blocks are allocated dynamically as the sequence grows.

### 1.2 Memory-Aware Batch Selection
The scheduler prioritizes memory availability over simple request counts.
- **Constraint**: `M_avl >= M_req`
    - `M_avl`: Available free blocks in `KVCacheManager`.
    - `M_req`: Estimated blocks required for the next step of a request.
- **Working Set Estimation**:
    - For a new request (prefill): `M_req = ceil(prompt_len / block_size)`
    - For a running request (decode): `M_req = 1` (if current block is full) or `0` (if current block has space).

## 2. Core Scheduling Mechanisms

### 2.1 Request State Management
Requests are managed in three queues:
1.  **WAITING**: New requests received but not yet scheduled.
2.  **RUNNING**: Requests currently being processed (prefill or decode).
3.  **FINISHED**: Requests that have completed generation.

### 2.2 The Continuous Batching Loop (`step()`)
The engine's `step()` function performs the following:

1.  **Egress (Eviction)**:
    - Check all `RUNNING` requests.
    - If a request has finished (EOS or max length), move it to `FINISHED` and free its KV cache blocks.

2.  **Ingress (Scheduling)**:
    - Calculate available memory blocks.
    - Select requests from `WAITING` queue that fit into available memory.
    - Move selected requests to `RUNNING`.
    - **Policy**: First-Come-First-Served (FCFS) initially, with memory constraints.

3.  **Execution**:
    - **Flattened Batch**: Concatenate all active sequences (prefill tokens + decode tokens) into a single "super sequence".
    - **Metadata Preparation**:
        - `block_tables`: List of block tables for each sequence.
        - `input_lengths`: Length of input for each sequence (prompt len or 1).
        - `sequence_lengths`: Total length of each sequence so far.
    - **Forward Pass**: Call the model with the flattened input and metadata.
    - **Sampling**: Sample next tokens for all active requests.
    - **Update**: Append new tokens to `Sequence` objects (triggering KV cache updates).

## 3. Implementation Details

### 3.1 Sequence Class (`include/core/sequence.hpp`)
The `Sequence` class is the fundamental unit of work in the continuous batching system. It encapsulates the state of a single user request.

*   **Responsibility**:
    *   **Token Storage**: Maintains the full history of tokens (prompt + generated).
    *   **KV Cache Ownership**: Owns a vector of `PagedKVCache` unique pointers, one for each transformer layer.
    *   **Lifecycle Management**: Automatically frees all associated KV cache blocks when the `Sequence` object is destroyed (RAII).
    *   **State Tracking**: Tracks whether the sequence is finished (EOS reached or max length).

*   **Layer-wise Management**:
    *   The transformer model has $L$ layers.
    *   Each layer needs its own independent KV cache to store the keys and values for that specific depth in the network.
    *   `Sequence` initializes `n_layers` instances of `PagedKVCache` in its constructor.
    *   During the forward pass, the `TransformerBlock` at layer $i$ retrieves the corresponding `PagedKVCache` from the `Sequence` via `seq.kv_cache(layer_idx)`.

### 3.2 Scheduler & Batching Loop
The scheduler is responsible for forming batches from the pool of active sequences.

#### The `step()` Loop
The core inference loop operates as follows:

1.  **Sequence Management**:
    *   Remove finished sequences from the `running_queue`.
    *   Add new sequences from `waiting_queue` to `running_queue` if there is sufficient memory.
        *   *Memory Check*: `available_blocks >= required_blocks`.
        *   For a new sequence (prefill), `required = prompt_len / block_size`.
        *   For a running sequence (decode), `required = 1` (worst case).

2.  **Batch Construction**:
    *   The model expects a flattened input tensor.
    *   We concatenate the input tokens of all running sequences into a single 1D tensor.
        *   **Prefill Sequences**: Contribute all prompt tokens.
        *   **Decode Sequences**: Contribute only the last generated token.
    *   We prepare auxiliary metadata vectors:
        *   `position_ids`: The logical position of each token in its sequence (for RoPE).
        *   `block_tables`: A list of block tables (one per sequence) to be passed to the PagedAttention kernel.
        *   `slot_mapping`: (Optional) Optimization to map each flattened token index directly to a physical memory slot.

3.  **Model Execution**:
    *   Call `model.forward()` with the flattened input and metadata.
    *   The `Attention` layers use the `paged_scaled_dot_product_attention` kernel, reading directly from the non-contiguous blocks specified by the `block_tables`.

4.  **Sampling & Update**:
    *   Extract logits for the last token of each sequence.
    *   Sample the next token for each sequence independently.
    *   Append the new token to the `Sequence` object.
    *   Check for EOS or max length; mark sequence as `finished` if condition met.

## 4. Future Optimizations
- **Chunked Prefill**: Split long prompts to avoid head-of-line blocking.
- **Prefix Sharing**: Reuse KV blocks for common prefixes.
- **Speculative Decoding**: Verify draft tokens in batch.
