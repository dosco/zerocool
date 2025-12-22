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
    *   **Backend Integration**: The model uses the `ComputeBackend` interface (from `hybrid_jit_design.md`) to dispatch kernels.
    *   The `Attention` layers dispatch the `paged_attention` kernel (via `execute_kernel`), reading directly from the non-contiguous blocks specified by the `block_tables`.

4.  **Sampling & Update**:
    *   Extract logits for the last token of each sequence.
    *   Sample the next token for each sequence independently.
    *   Append the new token to the `Sequence` object.
    *   Check for EOS or max length; mark sequence as `finished` if condition met.

## 4. Ragged Continuous Batching & Kernel Infrastructure

To achieve maximum efficiency, the system implements **Ragged Continuous Batching**, which eliminates the need for padding tokens. This allows processing sequences of varying lengths (during prefill) and varying positions (during decode) in a single kernel launch.

### 4.1 Concept: Ragged Batching
Instead of padding sequences to the same length (e.g., `[B, Max_L]`), we flatten all tokens from all active sequences into a single 1D tensor `[Total_Tokens]`.
-   **Batched Prefill**: Multiple new prompts ($S_1, S_2, \dots$) are processed similarly to a single long sequence, but with attention masks respecting sequence boundaries.
-   **Ragged Decode**: Multiple sequences are generating tokens at different current positions ($P_1, P_2, \dots$).

### 4.2 Kernel Infrastructure Changes
To support this, standard kernels (which often assume batch size 1 or fixed sequence length) are replaced or augmented with "Ragged" versions.

#### 4.2.1 `rope_ragged`
*   **Problem**: Standard RoPE kernels often take a scalar `start_pos`. In continuous batching, every sequence in the batch is at a different position in its generation (e.g., Seq A at token 10, Seq B at token 2048).
*   **Solution**: The `rope_ragged` kernel accepts a `positions` buffer of size `[batch_size]`.
    *   **Input**: `buffer(0)` input, `buffer(3)` positions.
    *   **Logic**: Thread $t$ for Batch $b$ reads `pos = positions[b]` and applies rotary embedding for that specific position.

#### 4.2.2 `gqa_attention_prefill_ragged`
*   **Problem**: Standard prefill attention (FlashAttention/GQA) processes a single sequence with causal masking `[0..t]`. We want to process $N$ prompts in parallel.
*   **Solution**: The kernel treats the input as one large stream of tokens but uses `cu_seqlens` (Cumulative Sequence Lengths) to identify boundaries.
*   **Inputs**:
    *   `q`, `k`, `v`, `block_tables` (all flattened)
    *   `cu_seqlens`: Array `[0, L_1, L_1+L_2, ..., Total_Tokens]`.
*   **Logic**:
    1.  Kernel launched with `Total_Tokens` threads (or groups).
    2.  Each thread determines which sequence $S$ it belongs to (via binary search or linear scan on `cu_seqlens`).
    3.  Calculates local position `local_pos = global_pos - cu_seqlens[S]`.
    4.  Fetches `block_table[S]` to access Paged KV Cache.
    5.  Performs Causal Attention over `[0..local_pos]`.

## 5. Future Optimizations
- **Chunked Prefill**: Split long prompts to avoid head-of-line blocking.
- **Prefix Sharing**: Reuse KV blocks for common prefixes.
- **Speculative Decoding**: Verify draft tokens in batch.
