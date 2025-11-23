# Paged KV Cache Implementation Details

## Overview

The Paged KV Cache is a memory management technique designed to optimize the storage of Key-Value (KV) pairs during Large Language Model (LLM) inference. It addresses the memory fragmentation and waste issues associated with traditional contiguous KV cache allocation, enabling larger batch sizes and efficient support for Continuous Batching.

This implementation is inspired by operating system paging mechanisms. Instead of allocating a contiguous block of memory for each sequence's KV cache (which requires pre-allocating for the maximum possible sequence length), we divide the memory into fixed-size blocks and allocate them dynamically as needed.

## Core Components

The implementation consists of three main components defined in `include/core/paged_kv_cache.hpp`:

### 1. KVCacheConfig

A simple structure holding the configuration parameters for the cache:

*   `block_size`: The number of tokens stored in a single block (default: 16).
*   `max_num_blocks`: The total number of blocks available in the global pool (default: 1024).
*   `n_kv_heads`: The number of KV heads in the model.
*   `head_dim`: The dimension of each head.

### 2. KVCacheManager

The `KVCacheManager` is responsible for managing the global physical memory pool. It acts as the "Physical Memory Manager".

**Key Responsibilities:**
*   **Global Memory Allocation:** It allocates two large tensors, `global_keys_` and `global_values_`, which represent the entire physical memory available for the cache.
    *   Shape: `[max_num_blocks, block_size, n_kv_heads, head_dim]`
*   **Block Management:** It maintains a "free list" (`free_blocks_`) of available block indices.
*   **Allocation/Deallocation:**
    *   `allocate_block()`: Pops a block ID from the free list. Throws an error if OOM.
    *   `free_block(block_id)`: Pushes a block ID back onto the free list.
*   **Thread Safety:** Uses a `std::mutex` to ensure thread-safe access to the free list.

### 3. PagedKVCache

The `PagedKVCache` represents the KV cache for a **single sequence**. It acts as the "Virtual Memory" view for a sequence.

**Key Responsibilities:**
*   **Block Table:** Maintains a `block_table_` (vector of `int32_t`), which maps logical block indices (sequence position) to physical block IDs in the `KVCacheManager`.
    *   `logical_block_idx = logical_token_pos / block_size`
    *   `physical_block_id = block_table_[logical_block_idx]`
*   **Dynamic Allocation:** As the sequence grows, new blocks are requested from the `KVCacheManager` and added to the `block_table_`.
*   **Data Copying:** The `update()` method handles copying new KV data into the correct physical location.

## Memory Layout

The physical memory is organized as a 4D tensor:

```
[max_num_blocks, block_size, n_kv_heads, head_dim]
```

*   **Block ID**: The first dimension indexes the physical block.
*   **Block Offset**: The second dimension indexes the token position within that block (0 to `block_size - 1`).

## Data Flow: The `update` Method

When new tokens are generated, their Key and Value tensors need to be stored in the cache. The `update` method in `PagedKVCache` handles this:

1.  **Calculate Position:** For each new token, determine its logical position (`current_len_ + i`).
2.  **Map to Block:**
    *   `logical_block_idx = logical_pos / block_size`
    *   `block_offset = logical_pos % block_size`
3.  **Allocate if Needed:** If `logical_block_idx` exceeds the current size of the `block_table_`, a new physical block is allocated from the `KVCacheManager` and added to the table.
4.  **Copy Data:** The Key and Value vectors for the token are copied from the input tensor to the global storage at:
    *   `global_keys_[physical_block_id][block_offset]`
    *   `global_values_[physical_block_id][block_offset]`

## Example Scenario

**Config:** `block_size = 16`

1.  **Initial State:** Sequence length 0. `block_table_` is empty.
2.  **Prefill (Prompt = 10 tokens):**
    *   `logical_block_idx` is 0 for all tokens.
    *   One block (e.g., Physical ID 42) is allocated. `block_table_ = [42]`.
    *   Tokens 0-9 are written to Block 42, offsets 0-9.
3.  **Generation (Token 11 - 16):**
    *   Tokens 10-15 map to `logical_block_idx` 0.
    *   They are written to Block 42, offsets 10-15. Block 42 is now full.
4.  **Generation (Token 17):**
    *   Token 16 maps to `logical_block_idx` 1.
    *   A new block (e.g., Physical ID 99) is allocated. `block_table_ = [42, 99]`.
    *   Token 16 is written to Block 99, offset 0.

## Block Lifecycle & Reuse

The "expiration" of blocks is tied directly to the lifecycle of the sequence they belong to. There is no automatic garbage collection or timeout; blocks are freed explicitly when a sequence is finished.

### 1. Allocation
Blocks are allocated from the `free_blocks_` list in `KVCacheManager` when a sequence grows longer than its current capacity.

### 2. Active Use
While a sequence is being generated, its `PagedKVCache` object holds exclusive ownership of the physical blocks listed in its `block_table_`.

### 3. Deallocation (Reuse)
Blocks become available for reuse in two scenarios:
1.  **Sequence Completion:** When a request finishes (e.g., generates an End-Of-Sequence token), the system destroys the `PagedKVCache` object associated with that request.
2.  **Destructor/Reset:** The `~PagedKVCache()` destructor calls `reset()`, which iterates through the `block_table_` and returns every block ID back to the `KVCacheManager`'s free list via `free_block()`.

Once returned to the free list, these blocks can be immediately re-allocated to new incoming requests or other growing sequences.

## Advantages

*   **Zero Waste:** No need to reserve memory for the maximum possible sequence length. Memory is consumed only as tokens are generated.
*   **Fragmentation Reduction:** Small, fixed-size blocks reduce external fragmentation compared to variable-sized allocations.
*   **Sharing (Future):** Enables easy sharing of prefixes (e.g., system prompts) by pointing multiple sequences' block tables to the same physical blocks (Copy-on-Write).
