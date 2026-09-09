#include <metal_stdlib>
using namespace metal;

// RMSNorm
// y = x * w * rsqrt(mean(x^2) + epsilon)
// Inputs: input(0), weight(1)
// Outputs: output(2)
// Params: N(3), epsilon(4)
kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint row_idx = id;
    device const float* row_in = input + row_idx * N;
    device float* row_out = output + row_idx * N;
    
    float sum_sq = 0.0f;
    for (uint i = 0; i < N; ++i) {
        sum_sq += row_in[i] * row_in[i];
    }
    
    float mean_sq = sum_sq / float(N);
    float scale = rsqrt(mean_sq + epsilon);
    
    for (uint i = 0; i < N; ++i) {
        row_out[i] = row_in[i] * scale * weight[i];
    }
}

// RoPE
// Inputs: input(0), cos(1), sin(2)
// Outputs: output(3)
// Params: head_dim(4), pos(5)
kernel void rope(
    device const float* input [[buffer(0)]],
    device const float* cos_theta [[buffer(1)]],
    device const float* sin_theta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& pos [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint pair_idx = id;
    uint half_dim = head_dim / 2;
    uint token_head_idx = pair_idx / half_dim;
    uint rot_dim = pair_idx % half_dim;
    
    // Global index
    uint i1 = token_head_idx * head_dim + rot_dim;
    uint i2 = i1 + half_dim;
    
    float x1 = input[i1];
    float x2 = input[i2];
    
    // Index into RoPE table
    // Table shape: [max_seq_len, head_dim/2]
    // We want row 'pos'.
    // uint rot_dim is already calculated
    uint table_idx = pos * (head_dim / 2) + rot_dim;
    
    float c = cos_theta[table_idx];
    float s = sin_theta[table_idx];
    
    output[i1] = x1 * c - x2 * s;
    output[i2] = x2 * c + x1 * s;
}

// SiLU
// Inputs: input(0)
// Outputs: output(1)
kernel void silu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = x / (1.0f + exp(-x));
}

// Softmax
// Inputs: input(0)
// Outputs: output(1)
// Params: N(2)
kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint row_idx = id;
    device const float* row_in = input + row_idx * N;
    device float* row_out = output + row_idx * N;
    
    float max_val = -INFINITY;
    for (uint i = 0; i < N; ++i) {
        max_val = max(max_val, row_in[i]);
    }
    
    float sum_exp = 0.0f;
    for (uint i = 0; i < N; ++i) {
        float val = exp(row_in[i] - max_val);
        row_out[i] = val; // Store temporarily
        sum_exp += val;
    }
    
    for (uint i = 0; i < N; ++i) {
        row_out[i] /= sum_exp;
    }
}

// Add
// Inputs: a(0), b(1)
// Outputs: output(2)
kernel void add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = a[id] + b[id];
}

// Mul
// Inputs: a(0), b(1)
// Outputs: output(2)
kernel void mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = a[id] * b[id];
}

// GEMM F32 (For FP32 weights like lm_head)
// C = A * B
// A: [rows, cols] (float, weight matrix - transposed)
// B: [batch_size, cols] (float, input)
// C: [batch_size, rows] (float, output)
// Grid: (rows, batch_size, 1)
// Block: 32 (simd group per row)
kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 num_groups [[threadgroups_per_grid]])
{
    uint row = tgid.x;
    uint batch_idx = tgid.y;
    
    float sum = 0.0f;
    
    // A is [rows, cols] - each row has 'cols' elements
    // B is [batch_size, cols] - each batch row has 'cols' elements
    // We compute: C[batch_idx, row] = dot(A[row, :], B[batch_idx, :])
    
    // Each simd group processes one row
    // tid.x goes from 0..31, we stride over cols
    for (uint i = tid.x; i < cols; i += 32) {
        float a_val = A[row * cols + i];
        float b_val = B[batch_idx * cols + i];
        sum += a_val * b_val;
    }
    
    // Reduction within simd group
    sum = simd_sum(sum);
    
    if (tid.x == 0) {
        // C is [batch_size, rows] in row-major
        C[batch_idx * num_groups.x + row] = sum;
    }
}

// Copy KV
// Inputs: k_src(0), v_src(1)
// Outputs: k_cache(2), v_cache(3) (passed as inputs in cpp, but effectively outputs for kernel writing)
// Params: pos(4), head_dim(5), n_kv_heads(6), max_seq_len(7)
kernel void copy_kv(
    device const float* k_src [[buffer(0)]],
    device const float* v_src [[buffer(1)]],
    device float* k_cache [[buffer(2)]],
    device float* v_cache [[buffer(3)]],
    constant uint& pos [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_kv_heads [[buffer(6)]],
    constant uint& max_seq_len [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    uint head_idx = id / head_dim;
    uint dim_idx = id % head_dim;
    
    uint cache_idx = (pos * n_kv_heads + head_idx) * head_dim + dim_idx;
    
    k_cache[cache_idx] = k_src[id];
    v_cache[cache_idx] = v_src[id];
}

// Q4_0 Block structure
struct BlockQ4_0 {
    float scale;
    uchar qs[16];
};

// GEMV Q4_0
// y = A * x
// A: [rows, cols] (quantized)
// x: [cols] (float)
// y: [rows] (float)
// Grid: rows
// Block: 32 (one warp/simdgroup per row for simplicity, or loop)
kernel void gemv_q4_0(
    device const uchar* A [[buffer(0)]], // Treat as raw bytes
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    // Each threadgroup handles one row of A.
    // A row has 'cols' elements.
    // 'cols' must be multiple of 32.
    
    float sum = 0.0f;
    uint n_blocks = cols / 32;
    
    // Stride of one block in bytes = 20 (4 bytes scale + 16 bytes qs)
    uint block_stride = 20;
    
    // Offset to start of row
    uint row_offset = row * n_blocks * block_stride;
    
    // Loop over blocks
    for (uint i = 0; i < n_blocks; ++i) { // Iterate over ALL blocks
        // Calculate offset for this block
        uint block_offset = row_offset + i * block_stride;
        
        // Load scale (first 4 bytes)
        // Use as_type for safe casting from bytes
        uint scale_bits = *(device const uint*)(A + block_offset);
        float scale = as_type<float>(scale_bits);
        
        // Load qs (next 16 bytes)
        // We only need the byte corresponding to our tid.
        // tid 0..31.
        // Byte index = tid / 2.
        uint8_t q_packed = A[block_offset + 4 + (tid / 2)];
        
        int8_t q = (tid % 2 == 0) ? (q_packed & 0x0F) : (q_packed >> 4);
        
        float val = (q - 8) * scale;
        
        sum += val * x[i * 32 + tid];
    }
    
    // Reduction within threadgroup
    sum = simd_sum(sum);
    
    if (tid == 0) {
        y[row] = sum;
    }
}

// GEMM Q4_0 (Batched GEMV)
// C = A * B
// A: [rows, cols] (quantized)
// B: [cols, batch_size] (float)
// C: [rows, batch_size] (float)
// Grid: (rows, batch_size, 1)
// Block: 32
kernel void gemm_q4_0(
    device const uchar* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 num_groups [[threadgroups_per_grid]])
{
    uint row = tgid.x;
    uint batch_idx = tgid.y;
    
    float sum = 0.0f;
    uint n_blocks = cols / 32;
    uint block_stride = 20;
    uint row_offset = row * n_blocks * block_stride;
    
    // B is column-major or row-major? 
    // Usually B is [cols, batch] (column major if we view it as matrix, but in memory it's flat).
    // Let's assume B is [batch, cols] or [cols, batch]?
    // In inference, we usually have tokens as rows: [batch, d_model].
    // So B is [batch, cols].
    // If B is [batch, cols], then B[batch_idx, col] = B[batch_idx * cols + col].
    
    // Loop over blocks
    for (uint i = 0; i < n_blocks; ++i) {
        uint block_offset = row_offset + i * block_stride;
        uint scale_bits = *(device const uint*)(A + block_offset);
        float scale = as_type<float>(scale_bits);
        uint8_t q_packed = A[block_offset + 4 + (tid.x / 2)];
        int8_t q = (tid.x % 2 == 0) ? (q_packed & 0x0F) : (q_packed >> 4);
        float val = (q - 8) * scale;
        
        // B access: batch_idx is row in B, (i*32 + tid.x) is col in B
        uint b_idx = batch_idx * cols + (i * 32 + tid.x);
        sum += val * B[b_idx];
    }
    
    sum = simd_sum(sum);
    
    if (tid.x == 0) {
        // C index: batch_idx * rows + row
        C[batch_idx * num_groups.x + row] = sum;
    }
}

// GQA Attention Prefill (Batched with Causal Masking)
kernel void gqa_attention_prefill(
    device const float* q [[buffer(0)]],
    device const float* k_global [[buffer(1)]], // Global K pool
    device const float* v_global [[buffer(2)]], // Global V pool
    device const int* block_table [[buffer(3)]], // [max_num_blocks]
    device float* output [[buffer(4)]],        // [batch, n_heads, head_dim]
    constant uint& start_pos [[buffer(5)]],
    constant uint& n_heads [[buffer(6)]],
    constant uint& n_kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& block_size [[buffer(10)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]])
{
    // Grid: (n_heads, batch_size, 1)
    uint h = tgid.x;
    uint batch_idx = tgid.y;
    
    uint simd_lane_id = tid.x % 32;
    uint simd_group_id = tid.x / 32;
    
    uint kv_h = h / (n_heads / n_kv_heads);
    uint current_pos = start_pos + batch_idx;
    
    // Shared memory
    threadgroup float shared_scores[32]; 
    threadgroup float shared_max_score;
    threadgroup float shared_sum_exp;
    if (tid.x == 0) {
        shared_sum_exp = 1.0f; // This will be overwritten
        shared_max_score = -1e9f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_val = 0.0f;
    if (tid.x < head_dim) {
        // Q: [batch, n_heads, head_dim]
        q_val = q[(batch_idx * n_heads + h) * head_dim + tid.x];
    }
    
    // Pass 1: Max Score
    float max_score = -1e9f;
    
    for (uint t = 0; t <= current_pos; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            // Paged K access
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_table[logical_block_idx];
            
            // Global K: [max_blocks, block_size, n_kv_heads, head_dim]
            uint k_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            k_val = k_global[k_idx];
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            max_score = max(max_score, score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_max_score = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = shared_max_score;
    
    // Pass 2: Sum Exp
    float sum_exp = 0.0f;
    
    for (uint t = 0; t <= current_pos; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            // Paged K access
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_table[logical_block_idx];
            
            uint k_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            k_val = k_global[k_idx];
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            sum_exp += exp(score - max_score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_sum_exp = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_exp = shared_sum_exp;
    
    // Pass 3: Weighted Sum
    float weighted_sum = 0.0f;
    
    for (uint t = 0; t <= current_pos; ++t) {
        float k_val = 0.0f;
        float v_val = 0.0f;
        if (tid.x < head_dim) {
            // Paged K/V access
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_table[logical_block_idx];
            
            uint k_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            uint v_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            
            k_val = k_global[k_idx];
            v_val = v_global[v_idx];
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float weight = 0.0f;
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            weight = exp(score - max_score) / sum_exp;
            shared_scores[0] = weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        weight = shared_scores[0];
        weighted_sum += weight * v_val;
    }
    
    // Write output
    if (tid.x < head_dim) {
        // Output: [batch, n_heads, head_dim]
        // We need to write to the correct position for this head and batch index
        // Since this kernel is called for each token in the batch (batch_idx corresponds to token position in sequence)
        // Wait, batch_idx in prefill usually means "sequence index in batch", but here we are processing a single sequence of length N.
        // So batch_idx is actually the token index 'i' in the prompt [0..N-1].
        // Yes, tgid.y is batch_idx.
        
        // Output buffer is [n_tokens, n_heads, head_dim]
        // But wait, gqa_attention_prefill output was [batch_size, n_heads, head_dim]
        // where batch_size = prompt_len.
        // Yes.
        
        // But wait, weighted_sum is partial? No, we iterated over all t <= current_pos.
        // So weighted_sum is the final value for this head.
        // But we need to write it.
        
        // The output pointer is `device float* output`.
        // Index: (batch_idx * n_heads + h) * head_dim + tid.x
        output[(batch_idx * n_heads + h) * head_dim + tid.x] = weighted_sum;
    }
}

// RoPE Batched
// Inputs: input(0), cos(1), sin(2)
// Outputs: output(3)
// Params: head_dim(4), start_pos(5), n_heads(6)
// Grid: (head_dim/2, n_heads, batch_size)
kernel void rope_batched(
    device const float* input [[buffer(0)]],
    device const float* cos_theta [[buffer(1)]],
    device const float* sin_theta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& start_pos [[buffer(5)]],
    constant uint& n_heads [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint rot_dim = gid.x;
    uint head_idx = gid.y;
    uint batch_idx = gid.z;
    
    uint i1 = (batch_idx * n_heads + head_idx) * head_dim + rot_dim;
    uint i2 = i1 + head_dim / 2;
    
    float x1 = input[i1];
    float x2 = input[i2];
    
    uint pos = start_pos + batch_idx;
    uint table_idx = pos * (head_dim / 2) + rot_dim;
    
    float c = cos_theta[table_idx];
    float s = sin_theta[table_idx];
    
    output[i1] = x1 * c - x2 * s;
    output[i2] = x2 * c + x1 * s;
}

// Copy KV Batched
// Inputs: k_src(0), v_src(1)
// Outputs: k_cache(2), v_cache(3)
// Params: start_pos(4), head_dim(5), n_kv_heads(6), max_seq_len(7)
// Grid: (head_dim, n_kv_heads, batch_size)
kernel void copy_kv_batched(
    device const float* k_src [[buffer(0)]],
    device const float* v_src [[buffer(1)]],
    device float* k_cache [[buffer(2)]],
    device float* v_cache [[buffer(3)]],
    constant uint& start_pos [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_kv_heads [[buffer(6)]],
    constant uint& max_seq_len [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint dim_idx = gid.x;
    uint head_idx = gid.y;
    uint batch_idx = gid.z;
    
    uint src_idx = (batch_idx * n_kv_heads + head_idx) * head_dim + dim_idx;
    
    uint pos = start_pos + batch_idx;
    uint dst_idx = (pos * n_kv_heads + head_idx) * head_dim + dim_idx;
    
    k_cache[dst_idx] = k_src[src_idx];
    v_cache[dst_idx] = v_src[src_idx];
}

// Paged Attention (Decoding Phase)
kernel void paged_attention(
    device const float* q [[buffer(0)]],           // [batch_size, n_heads, head_dim]
    device const float* k_global [[buffer(1)]],    // Global K pool
    device const float* v_global [[buffer(2)]],    // Global V pool
    device const int* block_tables [[buffer(3)]],  // [batch_size * max_num_blocks]
    device const int* context_lens [[buffer(4)]],  // [batch_size]
    device float* output [[buffer(5)]],            // [batch_size, n_heads, head_dim]
    constant uint& n_heads [[buffer(6)]],
    constant uint& n_kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& block_size [[buffer(10)]],
    constant uint& max_num_blocks [[buffer(11)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]])
{
    // Grid: (n_heads, batch_size, 1)
    uint h = tgid.x;
    uint batch_idx = tgid.y;
    
    uint simd_lane_id = tid.x % 32;
    uint simd_group_id = tid.x / 32;
    
    uint kv_h = h / (n_heads / n_kv_heads);
    int context_len = context_lens[batch_idx];
    
    // Shared memory
    threadgroup float shared_scores[32]; 
    threadgroup float shared_max_score;
    threadgroup float shared_sum_exp;
    if (tid.x == 0) {
        shared_sum_exp = 1.0f;
        shared_max_score = -1e9f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_val = 0.0f;
    if (tid.x < head_dim) {
        // Q: [batch_size, n_heads, head_dim]
        q_val = q[(batch_idx * n_heads + h) * head_dim + tid.x];
    }
    
    // Pass 1: Max Score
    float max_score = -1e9f;
    
    // Iterate over tokens in the sequence
    for (int t = 0; t < context_len; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            // Paged K access
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            
            // block_tables is flattened [batch_size * max_num_blocks]
            int physical_block_id = block_tables[batch_idx * max_num_blocks + logical_block_idx];
            
            // Global K: [max_blocks, block_size, n_kv_heads, head_dim]
            uint k_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            k_val = k_global[k_idx];
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            max_score = max(max_score, score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_max_score = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = shared_max_score;
    
    // Pass 2: Sum Exp
    float sum_exp = 0.0f;
    
    for (int t = 0; t < context_len; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[batch_idx * max_num_blocks + logical_block_idx];
            
            uint k_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            k_val = k_global[k_idx];
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            sum_exp += exp(score - max_score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_sum_exp = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_exp = shared_sum_exp;
    
    // Pass 3: Weighted Sum
    float weighted_sum = 0.0f;
    
    for (int t = 0; t < context_len; ++t) {
        float k_val = 0.0f;
        float v_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[batch_idx * max_num_blocks + logical_block_idx];
            
            uint k_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            uint v_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            
            k_val = k_global[k_idx];
            v_val = v_global[v_idx];
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float weight = 0.0f;
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            weight = exp(score - max_score) / sum_exp;
            shared_scores[0] = weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        weight = shared_scores[0];
        weighted_sum += weight * v_val;
    }
    
    // Write output
    if (tid.x < head_dim) {
        output[(batch_idx * n_heads + h) * head_dim + tid.x] = weighted_sum;
    }
}

// RoPE Ragged (Batched with individual positions)
// Inputs: input(0), cos(1), sin(2), positions(3)
// Outputs: output(4)
// Params: head_dim(5), n_heads(6)
// Grid: (head_dim/2, n_heads, batch_size)
kernel void rope_ragged(
    device const float* input [[buffer(0)]],
    device const float* cos_theta [[buffer(1)]],
    device const float* sin_theta [[buffer(2)]],
    device const int* positions [[buffer(3)]], // [batch_size]
    device float* output [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_heads [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint rot_dim = gid.x;
    uint head_idx = gid.y;
    uint batch_idx = gid.z;
    
    uint i1 = (batch_idx * n_heads + head_idx) * head_dim + rot_dim;
    uint i2 = i1 + head_dim / 2;
    
    float x1 = input[i1];
    float x2 = input[i2];
    
    uint pos = positions[batch_idx];
    uint table_idx = pos * (head_dim / 2) + rot_dim;
    
    float c = cos_theta[table_idx];
    float s = sin_theta[table_idx];
    
    output[i1] = x1 * c - x2 * s;
    output[i2] = x2 * c + x1 * s;
}

// GQA Attention Prefill Ragged (Batched with various lengths)
// Inputs: q, k, v, block_table, cu_seqlens, output
// Grid: (n_heads, total_tokens, 1) or similar?
// Since we have variable lengths, we can't map y to batch_idx easily if we want 1 threadgroup per token.
// Standard approach: 
// 1. One kernel launch per sequence (what we do now).
// 2. Ragged kernel: Grid Y = total_tokens? 
//    Wait, attention is usually one threadgroup per HEAD per QUERY TOKEN.
//    If we have N total query tokens (sum of all prompt lengths), we can launch N * n_heads threadgroups.
//    Grid: (n_heads, total_tokens, 1).
//    In kernel, `token_idx = tgid.y`.
//    We need to find which sequence `token_idx` belongs to.
//    Binary search on `cu_seqlens`? Or pass an array `token_to_seq_id`?
//    `token_to_seq_id` is O(N) memory but O(1) in kernel.
//    `cu_seqlens` is small. Binary search is O(log B).
//    Let's use `cu_seqlens` with binary search for now.
kernel void gqa_attention_prefill_ragged(
    device const float* q [[buffer(0)]],
    device const float* k_global [[buffer(1)]],
    device const float* v_global [[buffer(2)]],
    device const int* block_tables [[buffer(3)]], // Flat: [n_seqs * max_blocks] or [sum_blocks]? Usually flat per seq.
    device const int* cu_seqlens [[buffer(4)]],   // [n_seqs + 1]
    device float* output [[buffer(5)]],
    constant uint& n_seqs [[buffer(6)]],
    constant uint& n_heads [[buffer(7)]],
    constant uint& n_kv_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& block_size [[buffer(11)]],
    constant uint& max_num_blocks [[buffer(12)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]])
{
    // Grid: (n_heads, total_tokens, 1)
    uint h = tgid.x;
    uint global_token_idx = tgid.y;
    
    // Find sequence index
    // Simple linear scan for small batch size, or binary search.
    // Assuming batch size < 128, linear scan is fast enough given threads.
    uint seq_idx = 0;
    for (uint i = 0; i < n_seqs; ++i) {
        if (global_token_idx < cu_seqlens[i+1]) {
            seq_idx = i;
            break;
        }
    }
    
    // Local token index within sequence
    uint seq_start_token = cu_seqlens[seq_idx];
    uint current_pos = global_token_idx - seq_start_token;
    
    uint simd_lane_id = tid.x % 32;
    uint simd_group_id = tid.x / 32;
    uint kv_h = h / (n_heads / n_kv_heads);
    
    // Shared memory
    threadgroup float shared_scores[32]; 
    threadgroup float shared_max_score;
    threadgroup float shared_sum_exp;
    if (tid.x == 0) {
        shared_sum_exp = 1.0f;
        shared_max_score = -1e9f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_val = 0.0f;
    if (tid.x < head_dim) {
        // Q: [total_tokens, n_heads, head_dim] (Flattened batch)
        q_val = q[(global_token_idx * n_heads + h) * head_dim + tid.x];
    }
    
    // Pass 1: Max Score
    float max_score = -1e9f;
    
    // Causal masking: attend to [0, current_pos]
    for (uint t = 0; t <= current_pos; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[seq_idx * max_num_blocks + logical_block_idx];
            
            uint k_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            k_val = k_global[k_idx];
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            max_score = max(max_score, score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_max_score = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = shared_max_score;
    
    // Pass 2: Sum Exp
    float sum_exp = 0.0f;
    for (uint t = 0; t <= current_pos; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[seq_idx * max_num_blocks + logical_block_idx];
            uint k_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            k_val = k_global[k_idx];
        }
        
        float partial_score = simd_sum(q_val * k_val);
        if (simd_lane_id == 0) shared_scores[simd_group_id] = partial_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            sum_exp += exp(score - max_score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_sum_exp = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_exp = shared_sum_exp;
    
    // Pass 3: Weighted Sum
    float weighted_sum = 0.0f;
    for (uint t = 0; t <= current_pos; ++t) {
        float k_val = 0.0f;
        float v_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[seq_idx * max_num_blocks + logical_block_idx];
            uint k_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            uint v_idx = (physical_block_id * block_size + block_offset) * n_kv_heads * head_dim + kv_h * head_dim + tid.x;
            k_val = k_global[k_idx];
            v_val = v_global[v_idx];
        }
        
        float partial_score = simd_sum(q_val * k_val);
        if (simd_lane_id == 0) shared_scores[simd_group_id] = partial_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float weight = 0.0f;
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            weight = exp(score - max_score) / sum_exp;
            shared_scores[0] = weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        weight = shared_scores[0];
        weighted_sum += weight * v_val;
    }
    
    if (tid.x < head_dim) {
        output[(global_token_idx * n_heads + h) * head_dim + tid.x] = weighted_sum;
    }
}


kernel void quantize_store_kv(
    device const float* src [[buffer(0)]],
    device char* dst_global [[buffer(1)]],
    device float* scales_global [[buffer(2)]],
    device const int* dst_offsets [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& src_elem_offset [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    // Grid: (seq_len, n_kv_heads, 1) threadgroups
    // Block: (32, 1, 1) threads
    uint token_idx = tgid.x;
    uint head_idx = tgid.y;
    uint lane = tid.x;
    
    // Add base offset to source index
    uint src_offset = src_elem_offset + (token_idx * n_kv_heads + head_idx) * head_dim;
    uint slot_idx = dst_offsets[token_idx];
    
    // Global data/scale offsets (per slot)
    uint dst_data_offset = (slot_idx * n_kv_heads + head_idx) * head_dim;
    uint dst_scale_offset = slot_idx * n_kv_heads + head_idx;
    
    // 1. Find Max (Parallel Reduction)
    float max_val = 0.0f;
    for (uint i = lane; i < head_dim; i += 32) {
        float val = src[src_offset + i];
        max_val = max(max_val, abs(val));
    }
    max_val = simd_max(max_val);
    
    // Leader writes scale
    float scale = max_val / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    
    if (lane == 0) {
        scales_global[dst_scale_offset] = scale;
    }
    
    // 2. Quantize (Parallel)
    float inv_scale = 1.0f / scale;
    for (uint i = lane; i < head_dim; i += 32) {
        float val = src[src_offset + i];
        float q = val * inv_scale;
        dst_global[dst_data_offset + i] = (char)clamp(round(q), -127.0f, 127.0f);
    }
}

kernel void paged_attention_int8(
    device const float* q [[buffer(0)]],           // [batch_size, n_heads, head_dim]
    device const char* k_global [[buffer(1)]],     // Int8 Global K
    device const char* v_global [[buffer(2)]],     // Int8 Global V
    device const float* k_scales [[buffer(3)]],    // Global K Scales [total_slots, n_kv_heads]
    device const float* v_scales [[buffer(4)]],    // Global V Scales
    device const int* block_tables [[buffer(5)]],  // [batch_size * max_num_blocks]
    device const int* context_lens [[buffer(6)]],  // [batch_size]
    device float* output [[buffer(7)]],            // [batch_size, n_heads, head_dim]
    constant uint& n_heads [[buffer(8)]],
    constant uint& n_kv_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant uint& block_size [[buffer(12)]],
    constant uint& max_num_blocks [[buffer(13)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]])
{
    uint h = tgid.x;
    uint batch_idx = tgid.y;
    
    uint simd_lane_id = tid.x % 32;
    uint simd_group_id = tid.x / 32;
    
    uint kv_h = h / (n_heads / n_kv_heads);
    int context_len = context_lens[batch_idx];
    
    // Shared memory
    threadgroup float shared_scores[32]; 
    threadgroup float shared_max_score;
    threadgroup float shared_sum_exp;
    if (tid.x == 0) {
        shared_sum_exp = 1.0f;
        shared_max_score = -1e9f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_val = 0.0f;
    if (tid.x < head_dim) {
        q_val = q[(batch_idx * n_heads + h) * head_dim + tid.x];
    }
    
    // Pass 1: Max Score
    float max_score = -1e9f;
    
    for (int t = 0; t < context_len; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[batch_idx * max_num_blocks + logical_block_idx];
            
            // Layout logic from paged_kv_cache:
            // Data: (block_id * block_size + block_offset) * n_kv_heads * head_dim
            // Scale: (block_id * block_size + block_offset) * n_kv_heads
            
            uint slot_idx = physical_block_id * block_size + block_offset;
            uint head_offset = kv_h;
            
            uint k_data_idx = (slot_idx * n_kv_heads + head_offset) * head_dim + tid.x;
            uint k_scale_idx = slot_idx * n_kv_heads + head_offset;
            
            float k_scale = k_scales[k_scale_idx];
            float k_q = (float)k_global[k_data_idx];
            k_val = k_q * k_scale;
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            max_score = max(max_score, score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_max_score = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = shared_max_score;
    
    // Pass 2: Sum Exp
    float sum_exp = 0.0f;
    
    for (int t = 0; t < context_len; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[batch_idx * max_num_blocks + logical_block_idx];
            
            uint slot_idx = physical_block_id * block_size + block_offset;
            uint head_offset = kv_h;
            
            uint k_data_idx = (slot_idx * n_kv_heads + head_offset) * head_dim + tid.x;
            uint k_scale_idx = slot_idx * n_kv_heads + head_offset;
            
            float k_scale = k_scales[k_scale_idx];
            k_val = ((float)k_global[k_data_idx]) * k_scale;
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            sum_exp += exp(score - max_score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_sum_exp = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_exp = shared_sum_exp;
    
    // Pass 3: Weighted Sum
    float weighted_sum = 0.0f;
    
    for (int t = 0; t < context_len; ++t) {
        float k_val = 0.0f;
        float v_val = 0.0f; // This is actually unused in the logic but good to keep structure? No, V is multiplied by weight.
        
        // Wait, Pass 3 iterates V. Not K.
        // We compute weight using K (recalculated) and mul by V.
        // Yes, standard attention re-computes attention score to save memory (FlashAttention style).
        
        // Re-compute Score
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[batch_idx * max_num_blocks + logical_block_idx];
            
            uint slot_idx = physical_block_id * block_size + block_offset;
            uint head_offset = kv_h;
            
            uint k_data_idx = (slot_idx * n_kv_heads + head_offset) * head_dim + tid.x;
            uint k_scale_idx = slot_idx * n_kv_heads + head_offset;
            
            float k_scale = k_scales[k_scale_idx];
            k_val = ((float)k_global[k_data_idx]) * k_scale;
            
            // V Access
            uint v_data_idx = (slot_idx * n_kv_heads + head_offset) * head_dim + tid.x;
            uint v_scale_idx = slot_idx * n_kv_heads + head_offset;
            
            float v_scale = v_scales[v_scale_idx];
            v_val = ((float)v_global[v_data_idx]) * v_scale;
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float weight = 0.0f;
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            weight = exp(score - max_score) / sum_exp;
            shared_scores[0] = weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        weight = shared_scores[0];
        weighted_sum += weight * v_val;
    }
    
    if (tid.x < head_dim) {
        output[(batch_idx * n_heads + h) * head_dim + tid.x] = weighted_sum;
    }
}

// GQA Attention Prefill Int8 (Ragged)
kernel void gqa_attention_prefill_int8(
    device const float* q [[buffer(0)]],
    device const char* k_global [[buffer(1)]],
    device const char* v_global [[buffer(2)]],
    device const float* k_scales [[buffer(3)]],
    device const float* v_scales [[buffer(4)]],
    device const int* block_tables [[buffer(5)]],
    device const int* cu_seqlens [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& n_seqs [[buffer(8)]],
    constant uint& n_heads [[buffer(9)]],
    constant uint& n_kv_heads [[buffer(10)]],
    constant uint& head_dim [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant uint& block_size [[buffer(13)]],
    constant uint& max_num_blocks [[buffer(14)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]])
{
    // Grid: (n_heads, total_tokens, 1)
    uint h = tgid.x;
    uint global_token_idx = tgid.y;
    
    // Find sequence index
    uint seq_idx = 0;
    for (uint i = 0; i < n_seqs; ++i) {
        if (global_token_idx < cu_seqlens[i+1]) {
            seq_idx = i;
            break;
        }
    }
    
    uint seq_start_token = cu_seqlens[seq_idx];
    uint current_pos = global_token_idx - seq_start_token;
    
    uint simd_lane_id = tid.x % 32;
    uint simd_group_id = tid.x / 32;
    uint kv_h = h / (n_heads / n_kv_heads);
    
    // Shared memory
    threadgroup float shared_scores[32]; 
    if (tid.x == 0) {
        shared_scores[0] = -1e9f; // shared_max_score
        shared_scores[1] = 1.0f;  // shared_sum_exp
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shared_max_score = shared_scores[0];
    float shared_sum_exp = shared_scores[1];

    if (tid.x == 0) {
        shared_max_score = -1e9f;
        shared_sum_exp = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_val = 0.0f;
    if (tid.x < head_dim) {
        q_val = q[(global_token_idx * n_heads + h) * head_dim + tid.x];
    }
    
    // Pass 1: Max Score
    float max_score = -1e9f;
    
    for (uint t = 0; t <= current_pos; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[seq_idx * max_num_blocks + logical_block_idx];
            
            uint slot_idx = physical_block_id * block_size + block_offset;
            uint head_offset = kv_h;
            
            uint k_data_idx = (slot_idx * n_kv_heads + head_offset) * head_dim + tid.x;
            uint k_scale_idx = slot_idx * n_kv_heads + head_offset;
            
            float k_scale = k_scales[k_scale_idx];
            k_val = ((float)k_global[k_data_idx]) * k_scale;
        }
        
        float partial_score = simd_sum(q_val * k_val);
        
        if (simd_lane_id == 0) {
            shared_scores[simd_group_id] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            max_score = max(max_score, score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_max_score = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = shared_max_score;
    
    // Pass 2: Sum Exp
    float sum_exp = 0.0f;
    for (uint t = 0; t <= current_pos; ++t) {
        float k_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[seq_idx * max_num_blocks + logical_block_idx];
            
            uint slot_idx = physical_block_id * block_size + block_offset;
            uint head_offset = kv_h;
            
            uint k_idx = (slot_idx * n_kv_heads + head_offset) * head_dim + tid.x;
            uint k_scale_idx = slot_idx * n_kv_heads + head_offset;
            
            float k_scale = k_scales[k_scale_idx];
            k_val = ((float)k_global[k_idx]) * k_scale;
        }
        
        float partial_score = simd_sum(q_val * k_val);
        if (simd_lane_id == 0) shared_scores[simd_group_id] = partial_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            sum_exp += exp(score - max_score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid.x == 0) shared_sum_exp = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_exp = shared_sum_exp;
    
    // Pass 3: Weighted Sum
    float weighted_sum = 0.0f;
    for (uint t = 0; t <= current_pos; ++t) {
        float k_val = 0.0f;
        float v_val = 0.0f;
        if (tid.x < head_dim) {
            uint logical_block_idx = t / block_size;
            uint block_offset = t % block_size;
            int physical_block_id = block_tables[seq_idx * max_num_blocks + logical_block_idx];
            
            uint slot_idx = physical_block_id * block_size + block_offset;
            uint head_offset = kv_h;
            
            uint data_idx = (slot_idx * n_kv_heads + head_offset) * head_dim + tid.x;
            uint scale_idx = slot_idx * n_kv_heads + head_offset;
            
            float k_scale = k_scales[scale_idx];
            float v_scale = v_scales[scale_idx];
            
            k_val = ((float)k_global[data_idx]) * k_scale;
            v_val = ((float)v_global[data_idx]) * v_scale;
        }

        float partial_score = simd_sum(q_val * k_val);
        if (simd_lane_id == 0) shared_scores[simd_group_id] = partial_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float weight = 0.0f;
        if (tid.x == 0) {
            float score = 0.0f;
            uint n_simd_groups = (threads_per_group.x + 31) / 32;
            for (uint i = 0; i < n_simd_groups; ++i) {
                score += shared_scores[i];
            }
            score *= scale;
            weight = exp(score - max_score) / sum_exp;
            shared_scores[0] = weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        weight = shared_scores[0];
        weighted_sum += weight * v_val;
    }

    if (tid.x < head_dim) {
        output[(global_token_idx * n_heads + h) * head_dim + tid.x] = weighted_sum;
    }
}

// =============================================================================
// MoE (Mixture-of-Experts) Kernels
// =============================================================================

// MoE Gate + Softmax
// Computes gate logits and applies softmax over all experts per token
// Input: hidden [seq_len, d_model], gate_weight [d_model, num_experts]
// Output: probs [seq_len, num_experts]
// Grid: (seq_len, 1, 1)
// Block: (32, 1, 1) - one threadgroup per token
kernel void moe_gate_softmax(
    device const float* hidden [[buffer(0)]],      // [seq_len, d_model]
    device const float* gate_weight [[buffer(1)]], // [d_model, num_experts]
    device float* probs [[buffer(2)]],             // [seq_len, num_experts]
    constant uint& d_model [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint seq_idx = tgid.x;
    uint lane = tid.x;

    // Shared memory for logits
    threadgroup float shared_logits[64]; // Assume max 64 experts
    threadgroup float shared_max;
    threadgroup float shared_sum;

    // Step 1: Compute gate logits (each thread handles one expert at a time)
    for (uint e = lane; e < num_experts; e += 32) {
        float dot = 0.0f;
        for (uint d = 0; d < d_model; ++d) {
            // hidden: [seq_len, d_model], gate_weight: [d_model, num_experts]
            dot += hidden[seq_idx * d_model + d] * gate_weight[d * num_experts + e];
        }
        shared_logits[e] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Find max for numerical stability
    float local_max = -INFINITY;
    for (uint e = lane; e < num_experts; e += 32) {
        local_max = max(local_max, shared_logits[e]);
    }
    local_max = simd_max(local_max);

    if (lane == 0) {
        shared_max = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared_max;

    // Step 3: Compute exp and sum
    float local_sum = 0.0f;
    for (uint e = lane; e < num_experts; e += 32) {
        float exp_val = exp(shared_logits[e] - max_val);
        shared_logits[e] = exp_val;
        local_sum += exp_val;
    }
    local_sum = simd_sum(local_sum);

    if (lane == 0) {
        shared_sum = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_exp = shared_sum;

    // Step 4: Normalize and write output
    for (uint e = lane; e < num_experts; e += 32) {
        probs[seq_idx * num_experts + e] = shared_logits[e] / sum_exp;
    }
}

// MoE Top-K Expert Selection
// Selects top-k experts for each token using insertion sort (efficient for small k)
// Input: probs [seq_len, num_experts]
// Output: indices [seq_len, k], values [seq_len, k]
// Grid: (seq_len, 1, 1)
// Block: (32, 1, 1)
kernel void moe_topk_experts(
    device const float* probs [[buffer(0)]],    // [seq_len, num_experts]
    device int* indices [[buffer(1)]],          // [seq_len, k]
    device float* values [[buffer(2)]],         // [seq_len, k]
    constant uint& num_experts [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& norm_topk [[buffer(5)]],     // 1 to normalize top-k probs
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint seq_idx = tgid.x;
    uint lane = tid.x;

    // Only thread 0 does the work (serial top-k for simplicity)
    // For small k (typically 2-8), this is fast enough
    if (lane != 0) return;

    // Read probabilities for this token
    device const float* token_probs = probs + seq_idx * num_experts;
    device int* out_indices = indices + seq_idx * k;
    device float* out_values = values + seq_idx * k;

    // Initialize with worst possible values
    for (uint i = 0; i < k; ++i) {
        out_values[i] = -INFINITY;
        out_indices[i] = -1;
    }

    // Insertion sort: maintain sorted top-k list
    for (uint e = 0; e < num_experts; ++e) {
        float prob = token_probs[e];

        // Check if this probability belongs in top-k
        if (prob > out_values[k - 1]) {
            // Find insertion position
            uint pos = k - 1;
            while (pos > 0 && prob > out_values[pos - 1]) {
                out_values[pos] = out_values[pos - 1];
                out_indices[pos] = out_indices[pos - 1];
                pos--;
            }
            out_values[pos] = prob;
            out_indices[pos] = (int)e;
        }
    }

    // Normalize top-k probabilities if requested
    if (norm_topk) {
        float sum = 0.0f;
        for (uint i = 0; i < k; ++i) {
            sum += out_values[i];
        }
        if (sum > 0.0f) {
            for (uint i = 0; i < k; ++i) {
                out_values[i] /= sum;
            }
        }
    }
}

// MoE Aggregate Expert Outputs
// Computes weighted sum of expert outputs
// Input: expert_outputs [num_experts, seq_len, d_model], probs [seq_len, k], indices [seq_len, k]
// Output: output [seq_len, d_model]
// Grid: (d_model, seq_len, 1)
// Block: (1, 1, 1) - each thread handles one output element
kernel void moe_aggregate(
    device const float* expert_outputs [[buffer(0)]],  // [num_experts, seq_len, d_model]
    device const float* probs [[buffer(1)]],           // [seq_len, k]
    device const int* indices [[buffer(2)]],           // [seq_len, k]
    device float* output [[buffer(3)]],                // [seq_len, d_model]
    constant uint& seq_len [[buffer(4)]],
    constant uint& d_model [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d_idx = gid.x;
    uint seq_idx = gid.y;

    if (d_idx >= d_model || seq_idx >= seq_len) return;

    float accum = 0.0f;

    for (uint i = 0; i < k; ++i) {
        int expert_id = indices[seq_idx * k + i];
        float prob = probs[seq_idx * k + i];

        // expert_outputs layout: [num_experts, seq_len, d_model]
        float expert_val = expert_outputs[(expert_id * seq_len + seq_idx) * d_model + d_idx];
        accum += prob * expert_val;
    }

    output[seq_idx * d_model + d_idx] = accum;
}

// SwiGLU FFN for MoE Expert
// Fused gate * silu(up) -> down projection
// This is called per-expert
// Input: x [seq_len, d_model]
// Weights: W_gate [d_model, d_ff], W_up [d_model, d_ff], W_down [d_ff, d_model]
// Output: y [seq_len, d_model]
// Grid: (d_model, seq_len, 1)
// Block: (32, 1, 1)
kernel void moe_swiglu_ffn(
    device const float* x [[buffer(0)]],        // [seq_len, d_model]
    device const float* W_gate [[buffer(1)]],   // [d_model, d_ff]
    device const float* W_up [[buffer(2)]],     // [d_model, d_ff]
    device const float* W_down [[buffer(3)]],   // [d_ff, d_model]
    device float* output [[buffer(4)]],         // [seq_len, d_model]
    device float* intermediate [[buffer(5)]],   // [seq_len, d_ff] scratch
    constant uint& seq_len [[buffer(6)]],
    constant uint& d_model [[buffer(7)]],
    constant uint& d_ff [[buffer(8)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    // This kernel computes one output element per threadgroup
    uint out_dim = tgid.x;
    uint seq_idx = tgid.y;
    uint lane = tid.x;

    if (out_dim >= d_model || seq_idx >= seq_len) return;

    // Step 1: Compute gate and up projections (in intermediate buffer)
    // We do this in a loop since d_ff is large

    float accum = 0.0f;

    // For the down projection, we need the intermediate values
    // intermediate[seq_idx, d] = silu(gate[d]) * up[d]
    // output[seq_idx, out_dim] = sum_d(intermediate[d] * W_down[d, out_dim])

    for (uint d = lane; d < d_ff; d += 32) {
        // Compute gate[d] = dot(x, W_gate[:, d])
        float gate_val = 0.0f;
        float up_val = 0.0f;
        for (uint i = 0; i < d_model; ++i) {
            float x_val = x[seq_idx * d_model + i];
            gate_val += x_val * W_gate[i * d_ff + d];
            up_val += x_val * W_up[i * d_ff + d];
        }

        // SiLU activation on gate
        float silu_gate = gate_val / (1.0f + exp(-gate_val));

        // Element-wise multiply
        float hidden = silu_gate * up_val;

        // Accumulate for down projection
        accum += hidden * W_down[d * d_model + out_dim];
    }

    // Reduce across threads
    accum = simd_sum(accum);

    if (lane == 0) {
        output[seq_idx * d_model + out_dim] = accum;
    }
}
