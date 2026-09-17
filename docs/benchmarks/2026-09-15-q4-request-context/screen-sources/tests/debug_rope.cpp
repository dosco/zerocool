#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../external/doctest.h"
#include "infra/compute_backend.hpp"
#include "infra/metal_backend.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace freellm;
using namespace freellm::infra;

// CPU RoPE Reference
void rope_cpu(float* data, const float* cos_table, const float* sin_table, 
             int head_dim, int n_heads, int pos, int batch_idx, int n_batch) {
    // data shape: [batch, n_heads, head_dim] (flattened if batch sequential)
    // Here we assume simple [1, n_heads, head_dim] for single test vector
    
    // Rotations applied to pairs (i, i+1) for i in 0..head_dim/2
    int half_dim = head_dim / 2;
    
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < half_dim; ++i) {
            int i1 = h * head_dim + 2 * i;
            int i2 = i1 + 1;
            
            float x1 = data[i1];
            float x2 = data[i2];
            
            // Table index
            // Table shape: [max_seq, half_dim]
            int table_idx = pos * half_dim + i;
            
            float c = cos_table[table_idx];
            float s = sin_table[table_idx];
            
            data[i1] = x1 * c - x2 * s;
            data[i2] = x2 * c + x1 * s;
        }
    }
}

TEST_CASE("RoPE Verification") {
    // 1. Setup Data
    const int N_HEADS = 4;
    const int HEAD_DIM = 64;
    const int SEQ_LEN = 1; // Testing 1 token
    const int POS = 10;    // Arbitrary position
    
    std::vector<float> q_float(N_HEADS * HEAD_DIM);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for(auto& v : q_float) v = dist(gen);
    
    // Precompute cos/sin tables for just this pos
    int half_dim = HEAD_DIM / 2;
    // We need table for at least up to POS
    int table_len = (POS + 1) * half_dim;
    std::vector<float> cos_table(table_len);
    std::vector<float> sin_table(table_len);
    
    for (int p = 0; p <= POS; ++p) {
        for (int i = 0; i < half_dim; ++i) {
             double theta = 1.0 / std::pow(10000.0, 2.0 * i / HEAD_DIM);
             float angle = p * theta;
             cos_table[p * half_dim + i] = std::cos(angle);
             sin_table[p * half_dim + i] = std::sin(angle);
        }
    }
    
    // 2. CPU Reference
    std::vector<float> q_cpu = q_float;
    rope_cpu(q_cpu.data(), cos_table.data(), sin_table.data(), HEAD_DIM, N_HEADS, POS, 0, 1);
    
    // 3. Metal Execution
#ifdef __APPLE__
    auto backend = ComputeBackend::Create(DeviceType::METAL, 0);
    REQUIRE(backend != nullptr);
    
    auto dev_q = backend->allocate(q_float.size() * sizeof(float), DType::FLOAT32);
    backend->copy_to_device(dev_q.get(), 0, q_float.data(), q_float.size() * sizeof(float));
    
    auto dev_cos = backend->allocate(cos_table.size() * sizeof(float), DType::FLOAT32);
    auto dev_sin = backend->allocate(sin_table.size() * sizeof(float), DType::FLOAT32);
    backend->copy_to_device(dev_cos.get(), 0, cos_table.data(), cos_table.size() * sizeof(float));
    backend->copy_to_device(dev_sin.get(), 0, sin_table.data(), sin_table.size() * sizeof(float));
    
    // Position buffer for rope_ragged
    std::vector<int32_t> positions = {POS};
    auto dev_pos = backend->allocate(positions.size() * sizeof(int32_t), DType::INT32);
    backend->copy_to_device(dev_pos.get(), 0, positions.data(), positions.size() * sizeof(int32_t));
    
    // rope_ragged(input, cos, sin, positions, output, head_dim, n_heads)
    // Grid: (head_dim/2, n_heads, batch_size)
    // Code in GenerationEngine: 
    // grid_rope.grid = Dim3(head_dim / 2, n_heads, batch_size);
    
    KernelConfig config;
    config.grid = Dim3(HEAD_DIM / 2, N_HEADS, 1);
    config.block = Dim3(32, 1, 1);
    
    uint32_t head_dim_scalar = HEAD_DIM;
    uint32_t n_heads_scalar = N_HEADS;
    auto dev_hd = backend->allocate(sizeof(uint32_t), DType::INT32);
    auto dev_nh = backend->allocate(sizeof(uint32_t), DType::INT32);
    backend->copy_to_device(dev_hd.get(), 0, &head_dim_scalar, sizeof(uint32_t));
    backend->copy_to_device(dev_nh.get(), 0, &n_heads_scalar, sizeof(uint32_t));
    
    // Inputs: q, cos, sin, pos, output(q), head_dim(scalar-buf), n_heads(scalar-buf)
    // We pass scalars as buffers in 'inputs' list.
    // MetalBackend binds inputs effectively to buffer indices 0,1,2,3,4,5,6.
    
    // Note: 'dev_q' is passed as input(0) and output(4).
    std::vector<DeviceBuffer*> inputs = {
        dev_q.get(), dev_cos.get(), dev_sin.get(), dev_pos.get(), // 0,1,2,3
        dev_q.get(), // 4 (output)
        dev_hd.get(), dev_nh.get() // 5,6 (scalars)
    };
    
    backend->execute_kernel("rope_ragged", 
        inputs, 
        {}, // No separate outputs list, everything is in inputs for manual binding control
        config
        // No params
    );
    
    backend->synchronize();
    
    std::vector<float> q_metal(q_float.size());
    backend->copy_to_host(q_metal.data(), dev_q.get(), q_metal.size() * sizeof(float));
    
    // 4. Compare
    for (size_t i = 0; i < q_cpu.size(); ++i) {
        float diff = std::abs(q_cpu[i] - q_metal[i]);
        if (diff > 1e-4f) {
            std::cout << "Mismatch at " << i << ": CPU " << q_cpu[i] << " Metal " << q_metal[i] << " Diff " << diff << "\n";
            CHECK(diff < 1e-4f);
            break; // Stop after first failure
        }
    }
    CHECK(true); // If loop completes
#endif
}
