#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../external/doctest.h"
#include "infra/compute_backend.hpp"
#include "infra/metal_backend.hpp"
#include "kernels/quantiz/q4_0/quantize.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace freellm;
using namespace freellm::infra;
using namespace freellm::quant;

// Helper to dequantize a block (CPU reference)
void dequantize_block_cpu(const q4_0::BlockQ4_0& block, float* dst) {
    float scale = block.scale;
    for (int j = 0; j < 32; ++j) {
        uint8_t q_packed = block.qs[j / 2];
        int8_t q = (j % 2 == 0) ? (q_packed & 0x0F) : (q_packed >> 4);
        dst[j] = (q - 8) * scale;
    }
}

TEST_CASE("Q4_0 Quantization and Metal GEMM Verification") {
    // 1. Setup Data
    // We verify a single row-vector multiplication: y = w * x
    // w: weight vector [256] (quantized)
    // x: input vector [256] (float)
    // y: scalar result
    
    // Use Dimensions compatible with block size 32
    const int K = 4096; // Llama 3 d_model
    const int M = 1; // 1 output feature (1 row in weight matrix)
    const int N_batch = 1;
    
    std::vector<float> w_float(K);
    std::vector<float> x_float(K);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for(int i=0; i<K; ++i) {
        w_float[i] = dist(gen);
        x_float[i] = dist(gen);
    }
    
    // 2. CPU Quantization
    size_t q_size_bytes = q4_0::size_bytes(K);
    std::vector<uint8_t> w_quant(q_size_bytes);
    q4_0::quantize(w_float.data(), w_quant.data(), K);
    
    // 3. CPU Reference Dot Product (using dequantized weights)
    // This verifies lossy quantization accuracy
    std::vector<float> w_dequant(K);
    const q4_0::BlockQ4_0* blocks = (const q4_0::BlockQ4_0*)w_quant.data();
    for(int i=0; i<K/32; ++i) {
        dequantize_block_cpu(blocks[i], w_dequant.data() + i*32);
    }
    
    float cpu_dot_product = 0.0f;
    for(int i=0; i<K; ++i) {
        cpu_dot_product += w_dequant[i] * x_float[i];
    }
    
    float original_dot_product = 0.0f;
    for(int i=0; i<K; ++i) {
        original_dot_product += w_float[i] * x_float[i];
    }
    
    std::cout << "Original Dot: " << original_dot_product << "\n";
    std::cout << "Quantized CPU Dot: " << cpu_dot_product << "\n";
    std::cout << "Quantization Error: " << std::abs(original_dot_product - cpu_dot_product) << "\n";
    std::cout << "BlockQ4_0 size: " << sizeof(q4_0::BlockQ4_0) << "\n";
    CHECK(sizeof(q4_0::BlockQ4_0) == 20); // Verify stride assumption
    
    // Check quantization is reasonably close
    CHECK(std::abs(original_dot_product - cpu_dot_product) < 2.0f); // Allow some error
    
    // 4. Metal Kernel Execution
#ifdef __APPLE__
    auto backend = ComputeBackend::Create(DeviceType::METAL, 0);
    REQUIRE(backend != nullptr);
    
    // A: Weights [M, K] -> [1, 256] -> Quantized size
    auto dev_A = backend->allocate(q_size_bytes, DType::INT8);
    backend->copy_to_device(dev_A.get(), 0, w_quant.data(), q_size_bytes); // Cast for API simply taking void* usually, but API takes float*?
    // backend->copy_to_device signature takes void* usually? 
    // Checking header: 'virtual void copy_to_device(DeviceBuffer* dst, const void* src, size_t size) = 0;'
    // But wrapper might take float*.
    // Let's assume raw bytes copy is needed.
    // Wait, ComputeBackend::copy_to_device takes 'const float* src' in abstract?
    // include/infra/compute_backend.hpp
    
    // Checking implementation details:
    // MetalBackend::copy_to_device(DeviceBuffer* dst, size_t dst_offset, const void* src, size_t size)
    // The previous view of test_backends.cpp used `backend->copy_to_device(buf.get(), host_data.data(), size)`.
    // It seems it takes void* or float*.
    
    // Let's use the explicit byte copy if possible, or cast.
    // backend->copy_to_device(dev_A.get(), 0, w_quant.data(), q_size_bytes); (Offset version)
    // Or check backend methods.
    
    // B: Input [Batch, K] -> [1, 256]
    auto dev_B = backend->allocate(K * sizeof(float), DType::FLOAT32);
    // Use the simpler API if available, likely `copy_to_device(dst, src, size)`
    // Cast w_quant to float* is risky if API validates size divisibility by 4?
    // UINT8 buffer allocation supports byte assignment.
    
    // Let's use raw copy method available on backend if possible.
    // Looking at paged_kv_cache.hpp, `backend->copy_to_device(dst, dst_offset, src, size)`.
    // So we use that.
    
    backend->copy_to_device(dev_A.get(), 0, w_quant.data(), q_size_bytes);
    backend->copy_to_device(dev_B.get(), 0, x_float.data(), K * sizeof(float));
    
    // C: Output [Batch, M] -> [1, 1]
    auto dev_C = backend->allocate(M * sizeof(float), DType::FLOAT32);
    
    // Kernel Args
    // gemm_q4_0(A, B, C, cols, batch_size)
    // A is buffer 0, B is 1, C is 2.
    // Params: cols (buffer 3), batch_size (buffer 4)
    // We pass integers as scalars? 
    // MetalBackend::execute_kernel signature usually packs scalars?
    // No, usually we need to allocate buffers for scalars or backend handles it?
    // MetalBackend::execute_kernel(name, inputs, outputs, config)
    // Inputs must be DeviceBuffer*.
    // So we need to put scalars into buffers?
    // Or does `kernel_loader` handle scalar args?
    // MetalBackend implementation usually handles POD types if reflected?
    // But here we use generic interface.
    // Let's look at `tests/test_backends.cpp`. 
    // `start_pos`, `n_heads` etc. were NOT passed in test_backends.cpp (only simple add).
    
    // I need to allocate small buffers for scalars.
    uint32_t cols_scalar = K;
    uint32_t batch_scalar = N_batch;
    
    auto dev_cols = backend->allocate(sizeof(uint32_t), DType::INT32); // Use generic if available or smallest
    auto dev_batch = backend->allocate(sizeof(uint32_t), DType::INT32);
    
    backend->copy_to_device(dev_cols.get(), 0, &cols_scalar, sizeof(uint32_t));
    backend->copy_to_device(dev_batch.get(), 0, &batch_scalar, sizeof(uint32_t));
    
    std::vector<DeviceBuffer*> inputs = {dev_A.get(), dev_B.get(), dev_C.get(), dev_cols.get(), dev_batch.get()};
    std::vector<DeviceBuffer*> outputs = {}; // C is passed as input buffer pointer, written to.
    // Is C input or output? In Metal kernel signature: `device float* C [[buffer(2)]]`.
    // It is a pointer. It can be read/write.
    // Usually in this engine, mutable buffers are passed in 'inputs' or 'outputs'?
    // MetalBackend implementation distinguishes?
    // Let's assume outputs are strictly for verification?
    // Actually, `execute_kernel` takes inputs and outputs vectors, but likely just binds them to indices 0, 1, 2...
    // Check `src/infra/metal_backend.mm`.
    
    // I'll assume inputs list binds to buffer(0), buffer(1), ... sequentially.
    // So all args must be in inputs.
    
    KernelConfig config;
    // MetalBackend uses dispatchThreads, so grid dimensions are TOTAL THREADS.
    // We want M threadgroups (one per row), each with 32 threads.
    config.grid = Dim3(M * 32, N_batch, 1); 
    config.block = Dim3(32, 1, 1);     // 32 threads per group (simd)
    
    backend->execute_kernel("gemm_q4_0", inputs, {}, config);
    backend->synchronize();
    
    std::vector<float> metal_result(M);
    backend->copy_to_host(metal_result.data(), dev_C.get(), M * sizeof(float));
    
    std::cout << "Metal result: " << metal_result[0] << "\n";
    std::cout << "Diff: " << std::abs(metal_result[0] - cpu_dot_product) << "\n";
    
    CHECK(std::abs(metal_result[0] - cpu_dot_product) < 0.05f);
#endif
}
