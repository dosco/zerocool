#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "infra/compute_backend.hpp"
#include "infra/metal_backend.hpp"
#include "infra/cuda_backend.hpp"
#include <vector>
#include <iostream>

using namespace freellm;
using namespace freellm::infra;

TEST_CASE("ComputeBackend Factory") {
    // Test Metal creation if on Apple
#ifdef __APPLE__
    auto backend = ComputeBackend::Create(DeviceType::METAL, 0);
    CHECK(backend != nullptr);
    CHECK(backend->type() == DeviceType::METAL);
#endif

    // Test CUDA creation if enabled
#ifdef FREELM_ENABLE_CUDA
    auto backend = ComputeBackend::Create(DeviceType::CUDA, 0);
    CHECK(backend != nullptr);
    CHECK(backend->device_type() == DeviceType::CUDA);
#endif
}

#ifdef __APPLE__
TEST_CASE("Metal Backend Operations") {
    auto backend = ComputeBackend::Create(DeviceType::METAL, 0);
    REQUIRE(backend != nullptr);

    SUBCASE("Memory Allocation and Copy") {
        size_t size = 1024 * sizeof(float);
        auto buf = backend->allocate(size, DType::FLOAT32);
        CHECK(buf != nullptr);
        CHECK(buf->size_bytes() == size);

        std::vector<float> host_data(1024, 1.0f);
        backend->copy_to_device(buf.get(), host_data.data(), size);

        std::vector<float> result_data(1024);
        backend->synchronize(); // Ensure copy is done (though copy is usually sync or has internal sync)
        backend->copy_to_host(result_data.data(), buf.get(), size);

        CHECK(result_data[0] == 1.0f);
        CHECK(result_data[1023] == 1.0f);
    }

    SUBCASE("Simple Kernel Execution (Add)") {
        // Test the 'add' kernel we just added
        size_t N = 100;
        size_t size = N * sizeof(float);
        
        auto a = backend->allocate(size, DType::FLOAT32);
        auto b = backend->allocate(size, DType::FLOAT32);
        auto out = backend->allocate(size, DType::FLOAT32);

        std::vector<float> host_a(N, 1.0f);
        std::vector<float> host_b(N, 2.0f);
        
        backend->copy_to_device(a.get(), host_a.data(), size);
        backend->copy_to_device(b.get(), host_b.data(), size);

        // Kernel args: inputs, outputs
        std::vector<DeviceBuffer*> inputs = {a.get(), b.get()};
        std::vector<DeviceBuffer*> outputs = {out.get()};
        
        // Grid size: N, 1, 1
        // Block size: 256, 1, 1 (standard)
        KernelConfig config;
        config.grid = Dim3(N, 1, 1);
        config.block = Dim3(256, 1, 1);

        backend->execute_kernel("add", inputs, outputs, config);
        backend->synchronize();

        std::vector<float> result(N);
        backend->copy_to_host(result.data(), out.get(), size);

        CHECK(result[0] == 3.0f);
        CHECK(result[99] == 3.0f);
    }
}
#endif
