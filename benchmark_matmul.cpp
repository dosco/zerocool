#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include "../include/tensor.hpp"
#include "../include/tensor_ops.hpp"
#include "../include/tensor_ops_simd.hpp"
#include "../include/cpu_features.hpp"

using namespace freellm;

void benchmark(const std::string& name, size_t M, size_t K, size_t N, 
               std::function<Tensor(const Tensor&, const Tensor&)> func) {
    std::cout << "Benchmarking " << name << " [" << M << "x" << K << "x" << N << "]..." << std::endl;

    Tensor A({M, K});
    Tensor B({K, N});
    
    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for(size_t i=0; i<A.size(); ++i) A.data()[i] = dist(gen);
    for(size_t i=0; i<B.size(); ++i) B.data()[i] = dist(gen);

    // Warmup
    func(A, B);

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 5;
    for(int i=0; i<iterations; ++i) {
        Tensor C = func(A, B);
        // Prevent optimization
        if(C.data()[0] > 1000000) std::cout << ""; 
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double avg_time = diff.count() / iterations;
    double gflops = (2.0 * M * N * K) / (avg_time * 1e9);

    std::cout << "  Time: " << std::fixed << std::setprecision(4) << avg_time << " s" << std::endl;
    std::cout << "  GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
}

int main() {
    const size_t M = 512;
    const size_t K = 512;
    const size_t N = 512;

    benchmark("Naive", M, K, N, ops::matmul_naive);
    
    const auto& features = cpu::get_cpu_features();
    if (features.avx2) {
        benchmark("AVX2", M, K, N, ops::simd::matmul_avx2);
    } else if (features.neon) {
        benchmark("NEON (Current)", M, K, N, ops::simd::matmul_neon);
    } else {
        std::cout << "SIMD not supported on this machine." << std::endl;
    }

    return 0;
}
