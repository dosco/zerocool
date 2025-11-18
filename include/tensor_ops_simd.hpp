#pragma once

#include "tensor.hpp"
#include "cpu_features.hpp"
#include <cmath>
#include <algorithm>

// Platform-specific SIMD intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(__clang__) || defined(__GNUC__)
        #include <x86intrin.h>
    #elif defined(_MSC_VER)
        #include <immintrin.h>
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
#endif

namespace freellm {
namespace ops {
namespace simd {

// ============================================================================
// AVX2 Optimized Matrix Multiplication
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)

/**
 * @brief AVX2-optimized matrix multiplication: C = A × B
 *
 * Optimization techniques used:
 * 1. **Vectorization**: Process 8 floats at once using AVX2 (256-bit registers)
 * 2. **Cache blocking**: Tile matrices to fit in L1/L2 cache
 * 3. **Loop reordering**: i-k-j order for better memory access pattern
 * 4. **FMA instructions**: Fused multiply-add for fewer operations
 *
 * Performance expectations:
 * - Naive: ~0.5 GFLOPS (simple triple loop)
 * - AVX2:  ~5-10 GFLOPS (10-20x speedup on modern CPUs)
 * - Theoretical peak on 4 GHz CPU with AVX2: ~128 GFLOPS (32 FLOPs/cycle * 4 GHz)
 *
 * @param a Matrix A with shape [M, K]
 * @param b Matrix B with shape [K, N]
 * @return Result matrix C with shape [M, N]
 */
inline Tensor matmul_avx2(const Tensor& a, const Tensor& b) {
    // Validate inputs
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul_avx2 requires 2D tensors");
    }

    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    if (b.shape()[0] != K) {
        throw std::invalid_argument("matmul_avx2: inner dimensions must match");
    }

    Tensor result({M, N});
    result.zero();

    const float* a_data = a.data();
    const float* b_data = b.data();
    float* c_data = result.data();

    // Cache blocking parameters
    // L1 cache: ~32KB per core
    // We want A_block + B_block + C_block to fit in L1
    constexpr size_t BLOCK_SIZE = 64;  // Tune this for your CPU

    // Vectorization width for AVX2 (8 floats in 256-bit register)
    constexpr size_t SIMD_WIDTH = 8;

    // Blocked matrix multiplication with i-k-j loop order
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        size_t i_end = std::min(i0 + BLOCK_SIZE, M);

        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            size_t k_end = std::min(k0 + BLOCK_SIZE, K);

            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                size_t j_end = std::min(j0 + BLOCK_SIZE, N);

                // Process this block
                for (size_t i = i0; i < i_end; ++i) {
                    for (size_t k = k0; k < k_end; ++k) {
                        // Broadcast a[i, k] to all 8 elements of AVX2 register
                        __m256 a_ik = _mm256_set1_ps(a_data[i * K + k]);

                        size_t j = j0;

                        // Vectorized loop: process 8 elements at a time
                        for (; j + SIMD_WIDTH <= j_end; j += SIMD_WIDTH) {
                            // Load 8 elements from B[k, j:j+8]
                            __m256 b_kj = _mm256_loadu_ps(&b_data[k * N + j]);

                            // Load current C[i, j:j+8]
                            __m256 c_ij = _mm256_loadu_ps(&c_data[i * N + j]);

                            // FMA: C[i,j] += A[i,k] * B[k,j]
                            // This is a single instruction on modern CPUs
                            #ifdef __FMA__
                            c_ij = _mm256_fmadd_ps(a_ik, b_kj, c_ij);
                            #else
                            __m256 prod = _mm256_mul_ps(a_ik, b_kj);
                            c_ij = _mm256_add_ps(c_ij, prod);
                            #endif

                            // Store result back
                            _mm256_storeu_ps(&c_data[i * N + j], c_ij);
                        }

                        // Handle remaining elements (scalar tail)
                        for (; j < j_end; ++j) {
                            c_data[i * N + j] += a_data[i * K + k] * b_data[k * N + j];
                        }
                    }
                }
            }
        }
    }

    return result;
}

/**
 * @brief AVX2-optimized matrix-vector multiplication: y = A × x
 *
 * This is a critical operation for LLM inference as every linear layer
 * performs matrix-vector multiplication during single-token generation.
 *
 * @param mat Matrix with shape [M, N]
 * @param vec Vector with shape [N] or [N, 1]
 * @return Result vector with shape [M]
 */
inline Tensor matvec_avx2(const Tensor& mat, const Tensor& vec) {
    if (mat.ndim() != 2) {
        throw std::invalid_argument("matvec_avx2: matrix must be 2D");
    }

    size_t M = mat.shape()[0];
    size_t N = mat.shape()[1];

    size_t vec_size = (vec.ndim() == 1) ? vec.shape()[0] : vec.shape()[0] * vec.shape()[1];
    if (vec_size != N) {
        throw std::invalid_argument("matvec_avx2: dimension mismatch");
    }

    Tensor result({M});
    const float* mat_data = mat.data();
    const float* vec_data = vec.data();
    float* result_data = result.data();

    constexpr size_t SIMD_WIDTH = 8;

    // For each row of the matrix
    for (size_t i = 0; i < M; ++i) {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t j = 0;

        // Vectorized accumulation
        for (; j + SIMD_WIDTH <= N; j += SIMD_WIDTH) {
            __m256 mat_vals = _mm256_loadu_ps(&mat_data[i * N + j]);
            __m256 vec_vals = _mm256_loadu_ps(&vec_data[j]);

            #ifdef __FMA__
            sum_vec = _mm256_fmadd_ps(mat_vals, vec_vals, sum_vec);
            #else
            __m256 prod = _mm256_mul_ps(mat_vals, vec_vals);
            sum_vec = _mm256_add_ps(sum_vec, prod);
            #endif
        }

        // Horizontal sum of the 8 accumulated values
        // sum_vec contains [a, b, c, d, e, f, g, h]
        __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);  // [e, f, g, h]
        __m128 sum_low = _mm256_castps256_ps128(sum_vec);     // [a, b, c, d]
        __m128 sum_128 = _mm_add_ps(sum_low, sum_high);        // [a+e, b+f, c+g, d+h]

        sum_128 = _mm_hadd_ps(sum_128, sum_128);  // [a+e+b+f, c+g+d+h, ...]
        sum_128 = _mm_hadd_ps(sum_128, sum_128);  // [sum_all, ...]

        float sum = _mm_cvtss_f32(sum_128);

        // Scalar tail
        for (; j < N; ++j) {
            sum += mat_data[i * N + j] * vec_data[j];
        }

        result_data[i] = sum;
    }

    return result;
}

/**
 * @brief AVX2-optimized element-wise operations
 */
inline Tensor add_avx2(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have same shape");
    }

    Tensor result(a.shape(), a.dtype());
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();

    constexpr size_t SIMD_WIDTH = 8;
    size_t i = 0;
    size_t size = a.size();

    // Vectorized loop
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m256 a_vec = _mm256_loadu_ps(&a_data[i]);
        __m256 b_vec = _mm256_loadu_ps(&b_data[i]);
        __m256 result_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result_data[i], result_vec);
    }

    // Scalar tail
    for (; i < size; ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }

    return result;
}

inline Tensor multiply_avx2(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have same shape");
    }

    Tensor result(a.shape(), a.dtype());
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();

    constexpr size_t SIMD_WIDTH = 8;
    size_t i = 0;
    size_t size = a.size();

    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m256 a_vec = _mm256_loadu_ps(&a_data[i]);
        __m256 b_vec = _mm256_loadu_ps(&b_data[i]);
        __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result_data[i], result_vec);
    }

    for (; i < size; ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }

    return result;
}

#else
// Fallback: if not on x86_64, just use naive implementations
// (ARM NEON implementations could be added here in the future)

inline Tensor matmul_avx2(const Tensor& a, const Tensor& b) {
    // This will be defined in tensor_ops.hpp
    return matmul_naive(a, b);
}

inline Tensor matvec_avx2(const Tensor& mat, const Tensor& vec) {
    return matvec_naive(mat, vec);
}

inline Tensor add_avx2(const Tensor& a, const Tensor& b) {
    return add(a, b);
}

inline Tensor multiply_avx2(const Tensor& a, const Tensor& b) {
    return multiply(a, b);
}

#endif // x86_64

} // namespace simd
} // namespace ops
} // namespace freellm
