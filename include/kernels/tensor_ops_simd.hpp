#pragma once

#include "core/tensor.hpp"
#include "kernels/cpu_features.hpp"
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
    constexpr size_t BLOCK_SIZE_M = 64;
    constexpr size_t BLOCK_SIZE_K = 64;
    constexpr size_t BLOCK_SIZE_N = 64;

    // Register blocking: 4 rows of A/C processed at once
    constexpr size_t REG_BLOCK_M = 4;
    // AVX2 processes 8 floats at once
    constexpr size_t SIMD_WIDTH = 8;

    // Blocked matrix multiplication with i-k-j loop order
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE_M) {
        size_t i_end = std::min(i0 + BLOCK_SIZE_M, M);

        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE_K) {
            size_t k_end = std::min(k0 + BLOCK_SIZE_K, K);

            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE_N) {
                size_t j_end = std::min(j0 + BLOCK_SIZE_N, N);

                // Process 4 rows of A at a time
                size_t i = i0;
                for (; i + REG_BLOCK_M <= i_end; i += REG_BLOCK_M) {
                    for (size_t k = k0; k < k_end; ++k) {
                        // Broadcast 4 scalar values from A: a[i, k]...a[i+3, k]
                        __m256 a_val0 = _mm256_set1_ps(a_data[i * K + k]);
                        __m256 a_val1 = _mm256_set1_ps(a_data[(i + 1) * K + k]);
                        __m256 a_val2 = _mm256_set1_ps(a_data[(i + 2) * K + k]);
                        __m256 a_val3 = _mm256_set1_ps(a_data[(i + 3) * K + k]);

                        size_t j = j0;
                        // Vectorized loop: process 8 elements at a time
                        for (; j + SIMD_WIDTH <= j_end; j += SIMD_WIDTH) {
                            // Load 8 elements from B[k, j:j+8]
                            __m256 b_vec = _mm256_loadu_ps(&b_data[k * N + j]);

                            // Load current C[i...i+3, j:j+8]
                            __m256 c_vec0 = _mm256_loadu_ps(&c_data[i * N + j]);
                            __m256 c_vec1 = _mm256_loadu_ps(&c_data[(i + 1) * N + j]);
                            __m256 c_vec2 = _mm256_loadu_ps(&c_data[(i + 2) * N + j]);
                            __m256 c_vec3 = _mm256_loadu_ps(&c_data[(i + 3) * N + j]);

                            // FMA: C += A * B
                            #ifdef __FMA__
                            c_vec0 = _mm256_fmadd_ps(a_val0, b_vec, c_vec0);
                            c_vec1 = _mm256_fmadd_ps(a_val1, b_vec, c_vec1);
                            c_vec2 = _mm256_fmadd_ps(a_val2, b_vec, c_vec2);
                            c_vec3 = _mm256_fmadd_ps(a_val3, b_vec, c_vec3);
                            #else
                            c_vec0 = _mm256_add_ps(c_vec0, _mm256_mul_ps(a_val0, b_vec));
                            c_vec1 = _mm256_add_ps(c_vec1, _mm256_mul_ps(a_val1, b_vec));
                            c_vec2 = _mm256_add_ps(c_vec2, _mm256_mul_ps(a_val2, b_vec));
                            c_vec3 = _mm256_add_ps(c_vec3, _mm256_mul_ps(a_val3, b_vec));
                            #endif

                            // Store result back
                            _mm256_storeu_ps(&c_data[i * N + j], c_vec0);
                            _mm256_storeu_ps(&c_data[(i + 1) * N + j], c_vec1);
                            _mm256_storeu_ps(&c_data[(i + 2) * N + j], c_vec2);
                            _mm256_storeu_ps(&c_data[(i + 3) * N + j], c_vec3);
                        }

                        // Handle remaining elements (scalar tail)
                        for (; j < j_end; ++j) {
                            float b_val = b_data[k * N + j];
                            c_data[i * N + j]       += a_data[i * K + k] * b_val;
                            c_data[(i + 1) * N + j] += a_data[(i + 1) * K + k] * b_val;
                            c_data[(i + 2) * N + j] += a_data[(i + 2) * K + k] * b_val;
                            c_data[(i + 3) * N + j] += a_data[(i + 3) * K + k] * b_val;
                        }
                    }
                }

                // Handle remaining rows of A (less than 4)
                for (; i < i_end; ++i) {
                    for (size_t k = k0; k < k_end; ++k) {
                        __m256 a_ik = _mm256_set1_ps(a_data[i * K + k]);
                        size_t j = j0;

                        for (; j + SIMD_WIDTH <= j_end; j += SIMD_WIDTH) {
                            __m256 b_kj = _mm256_loadu_ps(&b_data[k * N + j]);
                            __m256 c_ij = _mm256_loadu_ps(&c_data[i * N + j]);

                            #ifdef __FMA__
                            c_ij = _mm256_fmadd_ps(a_ik, b_kj, c_ij);
                            #else
                            c_ij = _mm256_add_ps(c_ij, _mm256_mul_ps(a_ik, b_kj));
                            #endif

                            _mm256_storeu_ps(&c_data[i * N + j], c_ij);
                        }

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

/**
 * @brief AVX2-optimized softmax along last dimension
 */
inline Tensor softmax_avx2(const Tensor& input, int dim) {
    if (input.empty()) {
        throw std::invalid_argument("softmax_avx2: input tensor is empty");
    }
    if (input.ndim() != 2 || (dim != -1 && dim != 1)) {
        throw std::invalid_argument("softmax_avx2: currently only supports 2D tensors along last dimension");
    }

    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];

    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    float* dst = result.data();

    // Apply softmax to each row
    for (size_t i = 0; i < rows; ++i) {
        const float* row = &src[i * cols];
        float* out_row = &dst[i * cols];

        // 1. Find max (Vectorized)
        float max_val = -std::numeric_limits<float>::infinity();
        size_t j = 0;

        if (cols >= 8) {
            __m256 max_vec = _mm256_loadu_ps(row);
            j = 8;
            for (; j + 8 <= cols; j += 8) {
                __m256 val = _mm256_loadu_ps(row + j);
                max_vec = _mm256_max_ps(max_vec, val);
            }
            
            // Horizontal max reduction
            __m128 max_low = _mm256_castps256_ps128(max_vec);
            __m128 max_high = _mm256_extractf128_ps(max_vec, 1);
            max_low = _mm_max_ps(max_low, max_high);
            __m128 temp = _mm_movehl_ps(max_low, max_low);
            max_low = _mm_max_ps(max_low, temp);
            temp = _mm_shuffle_ps(max_low, max_low, _MM_SHUFFLE(1, 1, 1, 1));
            max_low = _mm_max_ps(max_low, temp);
            max_val = _mm_cvtss_f32(max_low);
        }

        // Scalar tail for max
        for (; j < cols; ++j) {
            max_val = std::max(max_val, row[j]);
        }

        // 2. Compute exp(x - max) and sum
        float sum_exp = 0.0f;
        for (j = 0; j < cols; ++j) {
            out_row[j] = std::exp(row[j] - max_val);
            sum_exp += out_row[j];
        }

        // 3. Normalize (Vectorized)
        float scale = 1.0f / sum_exp;
        __m256 scale_vec = _mm256_set1_ps(scale);

        j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 val = _mm256_loadu_ps(out_row + j);
            val = _mm256_mul_ps(val, scale_vec);
            _mm256_storeu_ps(out_row + j, val);
        }

        // Scalar tail for normalization
        for (; j < cols; ++j) {
            out_row[j] *= scale;
        }
    }

    return result;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

inline Tensor matmul_neon(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) throw std::invalid_argument("matmul_neon requires 2D tensors");
    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];
    if (b.shape()[0] != K) throw std::invalid_argument("matmul_neon: inner dimensions must match");

    Tensor result({M, N});
    result.zero();
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* c_data = result.data();

    // Cache blocking parameters
    constexpr size_t BLOCK_SIZE_M = 64;
    constexpr size_t BLOCK_SIZE_K = 64;
    constexpr size_t BLOCK_SIZE_N = 64;
    
    // Register blocking: 4 rows of A/C processed at once
    constexpr size_t REG_BLOCK_M = 4;

    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE_M) {
        size_t i_end = std::min(i0 + BLOCK_SIZE_M, M);
        
        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE_K) {
            size_t k_end = std::min(k0 + BLOCK_SIZE_K, K);
            
            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE_N) {
                size_t j_end = std::min(j0 + BLOCK_SIZE_N, N);

                // Process 4 rows of A at a time
                size_t i = i0;
                for (; i + REG_BLOCK_M <= i_end; i += REG_BLOCK_M) {
                    for (size_t k = k0; k < k_end; ++k) {
                        // Load 4 scalar values from A: a[i, k], a[i+1, k], a[i+2, k], a[i+3, k]
                        float32x4_t a_val0 = vdupq_n_f32(a_data[i * K + k]);
                        float32x4_t a_val1 = vdupq_n_f32(a_data[(i + 1) * K + k]);
                        float32x4_t a_val2 = vdupq_n_f32(a_data[(i + 2) * K + k]);
                        float32x4_t a_val3 = vdupq_n_f32(a_data[(i + 3) * K + k]);

                        size_t j = j0;
                        // Vectorized loop over N
                        for (; j + 4 <= j_end; j += 4) {
                            // Load 4 values from B
                            float32x4_t b_vec = vld1q_f32(&b_data[k * N + j]);

                            // Load current C values
                            float32x4_t c_vec0 = vld1q_f32(&c_data[i * N + j]);
                            float32x4_t c_vec1 = vld1q_f32(&c_data[(i + 1) * N + j]);
                            float32x4_t c_vec2 = vld1q_f32(&c_data[(i + 2) * N + j]);
                            float32x4_t c_vec3 = vld1q_f32(&c_data[(i + 3) * N + j]);

                            // FMA: C += A * B
                            c_vec0 = vfmaq_f32(c_vec0, a_val0, b_vec);
                            c_vec1 = vfmaq_f32(c_vec1, a_val1, b_vec);
                            c_vec2 = vfmaq_f32(c_vec2, a_val2, b_vec);
                            c_vec3 = vfmaq_f32(c_vec3, a_val3, b_vec);

                            // Store back
                            vst1q_f32(&c_data[i * N + j], c_vec0);
                            vst1q_f32(&c_data[(i + 1) * N + j], c_vec1);
                            vst1q_f32(&c_data[(i + 2) * N + j], c_vec2);
                            vst1q_f32(&c_data[(i + 3) * N + j], c_vec3);
                        }
                        
                        // Handle scalar tail for N
                        for (; j < j_end; ++j) {
                            float b_val = b_data[k * N + j];
                            c_data[i * N + j]       += a_data[i * K + k] * b_val;
                            c_data[(i + 1) * N + j] += a_data[(i + 1) * K + k] * b_val;
                            c_data[(i + 2) * N + j] += a_data[(i + 2) * K + k] * b_val;
                            c_data[(i + 3) * N + j] += a_data[(i + 3) * K + k] * b_val;
                        }
                    }
                }

                // Handle remaining rows of A (less than 4)
                for (; i < i_end; ++i) {
                    for (size_t k = k0; k < k_end; ++k) {
                        float32x4_t a_ik = vdupq_n_f32(a_data[i * K + k]);
                        size_t j = j0;
                        for (; j + 4 <= j_end; j += 4) {
                            float32x4_t b_kj = vld1q_f32(&b_data[k * N + j]);
                            float32x4_t c_ij = vld1q_f32(&c_data[i * N + j]);
                            c_ij = vfmaq_f32(c_ij, a_ik, b_kj);
                            vst1q_f32(&c_data[i * N + j], c_ij);
                        }
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

inline Tensor matvec_neon(const Tensor& mat, const Tensor& vec) {
    if (mat.ndim() != 2) throw std::invalid_argument("matvec_neon: matrix must be 2D");
    size_t M = mat.shape()[0];
    size_t N = mat.shape()[1];
    size_t vec_size = (vec.ndim() == 1) ? vec.shape()[0] : vec.shape()[0] * vec.shape()[1];
    if (vec_size != N) throw std::invalid_argument("matvec_neon: dimension mismatch");

    Tensor result({M});
    const float* mat_data = mat.data();
    const float* vec_data = vec.data();
    float* result_data = result.data();

    constexpr size_t SIMD_WIDTH = 4;

    for (size_t i = 0; i < M; ++i) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + SIMD_WIDTH <= N; j += SIMD_WIDTH) {
            float32x4_t mat_vals = vld1q_f32(&mat_data[i * N + j]);
            float32x4_t vec_vals = vld1q_f32(&vec_data[j]);
            sum_vec = vfmaq_f32(sum_vec, mat_vals, vec_vals);
        }
        float sum = vaddvq_f32(sum_vec);
        for (; j < N; ++j) sum += mat_data[i * N + j] * vec_data[j];
        result_data[i] = sum;
    }
    return result;
}

inline Tensor add_neon(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::invalid_argument("Tensors must have same shape");
    Tensor result(a.shape(), a.dtype());
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();
    constexpr size_t SIMD_WIDTH = 4;
    size_t size = a.size();
    size_t i = 0;
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        float32x4_t a_vec = vld1q_f32(&a_data[i]);
        float32x4_t b_vec = vld1q_f32(&b_data[i]);
        vst1q_f32(&result_data[i], vaddq_f32(a_vec, b_vec));
    }
    for (; i < size; ++i) result_data[i] = a_data[i] + b_data[i];
    return result;
}

inline Tensor multiply_neon(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::invalid_argument("Tensors must have same shape");
    Tensor result(a.shape(), a.dtype());
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();
    constexpr size_t SIMD_WIDTH = 4;
    size_t size = a.size();
    size_t i = 0;
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        float32x4_t a_vec = vld1q_f32(&a_data[i]);
        float32x4_t b_vec = vld1q_f32(&b_data[i]);
        vst1q_f32(&result_data[i], vmulq_f32(a_vec, b_vec));
    }
    for (; i < size; ++i) result_data[i] = a_data[i] * b_data[i];
    return result;
}

inline Tensor softmax_neon(const Tensor& input, int dim) {
    (void)dim; // Unused parameter
    if (input.empty()) throw std::invalid_argument("empty tensor");
    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];
    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    float* dst = result.data();
    
    for (size_t i = 0; i < rows; ++i) {
        const float* row = &src[i * cols];
        float* out_row = &dst[i * cols];
        
        // Max
        float max_val = -std::numeric_limits<float>::infinity();
        size_t j = 0;
        if (cols >= 4) {
            float32x4_t max_vec = vld1q_f32(row);
            j = 4;
            for (; j + 4 <= cols; j += 4) max_vec = vmaxq_f32(max_vec, vld1q_f32(row + j));
            max_val = vmaxvq_f32(max_vec);
        }
        for (; j < cols; ++j) max_val = std::max(max_val, row[j]);

        // Exp and Sum
        float sum_exp = 0.0f;
        for (j = 0; j < cols; ++j) {
            out_row[j] = std::exp(row[j] - max_val);
            sum_exp += out_row[j];
        }

        // Normalize
        float scale = 1.0f / sum_exp;
        float32x4_t scale_vec = vdupq_n_f32(scale);
        j = 0;
        for (; j + 4 <= cols; j += 4) {
            vst1q_f32(out_row + j, vmulq_f32(vld1q_f32(out_row + j), scale_vec));
        }
        for (; j < cols; ++j) out_row[j] *= scale;
    }
    return result;
}

inline Tensor rms_norm_neon(const Tensor& input, const Tensor& weight, float eps = 1e-6f) {
    if (input.ndim() != 2) throw std::invalid_argument("rms_norm_neon: currently only supports 2D tensors");
    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];
    
    if (weight.shape() != std::vector<size_t>{features}) {
        throw std::invalid_argument("rms_norm_neon: weight shape must match feature dimension");
    }

    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    const float* w = weight.data();
    float* dst = result.data();

    for (size_t i = 0; i < batch_size; ++i) {
        const float* row = src + i * features;
        float* out_row = dst + i * features;

        // 1. Sum squares
        float32x4_t sum_sq_vec = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 4 <= features; j += 4) {
            float32x4_t val = vld1q_f32(row + j);
            sum_sq_vec = vfmaq_f32(sum_sq_vec, val, val);
        }
        float sum_squares = vaddvq_f32(sum_sq_vec);
        for (; j < features; ++j) {
            sum_squares += row[j] * row[j];
        }

        float rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(features) + eps);
        float32x4_t rms_vec = vdupq_n_f32(rms);

        // 2. Normalize and scale
        j = 0;
        for (; j + 4 <= features; j += 4) {
            float32x4_t val = vld1q_f32(row + j);
            float32x4_t w_val = vld1q_f32(w + j);
            val = vmulq_f32(val, rms_vec);
            val = vmulq_f32(val, w_val);
            vst1q_f32(out_row + j, val);
        }
        for (; j < features; ++j) {
            out_row[j] = row[j] * rms * w[j];
        }
    }
    return result;
}

// Helper dispatchers for ARM
inline Tensor matmul_avx2(const Tensor& a, const Tensor& b) { return matmul_neon(a, b); }
inline Tensor matvec_avx2(const Tensor& mat, const Tensor& vec) { return matvec_neon(mat, vec); }
inline Tensor add_avx2(const Tensor& a, const Tensor& b) { return add_neon(a, b); }
inline Tensor multiply_avx2(const Tensor& a, const Tensor& b) { return multiply_neon(a, b); }
inline Tensor softmax_avx2(const Tensor& input, int dim) { return softmax_neon(input, dim); }

#else
// Fallback: if not on x86_64 or ARM64, just use naive implementations
// (ARM NEON implementations could be added here in the future)

inline Tensor matmul_avx2(const Tensor& a, const Tensor& b) {
    // This will be defined in tensor_ops.hpp
    return matmul_naive(a, b);
}

inline Tensor matvec_avx2(const Tensor& mat, const Tensor& vec) {
    return matvec_naive(mat, vec);
}

inline Tensor add_avx2(const Tensor& a, const Tensor& b) {
    return add_naive(a, b);
}

inline Tensor multiply_avx2(const Tensor& a, const Tensor& b) {
    return multiply_naive(a, b);
}

inline Tensor softmax_avx2(const Tensor& input, int dim) {
    if (input.empty()) throw std::invalid_argument("empty tensor");
    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];
    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    float* dst = result.data();
    
    for (size_t i = 0; i < rows; ++i) {
        const float* row = &src[i * cols];
        float* out_row = &dst[i * cols];
        float max_val = row[0];
        for (size_t j = 1; j < cols; ++j) max_val = std::max(max_val, row[j]);
        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            out_row[j] = std::exp(row[j] - max_val);
            sum_exp += out_row[j];
        }
        for (size_t j = 0; j < cols; ++j) out_row[j] /= sum_exp;
    }
    return result;
}

#endif // x86_64

} // namespace simd
} // namespace ops
} // namespace freellm
