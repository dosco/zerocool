#pragma once

#include "core/tensor.hpp"
#include <cmath>
#include <algorithm>
#include <algorithm>

// Platform-specific SIMD intrinsics
// On x86/x64 (Intel/AMD), use immintrin.h
// On ARM (Apple Silicon, etc.), use arm_neon.h
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(__clang__) || defined(__GNUC__)
        #include <x86intrin.h>  // GCC/Clang on x86
    #elif defined(_MSC_VER)
        #include <immintrin.h>  // MSVC on x86
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>       // ARM NEON intrinsics
#endif

#include "kernels/tensor_ops_simd.hpp"

namespace freellm {
namespace ops {

/**
 * @brief Activation function type for future extensibility
 */
enum class Activation {
    ReLU,
    GELU,
    SiLU,
    Tanh
};

// ============================================================================
// Element-wise Operations (Naive implementations)
// ============================================================================

/**
 * @brief Element-wise addition: result = a + b
 * Broadcasting not yet supported - tensors must have same shape
 */
inline Tensor add_naive(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have same shape for addition");
    }

    Tensor result(a.shape(), a.dtype());
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();

    // Naive loop - clear and simple
    for (size_t i = 0; i < a.size(); ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }

    return result;
}

/**
 * @brief Element-wise addition with scalar: result = tensor + scalar
 */
inline Tensor add_scalar(const Tensor& tensor, float scalar) {
    Tensor result(tensor.shape(), tensor.dtype());
    const float* src = tensor.data();
    float* dst = result.data();

    for (size_t i = 0; i < tensor.size(); ++i) {
        dst[i] = src[i] + scalar;
    }

    return result;
}

/**
 * @brief Element-wise multiplication: result = a * b (Hadamard product)
 */
inline Tensor multiply_naive(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have same shape for multiplication");
    }

    Tensor result(a.shape(), a.dtype());
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();

    for (size_t i = 0; i < a.size(); ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }

    return result;
}

/**
 * @brief Element-wise multiplication with scalar: result = tensor * scalar
 */
inline Tensor multiply_scalar(const Tensor& tensor, float scalar) {
    Tensor result(tensor.shape(), tensor.dtype());
    const float* src = tensor.data();
    float* dst = result.data();

    for (size_t i = 0; i < tensor.size(); ++i) {
        dst[i] = src[i] * scalar;
    }

    return result;
}

// ============================================================================
// Activation Functions (Naive implementations)
// ============================================================================

/**
 * @brief ReLU activation: max(0, x)
 * Used in older architectures and some feedforward networks
 */
inline Tensor relu(const Tensor& input) {
    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    float* dst = result.data();

    for (size_t i = 0; i < input.size(); ++i) {
        dst[i] = std::max(0.0f, src[i]);
    }

    return result;
}

/**
 * @brief GELU activation (Gaussian Error Linear Unit)
 * Used in GPT-2, BERT, and many modern transformers
 *
 * Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * This is faster than the exact form with erf()
 */
inline Tensor gelu(const Tensor& input) {
    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    float* dst = result.data();

    constexpr float sqrt_2_over_pi = 0.7978845608f; // √(2/π)
    constexpr float coeff = 0.044715f;

    for (size_t i = 0; i < input.size(); ++i) {
        float x = src[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        dst[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }

    return result;
}

/**
 * @brief SiLU/Swish activation: x * sigmoid(x)
 * Used in LLaMA and many modern architectures
 *
 * SiLU(x) = x * σ(x) = x / (1 + e^(-x))
 */
inline Tensor silu(const Tensor& input) {
    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    float* dst = result.data();

    for (size_t i = 0; i < input.size(); ++i) {
        float x = src[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        dst[i] = x * sigmoid;
    }

    return result;
}

// ============================================================================
// Matrix Operations (Naive implementations)
// ============================================================================

/**
 * @brief Naive matrix multiplication: C = A × B
 *
 * Requirements:
 * - A must be 2D with shape [M, K]
 * - B must be 2D with shape [K, N]
 * - Result will have shape [M, N]
 *
 * Algorithm: Simple triple-nested loop (i-j-k order)
 * Time complexity: O(M × N × K)
 *
 * This is intentionally naive for educational purposes.
 * Optimized versions will be added separately.
 *
 * @param a Left matrix [M, K]
 * @param b Right matrix [K, N]
 * @return Result matrix [M, N]
 */
inline Tensor matmul_naive(const Tensor& a, const Tensor& b) {
    // Validate inputs
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }

    size_t M = a.shape()[0];  // Rows of A
    size_t K = a.shape()[1];  // Cols of A / Rows of B
    size_t N = b.shape()[1];  // Cols of B

    if (b.shape()[0] != K) {
        throw std::invalid_argument(
            "matmul: inner dimensions must match. "
            "A: [" + std::to_string(M) + ", " + std::to_string(K) + "], "
            "B: [" + std::to_string(b.shape()[0]) + ", " + std::to_string(N) + "]");
    }

    // Create result tensor [M, N]
    Tensor result({M, N});
    result.zero();

    const float* a_data = a.data();
    const float* b_data = b.data();
    float* c_data = result.data();

    // Triple-nested loop: i-j-k order
    // For each row of A
    for (size_t i = 0; i < M; ++i) {
        // For each column of B
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            // Dot product of A's row i with B's column j
            for (size_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            c_data[i * N + j] = sum;
        }
    }

    return result;
}

/**
 * @brief Matrix-vector multiplication: y = A × x
 *
 * @param mat Matrix with shape [M, N]
 * @param vec Vector with shape [N] or [N, 1]
 * @return Result vector with shape [M]
 */
inline Tensor matvec_naive(const Tensor& mat, const Tensor& vec) {
    if (mat.ndim() != 2) {
        throw std::invalid_argument("matvec: matrix must be 2D");
    }

    size_t M = mat.shape()[0];
    size_t N = mat.shape()[1];

    // Handle both [N] and [N, 1] vectors
    size_t vec_size = (vec.ndim() == 1) ? vec.shape()[0] : vec.shape()[0] * vec.shape()[1];

    if (vec_size != N) {
        throw std::invalid_argument("matvec: dimension mismatch");
    }

    Tensor result({M});
    const float* mat_data = mat.data();
    const float* vec_data = vec.data();
    float* result_data = result.data();

    // For each row of matrix
    for (size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            sum += mat_data[i * N + j] * vec_data[j];
        }
        result_data[i] = sum;
    }

    return result;
}

/**
 * @brief Transpose a 2D matrix
 *
 * @param input Matrix with shape [M, N]
 * @return Transposed matrix with shape [N, M]
 */
inline Tensor transpose(const Tensor& input) {
    if (input.ndim() != 2) {
        throw std::invalid_argument("transpose: input must be 2D");
    }

    size_t M = input.shape()[0];
    size_t N = input.shape()[1];

    Tensor result({N, M}, input.dtype());
    const float* src = input.data();
    float* dst = result.data();

    // Transpose: result[j, i] = input[i, j]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            dst[j * M + i] = src[i * N + j];
        }
    }

    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Sum all elements in tensor
 */
inline float sum(const Tensor& tensor) {
    const float* data = tensor.data();
    float result = 0.0f;

    for (size_t i = 0; i < tensor.size(); ++i) {
        result += data[i];
    }

    return result;
}

/**
 * @brief Find maximum value in tensor
 */
inline float max(const Tensor& tensor) {
    if (tensor.empty()) {
        throw std::invalid_argument("Cannot find max of empty tensor");
    }

    const float* data = tensor.data();
    float result = data[0];

    for (size_t i = 1; i < tensor.size(); ++i) {
        result = std::max(result, data[i]);
    }

    return result;
}

/**
 * @brief Compute mean of all elements
 */
inline float mean(const Tensor& tensor) {
    if (tensor.empty()) {
        throw std::invalid_argument("Cannot compute mean of empty tensor");
    }

    return sum(tensor) / static_cast<float>(tensor.size());
}

// ============================================================================
// Softmax and Normalization
// ============================================================================

/**
 * @brief Numerically stable softmax along last dimension
 *
 * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Subtracting max prevents overflow for large values
 *
 * @param input Input tensor (any shape)
 * @param dim Dimension to apply softmax (-1 for last dimension)
 * @return Output tensor with same shape as input
 */
inline Tensor softmax_naive(const Tensor& input, int dim = -1) {
    if (input.empty()) {
        throw std::invalid_argument("softmax: input tensor is empty");
    }

    // For now, only support 2D tensors and last dimension
    if (input.ndim() != 2 || (dim != -1 && dim != 1)) {
        throw std::invalid_argument("softmax: currently only supports 2D tensors along last dimension");
    }

    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];

    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    float* dst = result.data();

    // Apply softmax to each row
    for (size_t i = 0; i < rows; ++i) {
        const float* row = detail::row_ptr(src, i, cols);
        float* out_row = detail::row_ptr(dst, i, cols);

        // Find max for numerical stability
        float max_val = row[0];
        for (size_t j = 1; j < cols; ++j) {
            max_val = std::max(max_val, row[j]);
        }

        // Compute exp(x - max) and sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            out_row[j] = std::exp(row[j] - max_val);
            sum_exp += out_row[j];
        }

        // Normalize
        for (size_t j = 0; j < cols; ++j) {
            out_row[j] /= sum_exp;
        }
    }

    return result;
}



/**
 * @brief Element-wise addition with SIMD dispatch
 */
inline Tensor add(const Tensor& a, const Tensor& b) {
    #if defined(__x86_64__) || defined(_M_X64)
        #include "kernels/cpu_features.hpp"
        const auto& features = cpu::get_cpu_features();
        if (features.avx2) {
            return simd::add_avx2(a, b);
        }
    #elif defined(__aarch64__) || defined(_M_ARM64)
        return simd::add_neon(a, b);
    #endif
    return add_naive(a, b);
}

/**
 * @brief Element-wise multiplication with SIMD dispatch
 */
inline Tensor multiply(const Tensor& a, const Tensor& b) {
    #if defined(__x86_64__) || defined(_M_X64)
        #include "kernels/cpu_features.hpp"
        const auto& features = cpu::get_cpu_features();
        if (features.avx2) {
            return simd::multiply_avx2(a, b);
        }
    #elif defined(__aarch64__) || defined(_M_ARM64)
        return simd::multiply_neon(a, b);
    #endif
    return multiply_naive(a, b);
}

/**
 * @brief Numerically stable softmax along last dimension
 *
 * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Subtracting max prevents overflow for large values
 *
 * @param input Input tensor (any shape)
 * @param dim Dimension to apply softmax (-1 for last dimension)
 * @return Output tensor with same shape as input
 */
inline Tensor softmax(const Tensor& input, int dim = -1) {
    #if defined(__x86_64__) || defined(_M_X64)
        #include "kernels/cpu_features.hpp"
        const auto& features = cpu::get_cpu_features();
        if (features.avx2) {
            return simd::softmax_avx2(input, dim);
        }
    #elif defined(__aarch64__) || defined(_M_ARM64)
        // ARM NEON is mandatory on ARM64 (Apple Silicon, etc.)
        return simd::softmax_neon(input, dim);
    #endif

    return softmax_naive(input, dim);
}

/**
 * @brief Layer Normalization (used in GPT-2, BERT, etc.)
 *
 * LayerNorm(x) = weight * (x - mean) / sqrt(variance + eps) + bias
 *
 * Normalizes across the last dimension (features)
 *
 * @param input Input tensor with shape [..., D]
 * @param weight Scale parameter with shape [D]
 * @param bias Shift parameter with shape [D]
 * @param eps Small constant for numerical stability
 * @return Normalized tensor with same shape as input
 */
inline Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps = 1e-5f) {
    // For now, support 2D: [batch_size, features]
    if (input.ndim() != 2) {
        throw std::invalid_argument("layer_norm: currently only supports 2D tensors");
    }

    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];

    if (weight.shape() != std::vector<size_t>{features}) {
        throw std::invalid_argument("layer_norm: weight shape must match feature dimension");
    }
    if (bias.shape() != std::vector<size_t>{features}) {
        throw std::invalid_argument("layer_norm: bias shape must match feature dimension");
    }

    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    const float* w = weight.data();
    const float* b = bias.data();
    float* dst = result.data();

    // Normalize each sample in the batch
    for (size_t i = 0; i < batch_size; ++i) {
        const float* row = detail::row_ptr(src, i, features);
        float* out_row = detail::row_ptr(dst, i, features);

        // Compute mean
        float mean_val = 0.0f;
        for (size_t j = 0; j < features; ++j) {
            mean_val += row[j];
        }
        mean_val /= static_cast<float>(features);

        // Compute variance
        float var_val = 0.0f;
        for (size_t j = 0; j < features; ++j) {
            float diff = row[j] - mean_val;
            var_val += diff * diff;
        }
        var_val /= static_cast<float>(features);

        // Normalize and apply affine transformation
        float inv_std = 1.0f / std::sqrt(var_val + eps);
        for (size_t j = 0; j < features; ++j) {
            float normalized = (row[j] - mean_val) * inv_std;
            out_row[j] = w[j] * normalized + b[j];
        }
    }

    return result;
}

/**
 * @brief RMS (Root Mean Square) Normalization (used in LLaMA, Mistral)
 *
 * RMSNorm(x) = weight * x / sqrt(mean(x²) + eps)
 *
 * Simpler and faster than LayerNorm (no mean subtraction, no bias)
 *
 * @param input Input tensor with shape [..., D]
 * @param weight Scale parameter with shape [D]
 * @param eps Small constant for numerical stability
 * @return Normalized tensor with same shape as input
 */
inline Tensor rms_norm_naive(const Tensor& input, const Tensor& weight, float eps = 1e-6f) {
    // For now, support 2D: [batch_size, features]
    if (input.ndim() != 2) {
        throw std::invalid_argument("rms_norm: currently only supports 2D tensors");
    }

    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];

    if (weight.shape() != std::vector<size_t>{features}) {
        throw std::invalid_argument("rms_norm: weight shape must match feature dimension");
    }

    Tensor result(input.shape(), input.dtype());
    const float* src = input.data();
    const float* w = weight.data();
    float* dst = result.data();

    // Normalize each sample in the batch
    for (size_t i = 0; i < batch_size; ++i) {
        const float* row = detail::row_ptr(src, i, features);
        float* out_row = detail::row_ptr(dst, i, features);

        // Compute RMS: sqrt(mean(x²))
        float sum_squares = 0.0f;
        for (size_t j = 0; j < features; ++j) {
            sum_squares += row[j] * row[j];
        }
        float rms = std::sqrt(sum_squares / static_cast<float>(features) + eps);

        // Normalize and scale
        for (size_t j = 0; j < features; ++j) {
            out_row[j] = w[j] * (row[j] / rms);
        }
    }

    return result;
}

/**
 * @brief RMS (Root Mean Square) Normalization with SIMD dispatch
 */
inline Tensor rms_norm(const Tensor& input, const Tensor& weight, float eps = 1e-6f) {
    #if defined(__x86_64__) || defined(_M_X64)
        // AVX2 implementation not yet added, fall back to naive
        // if (features.avx2) return simd::rms_norm_avx2(input, weight, eps);
    #elif defined(__aarch64__) || defined(_M_ARM64)
        return simd::rms_norm_neon(input, weight, eps);
    #endif
    return rms_norm_naive(input, weight, eps);
}

// ============================================================================
// Smart Dispatch Wrappers (Forward declarations for SIMD ops)
// ============================================================================

// Forward declarations from tensor_ops_simd.hpp are now included directly

/**
 * @brief Smart matrix multiplication dispatcher
 *
 * Automatically selects the best implementation based on CPU features:
 * - AVX2 available: Uses matmul_avx2() for ~10-20x speedup
 * - Fallback: Uses matmul_naive()
 *
 * Users can still call matmul_naive() or simd::matmul_avx2() explicitly
 * for benchmarking or testing purposes.
 */
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    #if defined(__x86_64__) || defined(_M_X64)
        // Include cpu_features.hpp for detection
        #include "kernels/cpu_features.hpp"
        const auto& features = cpu::get_cpu_features();
        if (features.avx2) {
            return simd::matmul_avx2(a, b);
        }
    #elif defined(__aarch64__) || defined(_M_ARM64)
        return simd::matmul_neon(a, b);
    #endif
    return matmul_naive(a, b);
}

/**
 * @brief Smart matrix-vector multiplication dispatcher
 */
inline Tensor matvec(const Tensor& mat, const Tensor& vec) {
    #if defined(__x86_64__) || defined(_M_X64)
        #include "kernels/cpu_features.hpp"
        const auto& features = cpu::get_cpu_features();
        if (features.avx2) {
            return simd::matvec_avx2(mat, vec);
        }
    #elif defined(__aarch64__) || defined(_M_ARM64)
        return simd::matvec_neon(mat, vec);
    #endif
    return matvec_naive(mat, vec);
}

} // namespace ops
} // namespace freellm
