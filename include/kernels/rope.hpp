#pragma once

#include "core/tensor.hpp"
#include <cmath>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
#endif

namespace freellm {

/**
 * @brief Precomputed sin/cos tables for Rotary Position Embeddings (RoPE)
 *
 * RoPE rotates query and key vectors using frequency-based embeddings:
 * - More efficient than absolute position embeddings
 * - Better extrapolation to longer sequences
 * - Used in LLaMA, GPT-NeoX, and many modern models
 *
 * Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
 * https://arxiv.org/abs/2104.09864
 */
class RoPECache {
public:
    /**
     * @brief Construct RoPE cache
     *
     * @param dim Dimension per head (must be even)
     * @param max_seq_len Maximum sequence length to precompute
     * @param theta Base frequency (10000.0 for LLaMA)
     */
    RoPECache(size_t dim, size_t max_seq_len = 2048, float theta = 10000.0f)
        : dim_(dim), max_seq_len_(max_seq_len), theta_(theta) {

        if (dim % 2 != 0) {
            throw std::invalid_argument("RoPE: dimension must be even");
        }

        // Precompute frequencies: theta^(-2i/dim) for i in [0, dim/2)
        freqs_.reserve(dim / 2);
        for (size_t i = 0; i < dim / 2; ++i) {
            float freq = 1.0f / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(dim));
            freqs_.push_back(freq);
        }

        // Precompute sin/cos for all positions
        cos_cache_ = Tensor({max_seq_len, dim / 2});
        sin_cache_ = Tensor({max_seq_len, dim / 2});

        for (size_t pos = 0; pos < max_seq_len; ++pos) {
            for (size_t i = 0; i < dim / 2; ++i) {
                float angle = static_cast<float>(pos) * freqs_[i];
                cos_cache_.at({pos, i}) = std::cos(angle);
                sin_cache_.at({pos, i}) = std::sin(angle);
            }
        }
    }

    /**
     * @brief Apply RoPE to query or key tensor
     *
     * Rotates pairs of elements: (x[2i], x[2i+1]) for each position
     *
     * The rotation is applied as:
     * [cos(θ)  -sin(θ)] [x_even]
     * [sin(θ)   cos(θ)] [x_odd ]
     *
     * where θ depends on the position and frequency
     *
     * @param x Input tensor with shape [seq_len, n_heads, head_dim]
     * @param position_offset Starting position in sequence (for incremental generation)
     * @return Rotated tensor with same shape
     */
    Tensor apply(const Tensor& x, size_t position_offset = 0) const {
        // For now, support simplified 3D shape: [seq_len, n_heads, head_dim]
        if (x.ndim() != 3) {
            throw std::invalid_argument("RoPE: currently only supports 3D tensors [seq_len, n_heads, head_dim]");
        }

        size_t seq_len = x.shape()[0];
        size_t n_heads = x.shape()[1];
        size_t head_dim = x.shape()[2];

        if (head_dim != dim_) {
            throw std::invalid_argument("RoPE: head_dim doesn't match cache dimension");
        }

        if (position_offset + seq_len > max_seq_len_) {
            throw std::out_of_range("RoPE: position exceeds max_seq_len");
        }

        Tensor result = x.copy();
        float* data = result.data();

        // Apply rotation for each position, head, and dimension pair
        for (size_t pos = 0; pos < seq_len; ++pos) {
            size_t cache_pos = position_offset + pos;

            for (size_t h = 0; h < n_heads; ++h) {
                size_t i = 0;

                #if defined(__aarch64__) || defined(_M_ARM64)
                // NEON optimization: process 4 pairs (8 elements) at a time
                for (; i + 4 <= head_dim / 2; i += 4) {
                    size_t idx_base = pos * (n_heads * head_dim) + h * head_dim + 2 * i;
                    
                    // Load 8 floats (4 pairs) deinterleaved:
                    // x_pairs.val[0] = evens, x_pairs.val[1] = odds
                    float32x4x2_t x_pairs = vld2q_f32(&data[idx_base]);
                    float32x4_t x_even = x_pairs.val[0];
                    float32x4_t x_odd = x_pairs.val[1];
                    
                    // Load cos and sin (contiguous in cache)
                    float32x4_t cos_vals = vld1q_f32(&cos_cache_.at({cache_pos, i}));
                    float32x4_t sin_vals = vld1q_f32(&sin_cache_.at({cache_pos, i}));
                    
                    // Compute rotation
                    // out_even = x_even * cos - x_odd * sin
                    float32x4_t out_even = vmlsq_f32(vmulq_f32(x_even, cos_vals), x_odd, sin_vals);
                    // out_odd  = x_even * sin + x_odd * cos
                    float32x4_t out_odd  = vmlaq_f32(vmulq_f32(x_even, sin_vals), x_odd, cos_vals);
                    
                    // Store interleaved
                    float32x4x2_t out_pairs;
                    out_pairs.val[0] = out_even;
                    out_pairs.val[1] = out_odd;
                    vst2q_f32(&data[idx_base], out_pairs);
                }
                #endif

                for (; i < head_dim / 2; ++i) {
                    size_t idx_even = pos * (n_heads * head_dim) + h * head_dim + 2 * i;
                    size_t idx_odd = idx_even + 1;

                    float x_even = data[idx_even];
                    float x_odd = data[idx_odd];

                    float cos_val = cos_cache_.at({cache_pos, i});
                    float sin_val = sin_cache_.at({cache_pos, i});

                    // Rotation matrix:
                    // [cos  -sin] [x_even]
                    // [sin   cos] [x_odd ]
                    data[idx_even] = x_even * cos_val - x_odd * sin_val;
                    data[idx_odd] = x_even * sin_val + x_odd * cos_val;
                }
            }
        }

        return result;
    }

    // Accessors
    size_t dim() const { return dim_; }
    size_t max_seq_len() const { return max_seq_len_; }
    float theta() const { return theta_; }

private:
    size_t dim_;
    size_t max_seq_len_;
    float theta_;
    std::vector<float> freqs_;
    Tensor cos_cache_;
    Tensor sin_cache_;
};

} // namespace freellm
