#pragma once

#include "types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

/**
 * @file quantiz/q4_K/quantize.hpp
 * @brief Q4_K quantization functions (FP32 -> Q4_K)
 *
 * These routines intentionally follow the logic from llama.cpp's ggml-quants.c
 * so that we can validate our results against the canonical implementation.
 */

namespace freellm::quant::q4_K {

namespace detail {

// Integer rounding helper that matches llama.cpp's nearest_int utility.
inline int nearest_int(float value) {
    return static_cast<int>(std::lrintf(value));
}

// Port of llama.cpp's make_qkx2_quants (see ggml-quants.c, lines ~622-701).
inline float make_qkx2_quants(
    int n,
    int nmax,
    const float* x,
    const float* weights,
    uint8_t* L,
    float* the_min,
    uint8_t* Laux,
    float rmin,
    float rdelta,
    int nstep,
    bool use_mad) {

    float min_val = x[0];
    float max_val = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];

    for (int i = 1; i < n; ++i) {
        const float xi = x[i];
        min_val = std::min(min_val, xi);
        max_val = std::max(max_val, xi);
        const float w = weights[i];
        sum_w += w;
        sum_x += w * xi;
    }

    if (min_val > 0.0f) {
        min_val = 0.0f;
    }

    if (max_val == min_val) {
        std::fill_n(L, n, uint8_t{0});
        *the_min = -min_val;
        return 0.0f;
    }

    float iscale = static_cast<float>(nmax) / (max_val - min_val);
    float scale = 1.0f / iscale;
    float best_error = 0.0f;

    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * (x[i] - min_val));
        l = std::clamp(l, 0, nmax);
        L[i] = static_cast<uint8_t>(l);
        float diff = scale * l + min_val - x[i];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        best_error += weights[i] * diff;
    }

    if (nstep < 1) {
        *the_min = -min_val;
        return scale;
    }

    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * static_cast<float>(is) + static_cast<float>(nmax)) /
                 (max_val - min_val);

        float sum_l = 0.0f;
        float sum_l2 = 0.0f;
        float sum_xl = 0.0f;

        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * (x[i] - min_val));
            l = std::clamp(l, 0, nmax);
            Laux[i] = static_cast<uint8_t>(l);
            const float w = weights[i];
            sum_l  += w * static_cast<float>(l);
            sum_l2 += w * static_cast<float>(l * l);
            sum_xl += w * static_cast<float>(l) * x[i];
        }

        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D <= 0.0f) {
            continue;
        }

        float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
        float this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / D;
        if (this_min > 0.0f) {
            this_min = 0.0f;
            this_scale = sum_xl / sum_l2;
        }

        float cur_error = 0.0f;
        for (int i = 0; i < n; ++i) {
            float diff = this_scale * static_cast<float>(Laux[i]) + this_min - x[i];
            diff = use_mad ? std::fabs(diff) : diff * diff;
            cur_error += weights[i] * diff;
        }

        if (cur_error < best_error) {
            std::memcpy(L, Laux, static_cast<size_t>(n) * sizeof(uint8_t));
            best_error = cur_error;
            scale = this_scale;
            min_val = this_min;
        }
    }

    *the_min = -min_val;
    return scale;
}

template <typename T>
inline T clamp(T value, T low, T high) {
    return std::min(std::max(value, low), high);
}

} // namespace detail

// -----------------------------------------------------------------------------
// Q4_K quantization (scalar reference path)
// -----------------------------------------------------------------------------

inline void quantize_row(const float* src, void* dst_void, size_t k) {
    if (k == 0) {
        return;
    }

    auto* dst = static_cast<block_q4_K*>(dst_void);
    const size_t nb = num_blocks(k);

    std::array<float, QK_K> block_data{};
    std::array<uint8_t, QK_K> codes{};
    std::array<uint8_t, 32> aux{};
    std::array<float, 32> weights{};
    std::array<float, QK_K / 32> mins{};
    std::array<float, QK_K / 32> scales{};

    for (size_t ib = 0; ib < nb; ++ib) {
        const size_t block_offset = ib * QK_K;

        std::fill(block_data.begin(), block_data.end(), 0.0f);
        if (block_offset < k) {
            const size_t elems = std::min(static_cast<size_t>(QK_K), k - block_offset);
            std::memcpy(
                block_data.data(),
                src + block_offset,
                elems * sizeof(float));
        }

        float max_scale = 0.0f;
        float max_min = 0.0f;

        for (size_t sub = 0; sub < QK_K / 32; ++sub) {
            const size_t chunk_offset = static_cast<size_t>(sub) * 32;
            const size_t global_offset = block_offset + chunk_offset;

            if (global_offset >= k) {
                scales[sub] = 0.0f;
                mins[sub] = 0.0f;
                std::fill_n(codes.data() + chunk_offset, 32, uint8_t{0});
                continue;
            }

            const size_t elems = std::min<size_t>(32, k - global_offset);
            float sum_x2 = 0.0f;
            for (size_t i = 0; i < elems; ++i) {
                const float v = block_data[chunk_offset + i];
                sum_x2 += v * v;
            }
            const float av_x = std::sqrt(sum_x2 / std::max<size_t>(1, elems));

            for (size_t i = 0; i < 32; ++i) {
                const float v = (i < elems) ? block_data[chunk_offset + i] : 0.0f;
                weights[i] = av_x + std::fabs(v);
            }

            scales[sub] = detail::make_qkx2_quants(
                32,
                15,
                block_data.data() + chunk_offset,
                weights.data(),
                codes.data() + chunk_offset,
                &mins[sub],
                aux.data(),
                -1.0f,
                0.1f,
                20,
                false);

            max_scale = std::max(max_scale, scales[sub]);
            max_min = std::max(max_min, mins[sub]);
        }

        auto& block = dst[ib];
        block.set_scale(max_scale > 0.0f ? max_scale / 63.0f : 0.0f);
        block.set_min_scale(max_min > 0.0f ? max_min / 63.0f : 0.0f);

        for (size_t sub = 0; sub < QK_K / 32; ++sub) {
            const uint8_t scale_q = (max_scale > 0.0f)
                ? static_cast<uint8_t>(detail::clamp(detail::nearest_int(63.0f * scales[sub] / max_scale), 0, 63))
                : 0;
            const uint8_t min_q = (max_min > 0.0f)
                ? static_cast<uint8_t>(detail::clamp(detail::nearest_int(63.0f * mins[sub] / max_min), 0, 63))
                : 0;
            set_scale_min(sub, block.scales, scale_q, min_q);
        }

        std::fill(block.qs, block.qs + (QK_K / 2), uint8_t{0});
        const float d_base = block.scale();
        const float m_base = block.min_scale();

        // Pack nibbles in llama.cpp's interleaved format:
        // For each 64-element chunk, pack values [j, j+31] in low nibbles
        // and values [j+32, j+63] in high nibbles of bytes [j/2, j/2+31]
        for (size_t chunk_base = 0; chunk_base < QK_K; chunk_base += 64) {
            // Get the two sub-block scales for this 64-element chunk
            const size_t sub0 = chunk_base / 32;      // First sub-block (32 values)
            const size_t sub1 = sub0 + 1;             // Second sub-block (32 values)
            
            uint8_t sc0 = 0, m0 = 0;
            uint8_t sc1 = 0, m1 = 0;
            get_scale_min(sub0, block.scales, &sc0, &m0);
            get_scale_min(sub1, block.scales, &sc1, &m1);

            const float d0 = d_base * static_cast<float>(sc0);
            const float d1 = d_base * static_cast<float>(sc1);
            const float dm0 = m_base * static_cast<float>(m0);
            const float dm1 = m_base * static_cast<float>(m1);

            // Pack 32 bytes: low nibbles from first half, high nibbles from second half
            uint8_t* q_out = block.qs + (chunk_base / 2);
            for (size_t i = 0; i < 32; ++i) {
                const size_t idx_low = chunk_base + i;
                const size_t idx_high = chunk_base + 32 + i;
                
                // Quantize lower nibble (using sub0 scale)
                float v_low = (idx_low < k) ? block_data[idx_low] : 0.0f;
                int q_low = (d0 > 0.0f) ? detail::nearest_int((v_low + dm0) / d0) : 0;
                q_low = detail::clamp(q_low, 0, 15);
                
                // Quantize upper nibble (using sub1 scale)
                float v_high = (idx_high < k) ? block_data[idx_high] : 0.0f;
                int q_high = (d1 > 0.0f) ? detail::nearest_int((v_high + dm1) / d1) : 0;
                q_high = detail::clamp(q_high, 0, 15);
                
                // Pack both nibbles into one byte
                q_out[i] = static_cast<uint8_t>(q_low) | (static_cast<uint8_t>(q_high) << 4);
            }
        }
    }
}

inline void quantize_matrix(const float* src, size_t rows, size_t cols, void* dst) {
    auto* row_dst = static_cast<uint8_t*>(dst);
    const size_t blocks_per_row = num_blocks(cols);
    const size_t row_bytes = blocks_per_row * sizeof(block_q4_K);

    for (size_t r = 0; r < rows; ++r) {
        quantize_row(
            src + r * cols,
            row_dst + r * row_bytes,
            cols);
    }
}

} // namespace freellm::quant::q4_K

