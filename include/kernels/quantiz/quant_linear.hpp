#pragma once

#include "core/tensor.hpp"
#include "kernels/tensor_ops.hpp"
#include "quantized_tensor.hpp"
#include "ops.hpp"

#include <optional>
#include <stdexcept>
#include <format>

namespace freellm::quant {

/**
 * @brief Apply a linear transformation with optional quantized weights.
 *
 * @param input   Tensor with shape [batch, in_features]
 * @param weight  FP32 weight matrix [in_features, out_features]
 * @param quantized Optional quantized tensor storing transposed weights
 *                  with shape [out_features, in_features].
 */
inline Tensor linear_forward(const Tensor& input,
                             const Tensor& weight,
                             const std::optional<QuantizedTensor>& quantized) {
    if (!quantized) {
        return ops::matmul(input, weight);
    }

    if (input.ndim() != 2) {
        throw std::invalid_argument("linear_forward: input must be 2D [batch, in_features]");
    }

    const auto& w_shape = quantized->shape();
    if (w_shape.size() != 2) {
        throw std::invalid_argument("linear_forward: quantized weights must be 2D");
    }

    const size_t batch = input.shape()[0];
    const size_t in_features = input.shape()[1];
    const size_t out_features = w_shape[0];

    if (w_shape[1] != in_features) {
        throw std::invalid_argument(std::format(
            "linear_forward: quantized weight inner dimension mismatch (expected {}, got {})",
            in_features,
            w_shape[1]));
    }

    Tensor output({batch, out_features});

    const float* input_data = input.data();
    float* output_data = output.data();

    // Hoist the switch outside the loop - qtype doesn't change between iterations
    const QuantType qtype = quantized->qtype();
    const void* quantized_data = quantized->data();

    switch (qtype) {
        case QuantType::Q8_0:
            for (size_t row = 0; row < batch; ++row) {
                const float* vec = input_data + row * in_features;
                float* dst = output_data + row * out_features;
                matvec_q8_0(quantized_data, out_features, in_features, vec, dst);
            }
            break;

        case QuantType::Q4_K:
            for (size_t row = 0; row < batch; ++row) {
                const float* vec = input_data + row * in_features;
                float* dst = output_data + row * out_features;
                matvec_q4_K(quantized_data, out_features, in_features, vec, dst);
            }
            break;

        default:
            throw std::runtime_error("linear_forward: unsupported quant type");
    }

    return output;
}

} // namespace freellm::quant

