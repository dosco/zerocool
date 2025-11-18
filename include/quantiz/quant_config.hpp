#pragma once

#include "types.hpp"

namespace freellm::quant {

/**
 * @brief High-level quantization policy used during weight loading.
 *
 * Each linear block can be configured independently. We quantize the
 * transposed version of every linear weight so that rows correspond to
 * output channels (facilitating dot products with activations).
 */
struct QuantConfig {
    QuantType default_type = QuantType::NONE;
    QuantType embedding = QuantType::NONE;
    QuantType attention = QuantType::NONE;
    QuantType feed_forward = QuantType::NONE;
    QuantType lm_head = QuantType::NONE;

    // Keep FP32 copies so we can fall back for debugging or mixed-precision ops
    bool keep_fp32_weights = true;

    constexpr bool enabled() const {
        return default_type != QuantType::NONE ||
               embedding != QuantType::NONE ||
               attention != QuantType::NONE ||
               feed_forward != QuantType::NONE ||
               lm_head != QuantType::NONE;
    }

    constexpr QuantType resolve_embedding() const {
        return embedding != QuantType::NONE ? embedding : default_type;
    }

    constexpr QuantType resolve_attention() const {
        return attention != QuantType::NONE ? attention : default_type;
    }

    constexpr QuantType resolve_feed_forward() const {
        return feed_forward != QuantType::NONE ? feed_forward : default_type;
    }

    constexpr QuantType resolve_lm_head() const {
        return lm_head != QuantType::NONE ? lm_head : default_type;
    }

    static constexpr QuantConfig Disabled() {
        return {};
    }
};

} // namespace freellm::quant

