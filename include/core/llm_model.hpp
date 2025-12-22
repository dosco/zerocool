#pragma once

#include "core/tensor.hpp"
#include "kernels/tensor_ops.hpp"
#include "core/model_config.hpp"
#include "core/transformer_block.hpp"
#include "core/paged_kv_cache.hpp"
#include "kernels/quantiz/quant_config.hpp"
#include "kernels/quantiz/quant_linear.hpp"
#include <vector>
#include <memory>
#include <optional>

namespace freellm {

/**
 * @brief Complete Large Language Model (LLM) implementation
 *
 * This class implements a decoder-only transformer model (like GPT, LLaMA).
 * The architecture follows the modern LLM design:
 *
 * Architecture Pipeline:
 * ┌──────────────────────────────────────────────────────────────┐
 * │ Input: Token IDs [seq_len]                                   │
 * └──────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌──────────────────────────────────────────────────────────────┐
 * │ Token Embedding: [seq_len] → [seq_len, d_model]             │
 * │   Converts discrete token IDs to continuous vectors          │
 * └──────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌──────────────────────────────────────────────────────────────┐
 * │ Transformer Block 0                                           │
 * │   RMSNorm → Attention → Residual → RMSNorm → F→ Residual │
 * └──────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌──────────────────────────────────────────────────────────────┐
 * │ Transformer Block 1                                           │
 * │   ... (repeated n_layers times) ...                          │
 * └──────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌──────────────────────────────────────────────────────────────┐
 * │ Final RMSNorm: [seq_len, d_model]                           │
 * │   Final normalization before output projection               │
 * └──────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌──────────────────────────────────────────────────────────────┐
 * │ LM Head: [seq_len, d_model] → [seq_len, vocab_size]        │
 * │   Projects to vocabulary logits for next token prediction    │
 * └──────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌──────────────────────────────────────────────────────────────┐
 * │ Output: Logits [seq_len, vocab_size]                        │
 * │   For each position, probability distribution over vocabulary│
 * └──────────────────────────────────────────────────────────────┘
 *
 * Key Design Decisions:
 * - **Token embeddings**: Convert discrete tokens to continuous vectors
 * - **No positional embeddings**: Using RoPE (in attention) instead
 * - **Pre-norm architecture**: Normalize before each sub-layer (modern approach)
 * - **RMSNorm**: Simpler than LayerNorm, works as well in practice
 * - **Residual connections**: Essential for training deep networks
 * - **Single sequence**: No batch dimension (batch_size=1 implicit)
 *
 * Parameter Count (TinyLLaMA 1.1B):
 * - Embeddings: vocab_size * d_model = 32000 * 2048 ≈ 65M
 * - Each block: ~50M (attention + FFN)
 * - 22 blocks: ~1.1B parameters
 * - LM head: ~65M (often shares weights with embeddings)
 */
class LLMModel {
public:
    /**
     * @brief Construct LLM model from configuration
     *
     * Initializes all layers and allocates memory for parameters.
     * Weights are initialized to small random values - they should be
     * loaded from a trained checkpoint before inference.
     *
     * @param config Model configuration (hyperparameters)
     */
    explicit LLMModel(const ModelConfig& config,
                      quant::QuantConfig quant_config = quant::QuantConfig::Disabled())
        : config_(config)
        , quant_config_(quant_config)
    {
        // Validate configuration
        config.validate();

        // ================================================================
        // Initialize Token Embedding Layer
        // ================================================================
        // Maps token IDs (0 to vocab_size-1) to dense vectors
        // Each token gets a learnable d_model-dimensional vector
        // Shape: [vocab_size, d_model]
        // Example: token ID 42 → embedding_[42] (a d_model-dim vector)
        token_embedding_ = Tensor({config.vocab_size, config.d_model});

        // ================================================================
        // Initialize Transformer Blocks (the main model)
        // ================================================================
        // Create a stack of identical transformer blocks
        // Each block refines the representations from the previous block
        // Modern LLMs use 20-100+ layers (TinyLLaMA uses 22)
        blocks_.reserve(config.n_layers);

        for (size_t i = 0; i < config.n_layers; ++i) {
            blocks_.push_back(std::make_unique<TransformerBlock>(
                config.d_model,          // Model dimension
                config.d_ff,             // Feed-forward dimension
                config.n_heads,          // Number of query attention heads
                config.n_kv_heads,       // Number of key/value heads (for GQA)
                config.norm_eps,         // RMSNorm epsilon
                true,                    // Use rotary position embeddings (always true for modern LLMs)
                config.rope_theta,       // RoPE base frequency
                config.max_seq_len       // Maximum sequence length
            ));
        }

        // ================================================================
        // Initialize Final Layer Norm
        // ================================================================
        // Final normalization before projecting to vocabulary
        // Stabilizes the final hidden states
        // Shape: [d_model] - one scale parameter per dimension
        final_norm_weight_ = Tensor({config.d_model}, 1.0f);

        // ================================================================
        // Initialize Language Modeling Head
        // ================================================================
        // Projects final hidden states to vocabulary logits
        // Shape: [d_model, vocab_size]
        // For each position, produces a distribution over all possible next tokens
        // Note: In many models, this shares weights with token_embedding_ (weight tying)
        lm_head_ = Tensor({config.d_model, config.vocab_size});
    }

    /**
     * @brief Forward pass through the entire model
     *
     * Processes a sequence of tokens and produces logits for next token prediction.
     *
     * Pipeline:
     *   1. Embed tokens: [seq_len] → [seq_len, d_model]
     *   2. Pass through transformer blocks (refine representations)
     *   3. Apply final normalization
     *   4. Project to vocabulary: [seq_len, d_model] → [seq_len, vocab_size]
     *
     * Output interpretation:
     *   - For position i, logits[i] gives scores for the next token after position i
     *   - During generation, we typically only use logits[-1] (last position)
     *   - Higher logit = higher probability after softmax
     *
     * @param token_ids Vector of token IDs (integers from 0 to vocab_size-1)
     * @param position_offset Position offset for RoPE (used in incremental generation)
     * @return Logits tensor with shape [seq_len, vocab_size]
     */
    Tensor forward(const std::vector<int>& token_ids, size_t position_offset = 0) {
        // Validate inputs
        if (token_ids.empty()) {
            throw std::invalid_argument("LLMModel: token_ids cannot be empty");
        }
        if (token_ids.size() > config_.max_seq_len) {
            throw std::invalid_argument(
                "LLMModel: sequence length " + std::to_string(token_ids.size()) +
                " exceeds max_seq_len " + std::to_string(config_.max_seq_len)
            );
        }

        size_t seq_len = token_ids.size();

        // ================================================================
        // Step 1: Token Embedding
        // ================================================================
        // Look up embedding vectors for each token ID
        // token_ids[i] → embedding matrix row token_ids[i]
        // Result: [seq_len, d_model]
        //
        // Example: If token_ids = [5, 42, 123], we get:
        //   [[embedding_[5]],   ← d_model dimensions
        //    [embedding_[42]],  ← d_model dimensions
        //    [embedding_[123]]] ← d_model dimensions

        Tensor hidden_states({seq_len, config_.d_model});

        for (size_t i = 0; i < seq_len; ++i) {
            int token_id = token_ids[i];

            // Bounds check
            if (token_id < 0 || static_cast<size_t>(token_id) >= config_.vocab_size) {
                throw std::out_of_range(
                    "Token ID " + std::to_string(token_id) +
                    " out of vocabulary range [0, " + std::to_string(config_.vocab_size) + ")"
                );
            }

            // Copy embedding vector for this token
            // embedding_[token_id] is a row vector of size d_model
            for (size_t j = 0; j < config_.d_model; ++j) {
                hidden_states.at({i, j}) = token_embedding_.at({static_cast<size_t>(token_id), j});
            }
        }

        // At this point: hidden_states shape = [seq_len, d_model]
        // Each row is a dense representation of the corresponding token

        // ================================================================
        // Step 2: Pass through Transformer Blocks
        // ================================================================
        // Each block refines the representations
        // Block 0: captures basic patterns (e.g., common word sequences)
        // Block N: captures complex patterns (e.g., long-range dependencies, semantics)
        //
        // Shape remains [seq_len, d_model] throughout all blocks
        //
        // KV CACHE USAGE:
        // When KV cache is initialized, each layer uses its own cache.
        // This dramatically speeds up autoregressive generation by avoiding
        // recomputation of K/V for all previous tokens.

        for (size_t layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
            // Forward through this transformer block
            // The block internally handles:
            //   - Self-attention (tokens interact with each other, with optional KV cache)
            //   - Feed-forward (process each token independently)
            //   - Residual connections (preserve information flow)
            //   - Normalizations (stabilize activations)

            // Pass Paged KV cache pointer if available
            PagedKVCache* cache_ptr = kv_cache_initialized_ ? kv_caches_[layer_idx].get() : nullptr;
            hidden_states = blocks_[layer_idx]->forward(hidden_states, position_offset, cache_ptr);

            // Shape: still [seq_len, d_model]
        }

        // After all blocks: hidden_states contains refined contextual representations
        // Each position's vector encodes information from the entire sequence

        // ================================================================
        // Step 3: Final Layer Normalization
        // ================================================================
        // Normalize hidden states one last time before output projection
        // This ensures stable inputs to the LM head
        // Shape: [seq_len, d_model] → [seq_len, d_model]

        hidden_states = ops::rms_norm(hidden_states, final_norm_weight_, config_.norm_eps);

        // ================================================================
        // Step 4: Language Modeling Head (Project to Vocabulary)
        // ================================================================
        // Project from d_model dimensions to vocab_size dimensions
        // This produces "logits" - unnormalized scores for each token in vocabulary
        //
        // hidden_states: [seq_len, d_model] @ lm_head_: [d_model, vocab_size]
        //   → logits: [seq_len, vocab_size]
        //
        // For each position i:
        //   logits[i] = scores for "what token should come after position i?"
        //   Higher score = more likely next token
        //
        // To get probabilities: apply softmax(logits[i])

        Tensor logits = quant::linear_forward(hidden_states, lm_head_, lm_head_quant_);

        // Final output shape: [seq_len, vocab_size]
        // During generation, we typically use logits[-1] (last position)
        // to predict the next token
        return logits;
    }

    /**
     * @brief Get model configuration
     */
    const ModelConfig& config() const { return config_; }

    /**
     * @brief Access token embedding weights (for loading from file)
     */
    Tensor& token_embedding() { return token_embedding_; }
    const Tensor& token_embedding() const { return token_embedding_; }

    /**
     * @brief Access transformer blocks (for loading weights)
     */
    std::vector<std::unique_ptr<TransformerBlock>>& blocks() { return blocks_; }
    const std::vector<std::unique_ptr<TransformerBlock>>& blocks() const { return blocks_; }

    /**
     * @brief Access final norm weight (for loading from file)
     */
    Tensor& final_norm_weight() { return final_norm_weight_; }
    const Tensor& final_norm_weight() const { return final_norm_weight_; }

    /**
     * @brief Access LM head weights (for loading from file)
     */
    Tensor& lm_head() { return lm_head_; }
    const Tensor& lm_head() const { return lm_head_; }
    const quant::QuantConfig& quant_config() const { return quant_config_; }

    /**
     * @brief Load pretrained weights into the model
     *
     * This method copies weights from a WeightMap (loaded from safetensors)
     * into the model's parameters. It handles the name mapping and nested
     * structure of transformer blocks.
     *
     * Expected weight names in the WeightMap:
     * - "token_embedding": Token embedding matrix [vocab_size, d_model]
     * - "layers.{i}.attn.W_q": Query projection for layer i
     * - "layers.{i}.attn.W_k": Key projection for layer i
     * - "layers.{i}.attn.W_v": Value projection for layer i
     * - "layers.{i}.attn.W_o": Output projection for layer i
     * - "layers.{i}.ffn.W_gate": Gate projection for layer i (SwiGLU)
     * - "layers.{i}.ffn.W_up": Up projection for layer i
     * - "layers.{i}.ffn.W_down": Down projection for layer i
     * - "layers.{i}.attn_norm_weight": Attention norm weight for layer i
     * - "layers.{i}.ffn_norm_weight": FFN norm weight for layer i
     * - "final_norm_weight": Final RMSNorm weight [d_model]
     * - "lm_head": Language modeling head [d_model, vocab_size]
     *
     * @param weights WeightMap containing loaded weights (from safetensors)
     * @param pre_quantized Optional pointer to map of pre-quantized tensors
     *                      If provided, these will be used instead of quantizing during load
     * @throws std::runtime_error if required weights are missing or have wrong shapes
     */
    void load_weights(
        const WeightMap& weights,
        const std::unordered_map<std::string, QuantizedTensor>* pre_quantized = nullptr
    ) {
        std::println("Loading weights into model...");

        size_t loaded_count = 0;
        const quant::QuantType attn_qtype = quant_config_.resolve_attention();
        const quant::QuantType ffn_qtype = quant_config_.resolve_feed_forward();
        const quant::QuantType lm_qtype = quant_config_.resolve_lm_head();

        // ================================================================
        // Load Token Embedding
        // ================================================================
        auto it = weights.find("token_embedding");
        if (it != weights.end()) {
            const Tensor& weight = it->second;

            // Validate shape
            if (weight.shape() != token_embedding_.shape()) {
                throw std::runtime_error(
                    "token_embedding shape mismatch: expected [" +
                    std::to_string(token_embedding_.shape()[0]) + ", " +
                    std::to_string(token_embedding_.shape()[1]) + "], got [" +
                    std::to_string(weight.shape()[0]) + ", " +
                    std::to_string(weight.shape()[1]) + "]"
                );
            }

            // Copy data
            std::memcpy(token_embedding_.data(), weight.data(), weight.size() * sizeof(float));
            loaded_count++;
        } else {
            std::println("  Warning: token_embedding not found in weights");
        }

        // ================================================================
        // Load Transformer Block Weights
        // ================================================================
        for (size_t layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
            std::string layer_prefix = "layers." + std::to_string(layer_idx) + ".";
            auto& block = blocks_[layer_idx];

            // Load attention weights (W_q, W_k, W_v, W_o)
            load_attention_weight(weights, layer_prefix, "W_q", block->attention().W_q(), attn_qtype,
                [&](QuantizedTensor tensor) { block->attention().set_quantized_W_q(std::move(tensor)); },
                loaded_count, pre_quantized);
            load_attention_weight(weights, layer_prefix, "W_k", block->attention().W_k(), attn_qtype,
                [&](QuantizedTensor tensor) { block->attention().set_quantized_W_k(std::move(tensor)); },
                loaded_count, pre_quantized);
            load_attention_weight(weights, layer_prefix, "W_v", block->attention().W_v(), attn_qtype,
                [&](QuantizedTensor tensor) { block->attention().set_quantized_W_v(std::move(tensor)); },
                loaded_count, pre_quantized);
            load_attention_weight(weights, layer_prefix, "W_o", block->attention().W_o(), attn_qtype,
                [&](QuantizedTensor tensor) { block->attention().set_quantized_W_o(std::move(tensor)); },
                loaded_count, pre_quantized);

            // Load FFN weights (W_gate, W_up, W_down for SwiGLU)
            load_ffn_weight(weights, layer_prefix, "W_gate", block->ffn().W_gate(), ffn_qtype,
                [&](QuantizedTensor tensor) { block->ffn().set_quantized_W_gate(std::move(tensor)); },
                loaded_count, pre_quantized);
            load_ffn_weight(weights, layer_prefix, "W_up", block->ffn().W_up(), ffn_qtype,
                [&](QuantizedTensor tensor) { block->ffn().set_quantized_W_up(std::move(tensor)); },
                loaded_count, pre_quantized);
            load_ffn_weight(weights, layer_prefix, "W_down", block->ffn().W_down(), ffn_qtype,
                [&](QuantizedTensor tensor) { block->ffn().set_quantized_W_down(std::move(tensor)); },
                loaded_count, pre_quantized);

            // Load normalization weights
            load_norm_weight(weights, layer_prefix, "attn_norm_weight", block->attn_norm_weight(), loaded_count);
            load_norm_weight(weights, layer_prefix, "ffn_norm_weight", block->ffn_norm_weight(), loaded_count);

            // Progress indicator
            if ((layer_idx + 1) % 5 == 0) {
                std::println("  Loaded weights for layer {}/{}", layer_idx + 1, config_.n_layers);
            }
        }

        // ================================================================
        // Load Final Norm Weight
        // ================================================================
        it = weights.find("final_norm_weight");
        if (it != weights.end()) {
            const Tensor& weight = it->second;

            if (weight.shape()[0] != final_norm_weight_.shape()[0]) {
                throw std::runtime_error("final_norm_weight shape mismatch");
            }

            std::memcpy(final_norm_weight_.data(), weight.data(), weight.size() * sizeof(float));
            loaded_count++;
        } else {
            std::println("  Warning: final_norm_weight not found");
        }

        // ================================================================
        // Load LM Head
        // ================================================================
        it = weights.find("lm_head");
        if (it != weights.end()) {
            const Tensor& weight = it->second;

            // PyTorch/safetensors stores lm_head transposed
            if (weight.shape()[0] != lm_head_.shape()[1] || weight.shape()[1] != lm_head_.shape()[0]) {
                throw std::runtime_error(
                    "lm_head shape mismatch: expected [" +
                    std::to_string(lm_head_.shape()[0]) + ", " +
                    std::to_string(lm_head_.shape()[1]) + "] or transposed [" +
                    std::to_string(lm_head_.shape()[1]) + ", " +
                    std::to_string(lm_head_.shape()[0]) + "], got [" +
                    std::to_string(weight.shape()[0]) + ", " +
                    std::to_string(weight.shape()[1]) + "]"
                );
            }

            // Transpose lm_head weight
            Tensor weight_T = ops::transpose(weight);
            std::memcpy(lm_head_.data(), weight_T.data(), weight_T.size() * sizeof(float));

            // Use pre-quantized if available, otherwise quantize now
            if (lm_qtype != quant::QuantType::NONE) {
                if (pre_quantized && pre_quantized->count("lm_head")) {
                    // Use pre-quantized tensor (already quantized in parallel)
                    // Make a copy since QuantizedTensor's copy constructor is deleted
                    lm_head_quant_ = pre_quantized->at("lm_head").copy();
                } else {
                    // Quantize the original weight (not transposed) because matvec expects [out_features, in_features]
                    // weight from safetensors already has shape [out_features, in_features]
                    lm_head_quant_ = QuantizedTensor::from_tensor(weight, lm_qtype);
                }
            }
            loaded_count++;
        } else {
            std::println("  Warning: lm_head not found");
        }

        std::println("  ✓ Successfully loaded {} weights into model", loaded_count);
    }





private:
    /**
     * @brief Helper method to load attention weight with quantization support
     */
    template<typename QuantSetter>
    void load_attention_weight(const WeightMap& weights,
                               const std::string& layer_prefix,
                               const std::string& weight_name,
                               Tensor& target,
                               quant::QuantType qtype,
                               QuantSetter&& quant_setter,
                               size_t& loaded_count,
                               const std::unordered_map<std::string, QuantizedTensor>* pre_quantized = nullptr) {
        std::string full_name = layer_prefix + "attn." + weight_name;
        auto it = weights.find(full_name);
        if (it != weights.end()) {
            const Tensor& weight = it->second;

            // Safetensors stores weights transposed - transpose them to match our layout
            // Expected: [out_features, in_features] but stored as [in_features, out_features]
            if (weight.shape()[0] != target.shape()[1] || weight.shape()[1] != target.shape()[0]) {
                throw std::runtime_error(
                    full_name + " shape mismatch: expected [" +
                    std::to_string(target.shape()[0]) + ", " +
                    std::to_string(target.shape()[1]) + "] or transposed [" +
                    std::to_string(target.shape()[1]) + ", " +
                    std::to_string(target.shape()[0]) + "], got [" +
                    std::to_string(weight.shape()[0]) + ", " +
                    std::to_string(weight.shape()[1]) + "]. " +
                    "For GQA: W_q uses n_heads=" + std::to_string(config_.n_heads) +
                    ", W_k/W_v use n_kv_heads=" + std::to_string(config_.n_kv_heads)
                );
            }

            // Transpose the weight matrix
            Tensor weight_T = ops::transpose(weight);
            std::memcpy(target.data(), weight_T.data(), weight_T.size() * sizeof(float));

            // Use pre-quantized if available, otherwise quantize now
            if (qtype != quant::QuantType::NONE) {
                if (pre_quantized && pre_quantized->count(full_name)) {
                    // Use pre-quantized tensor (already quantized in parallel)
                    // Make a copy since QuantizedTensor's copy constructor is deleted
                    quant_setter(pre_quantized->at(full_name).copy());
                } else {
                    // Quantize the original weight (not transposed) because matvec expects [out_features, in_features]
                    // weight from safetensors already has shape [out_features, in_features] = [256, 2048] for W_k
                    quant_setter(QuantizedTensor::from_tensor(weight, qtype));
                }
            }
            loaded_count++;
        } else {
            std::println("  Warning: {} not found", full_name);
        }
    }

    /**
     * @brief Helper method to load FFN weight with quantization support
     */
    template<typename QuantSetter>
    void load_ffn_weight(const WeightMap& weights,
                        const std::string& layer_prefix,
                        const std::string& weight_name,
                        Tensor& target,
                        quant::QuantType qtype,
                        QuantSetter&& quant_setter,
                        size_t& loaded_count,
                        const std::unordered_map<std::string, QuantizedTensor>* pre_quantized = nullptr) {
        std::string full_name = layer_prefix + "ffn." + weight_name;
        auto it = weights.find(full_name);
        if (it != weights.end()) {
            const Tensor& weight = it->second;

            // Validate shape (allowing for transpose)
            if (weight.shape()[0] != target.shape()[1] || weight.shape()[1] != target.shape()[0]) {
                throw std::runtime_error(
                    full_name + " shape mismatch: expected [" +
                    std::to_string(target.shape()[0]) + ", " +
                    std::to_string(target.shape()[1]) + "] or transposed [" +
                    std::to_string(target.shape()[1]) + ", " +
                    std::to_string(target.shape()[0]) + "], got [" +
                    std::to_string(weight.shape()[0]) + ", " +
                    std::to_string(weight.shape()[1]) + "]"
                );
            }

            // Transpose the weight matrix
            Tensor weight_T = ops::transpose(weight);
            std::memcpy(target.data(), weight_T.data(), weight_T.size() * sizeof(float));

            // Use pre-quantized if available, otherwise quantize now
            if (qtype != quant::QuantType::NONE) {
                if (pre_quantized && pre_quantized->count(full_name)) {
                    // Use pre-quantized tensor (already quantized in parallel)
                    // Make a copy since QuantizedTensor's copy constructor is deleted
                    quant_setter(pre_quantized->at(full_name).copy());
                } else {
                    // Quantize the original weight (not transposed) because matvec expects [out_features, in_features]
                    // weight from safetensors already has shape [out_features, in_features] = [256, 2048] for W_k
                    quant_setter(QuantizedTensor::from_tensor(weight, qtype));
                }
            }
            loaded_count++;
        } else {
            std::println("  Warning: {} not found", full_name);
        }
    }

    /**
     * @brief Helper method to load normalization weight
     */
    void load_norm_weight(const WeightMap& weights,
                         const std::string& layer_prefix,
                         const std::string& weight_name,
                         Tensor& target,
                         size_t& loaded_count) {
        std::string full_name = layer_prefix + weight_name;
        auto it = weights.find(full_name);
        if (it != weights.end()) {
            const Tensor& weight = it->second;

            // Validate shape (1D tensors)
            if (weight.shape()[0] != target.shape()[0]) {
                throw std::runtime_error(
                    full_name + " shape mismatch: expected [" +
                    std::to_string(target.shape()[0]) + "], got [" +
                    std::to_string(weight.shape()[0]) + "]"
                );
            }

            std::memcpy(target.data(), weight.data(), weight.size() * sizeof(float));
            loaded_count++;
        } else {
            std::println("  Warning: {} not found", full_name);
        }
    }

    ModelConfig config_;  // Model hyperparameters

    Tensor token_embedding_;  // Token embedding matrix: [vocab_size, d_model]
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;  // Transformer layers
    Tensor final_norm_weight_;  // Final RMSNorm weight: [d_model]
    Tensor lm_head_;  // Output projection: [d_model, vocab_size]
    std::optional<QuantizedTensor> lm_head_quant_;

    // KV Cache (Paged)
    std::unique_ptr<KVCacheManager> kv_manager_;
    std::vector<std::unique_ptr<PagedKVCache>> kv_caches_;  // One per layer
    bool kv_cache_initialized_ = false;
    quant::QuantConfig quant_config_;

public:
    /**
     * @brief Initialize Paged KV caches for all layers
     *
     * Allocates memory for the global KV pool and creates paged caches for each layer.
     *
     * @param max_seq_len Maximum sequence length (used to calculate total memory budget)
     * @param block_size Block size for paged attention (default 16)
     */
    void init_kv_cache(size_t max_seq_len, size_t block_size = 16) {
        if (kv_cache_initialized_) {
            reset_kv_cache();
            return;
        }

        std::println("Initializing Paged KV cache (max_seq_len={}, block_size={})...", max_seq_len, block_size);

        // Calculate total blocks needed
        // We need enough blocks for n_layers * max_seq_len
        // Total tokens capacity = n_layers * max_seq_len
        // Total blocks = ceil(Total tokens / block_size)
        // We add a small buffer (e.g. 10%) for fragmentation/overhead if we were handling multiple requests,
        // but for single sequence, exact calculation is fine.
        
        size_t total_tokens = config_.n_layers * max_seq_len;
        size_t max_num_blocks = (total_tokens + block_size - 1) / block_size;

        KVCacheConfig cache_config;
        cache_config.block_size = block_size;
        cache_config.max_num_blocks = max_num_blocks;
        cache_config.n_kv_heads = config_.n_kv_heads;
        cache_config.head_dim = config_.d_model / config_.n_heads;

        // Create manager
        kv_manager_ = std::make_unique<KVCacheManager>(cache_config);

        // Create one PagedKVCache per layer
        kv_caches_.clear();
        kv_caches_.reserve(config_.n_layers);
        for (size_t i = 0; i < config_.n_layers; ++i) {
            kv_caches_.push_back(std::make_unique<PagedKVCache>(kv_manager_.get()));
        }

        kv_cache_initialized_ = true;

        // Calculate total memory usage
        size_t total_elements = max_num_blocks * block_size * cache_config.n_kv_heads * cache_config.head_dim * 2; // K+V
        size_t total_memory = total_elements * sizeof(float);
        double total_mb = static_cast<double>(total_memory) / (1024.0 * 1024.0);

        std::println("  ✓ Paged KV cache initialized: {:.1f} MB ({} blocks)", total_mb, max_num_blocks);
    }

    void reset_kv_cache() {
        if (!kv_cache_initialized_) return;
        for (auto& cache : kv_caches_) {
            cache->reset();
        }
    }

    bool has_kv_cache() const { return kv_cache_initialized_; }

    size_t kv_cache_length() const {
        if (!kv_cache_initialized_ || kv_caches_.empty()) return 0;
        return kv_caches_[0]->current_length();
    }

}; // class LLMModel

} // namespace freellm
