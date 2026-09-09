# MoE Design Document

## Architecture

We are adding Mixture-of-Experts (MoE) support to `freellm`.
The primary target is `allenai/OLMoE-1B-7B-0125` (and likely `Phi-3.5-MoE`).

### Components

1.  **Gating Network (Router)**
    - Config: `num_experts`, `k` (active experts).
    - Input: `x` [batch, seq_len, d_model]
    - Logic:
        - `logits = x @ W_gate` [..., num_experts]
        - `probs = softmax(logits)` (only on top-k?) -> Actually usually `softmax` over top-k or full then select.
        - OLMoE uses "input-independent noise" or standard top-k? -> Standard Top-K usually.
        - `selected_experts = top_k(logits)`
        - `weights = softmax(selected_logits)`

2.  **Experts**
    - `num_experts` independent FeedForward networks (SwiGLU).
    - In `OLMoE`, these are standard SwiGLU blocks.

3.  **Execution Flow (Naive CPU)**
    - For each token:
        - Compute gating logits.
        - Select top-K experts.
        - For each selected expert:
            - Calculate output.
            - Accumulate `weight * output` to final result.

### Data Structures

```cpp
struct MoEConfig {
    size_t num_experts;
    size_t num_experts_per_token;
    // ...
};

class MoELayer {
    Tensor W_gate; // [d_model, num_experts]
    std::vector<FeedForward> experts;

    Tensor forward(const Tensor& x) {
        // ... implementation ...
    }
};
```

### Weight Layout
Common MoE weight naming (HuggingFace):
- `model.layers.N.mlp.gate_proj.weight` (Router?) -> No, usually `gate_proj`, `up_proj`, `down_proj` are the FFN parts. The router is often `block_sparse_moe.gate`.
- **OLMoE specific**:
    - `model.layers.0.mlp.gate_proj` (expert 0 gate?) -> mixed.
    - Need to check `config.json` and `safetensors` structure of OLMoE.

**Action Item**: Inspect OLMoE model structure (using `safetensors_util` or python script if available, or just standard HF documentation).

## Integration
`TransformerBlock` will dispatch to `MoELayer` instead of `FeedForward` if `num_experts > 0`.
