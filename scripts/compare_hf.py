#!/usr/bin/env python3
"""Compare OLMoE outputs between freellm and HuggingFace.

This script loads the OLMoE model using HuggingFace transformers and dumps
intermediate activations that can be compared with freellm's debug output.

Usage:
    python scripts/compare_hf.py

Requirements:
    pip install torch transformers
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def dump_tensor(name: str, t: torch.Tensor, max_vals: int = 10):
    """Dump tensor statistics and first few values."""
    shape_str = ",".join(str(s) for s in t.shape)
    vals = t.flatten()[:max_vals].tolist()
    vals_str = " ".join(f"{v:.6f}" for v in vals)
    print(f"[HF DUMP] {name} shape=[{shape_str}] first {max_vals} values: {vals_str}")

def main():
    model_path = "models/OLMoE-1B-7B-0125"  # Use local model
    prompts = [
        "The capital of France is",
        "Paris is the capital of",
        "The quick brown fox jumps over"
    ]

    print("Loading model...")
    print(f"Model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()

    # Print model config
    print(f"\nModel config:")
    print(f"  vocab_size: {model.config.vocab_size}")
    print(f"  hidden_size (d_model): {model.config.hidden_size}")
    print(f"  intermediate_size (d_ff): {model.config.intermediate_size}")
    print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"  num_attention_heads: {model.config.num_attention_heads}")
    print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
    if hasattr(model.config, 'num_experts'):
        print(f"  num_experts: {model.config.num_experts}")
    if hasattr(model.config, 'num_experts_per_tok'):
        print(f"  num_experts_per_tok: {model.config.num_experts_per_tok}")
    if hasattr(model.config, 'norm_topk_prob'):
        print(f"  norm_topk_prob: {model.config.norm_topk_prob}")
    print()

    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt}")
        print('='*70)

        inputs = tokenizer(prompt, return_tensors="pt")
        token_ids = inputs.input_ids[0].tolist()
        print(f"Token IDs: {token_ids}")
        print(f"Tokens: {[tokenizer.decode([t]) for t in token_ids]}")

        with torch.no_grad():
            # Get embeddings
            embeddings = model.model.embed_tokens(inputs.input_ids)
            dump_tensor("embeddings", embeddings[0])

            # Run through first layer to get intermediate activations
            hidden = embeddings
            layer0 = model.model.layers[0]

            # Attention input norm
            normed = layer0.input_layernorm(hidden)
            dump_tensor("layer_0_input_layernorm", normed[0])

            # Get attention components
            attn = layer0.self_attn
            q = attn.q_proj(normed)
            k = attn.k_proj(normed)
            v = attn.v_proj(normed)
            dump_tensor("layer_0_Q (before reshape)", q[0])
            dump_tensor("layer_0_K (before reshape)", k[0])
            dump_tensor("layer_0_V (before reshape)", v[0])

            # Full forward with hidden states
            outputs = model(inputs.input_ids, output_hidden_states=True)
            logits = outputs.logits

            # Dump hidden states at key layers
            hidden_states = outputs.hidden_states
            dump_tensor("block_0_output", hidden_states[1][0])
            mid_layer = len(hidden_states) // 2
            dump_tensor(f"block_{mid_layer-1}_output (middle)", hidden_states[mid_layer][0])
            dump_tensor(f"block_{len(hidden_states)-2}_output (last)", hidden_states[-1][0])

            # Get final norm
            final_hidden = hidden_states[-1]
            final_normed = model.model.norm(final_hidden)
            dump_tensor("final_norm", final_normed[0])

            # Dump LAST position's final_norm specifically
            last_pos = final_normed.shape[1] - 1
            last_pos_vals = final_normed[0, last_pos, :10].tolist()
            vals_str = " ".join(f"{v:.6f}" for v in last_pos_vals)
            print(f"[HF DUMP] final_norm LAST position first 10 values: {vals_str}")

            # Logits
            dump_tensor("logits", logits[0])

            # Get top-5 predictions for last token
            last_logits = logits[0, -1]
            top5 = torch.topk(last_logits, 5)
            print(f"\n[HF DUMP] Top 5 predictions for last position:")
            for i, (idx, val) in enumerate(zip(top5.indices, top5.values)):
                token = tokenizer.decode([idx.item()])
                print(f"  {i+1}. token={idx.item()} ({repr(token)}) logit={val.item():.4f}")

            # Generate a few tokens to verify model works
            print("\n[HF Generation]")
            gen_outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=5,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.eos_token_id
            )
            generated = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
            print(f"  Generated: {generated}")

    print("\n" + "="*70)
    print("Done! Compare these values with freellm --dump-activations output.")
    print("="*70)

if __name__ == "__main__":
    main()
