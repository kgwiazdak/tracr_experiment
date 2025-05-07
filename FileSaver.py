import json
import os
from typing import Dict, Any
import torch


def extract_tensor(cache: Dict[str, torch.Tensor], key: str) -> Any:
    """Safely extracts and converts a tensor from the cache to a NumPy list."""
    return cache[key].detach().cpu().numpy().tolist() if key in cache else None


def save_file(output_file: str, cache: Dict[str, torch.Tensor], tl_model: Any) -> None:
    """Saves model cache data to a JSON file in an optimized format."""
    with open(output_file, "a") as f:
        for layer in range(tl_model.cfg.n_layers):
            data = {
                f"attn_{layer}_pattern": extract_tensor(cache, f"blocks.{layer}.attn.hook_pattern"),
                f"attn_{layer}_scores": extract_tensor(cache, f"blocks.{layer}.attn.hook_attn_scores"),
                f"resid_{layer}_pre": extract_tensor(cache, f"blocks.{layer}.hook_resid_pre"),
                f"resid_{layer}_mid": extract_tensor(cache, f"blocks.{layer}.hook_resid_mid"),
                f"resid_{layer}_post": extract_tensor(cache, f"blocks.{layer}.hook_resid_post"),
                f"q_values_{layer}": extract_tensor(cache, f"blocks.{layer}.attn.hook_q"),
                f"k_values_{layer}": extract_tensor(cache, f"blocks.{layer}.attn.hook_k"),
                f"v_values_{layer}": extract_tensor(cache, f"blocks.{layer}.attn.hook_v"),
                f"mlp_{layer}_pre": extract_tensor(cache, f"blocks.{layer}.mlp.hook_pre"),
                f"mlp_{layer}_post": extract_tensor(cache, f"blocks.{layer}.mlp.hook_post"),
                f"mlp_{layer}_out": extract_tensor(cache, f"blocks.{layer}.hook_mlp_out"),
            }
            json.dump(data, f)
            f.write("\n")

        # Save positional embeddings
        pos_embed = extract_tensor(cache, "hook_pos_embed")
        if pos_embed is not None:
            json.dump({"pos_embed": pos_embed}, f)
            f.write("\n")

    print(f"Model cache successfully saved to {output_file}")
