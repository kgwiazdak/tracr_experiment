import json
import os


def save_file(output_file, cache, tl_model):
    with open(output_file, "a") as f:
        for layer in range(tl_model.cfg.n_layers):
            attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"].detach().cpu().numpy()
            attn_scores = cache[f"blocks.{layer}.attn.hook_attn_scores"].detach().cpu().numpy()
            resid_pre = cache[f"blocks.{layer}.hook_resid_pre"].detach().cpu().numpy()
            resid_mid = cache[f"blocks.{layer}.hook_resid_mid"].detach().cpu().numpy()
            resid_post = cache[f"blocks.{layer}.hook_resid_post"].detach().cpu().numpy()

            q_values = cache[f"blocks.{layer}.attn.hook_q"].detach().cpu().numpy()
            k_values = cache[f"blocks.{layer}.attn.hook_k"].detach().cpu().numpy()
            v_values = cache[f"blocks.{layer}.attn.hook_v"].detach().cpu().numpy()

            mlp_pre = cache[f"blocks.{layer}.mlp.hook_pre"].detach().cpu().numpy()
            mlp_post = cache[f"blocks.{layer}.mlp.hook_post"].detach().cpu().numpy()
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"].detach().cpu().numpy()

            data = {
                f"attn_{layer}_pattern": attn_pattern.tolist(),
                f"attn_{layer}_scores": attn_scores.tolist(),
                f"resid_{layer}_pre": resid_pre.tolist(),
                f"resid_{layer}_mid": resid_mid.tolist(),
                f"resid_{layer}_post": resid_post.tolist(),
                f"q_values_{layer}": q_values.tolist(),
                f"k_values_{layer}": k_values.tolist(),
                f"v_values_{layer}": v_values.tolist(),
                f"mlp_{layer}_pre": mlp_pre.tolist(),
                f"mlp_{layer}_post": mlp_post.tolist(),
                f"mlp_{layer}_out": mlp_out.tolist()
            }
            json.dump(data, f)
            f.write("\n")
        pos_embed = cache["hook_pos_embed"].detach().cpu().numpy()
        json.dump({"pos_embed": pos_embed.tolist()}, f)
        f.write("\n")

