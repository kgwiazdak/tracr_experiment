# %%
import torch
import einops
import numpy as np
import plotly.express as px

from tracr.rasp import rasp
from tracr.compiler import compiling

from transformer_lens import HookedTransformer, HookedTransformerConfig


def extract_model_config(model):
    """Extracts configuration parameters from a Tracr model."""
    n_heads = model.model_config.num_heads
    n_layers = model.model_config.num_layers
    d_head = model.model_config.key_size
    d_mlp = model.model_config.mlp_hidden_size
    act_fn = "relu"
    normalization_type = "LN" if model.model_config.layer_norm else None
    attention_type = "causal" if model.model_config.causal else "bidirectional"

    n_ctx = model.params["pos_embed"]['embeddings'].shape[0]
    d_vocab = model.params["token_embed"]['embeddings'].shape[0]
    d_model = model.params["token_embed"]['embeddings'].shape[1]
    d_vocab_out = d_vocab - 2  # Exclude BOS and PAD tokens

    return HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out,
        d_mlp=d_mlp,
        n_heads=n_heads,
        act_fn=act_fn,
        attention_dir=attention_type,
        normalization_type=normalization_type,
    )

def extract_state_dict(model, d_model, d_vocab_out, n_heads, d_head, n_layers):
    """Extracts and reshapes state dictionary from Tracr model."""
    sd = {}
    sd["pos_embed.W_pos"] = model.params["pos_embed"]['embeddings']
    sd["embed.W_E"] = model.params["token_embed"]['embeddings']
    sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)

    for l in range(n_layers):
        sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/linear"]["w"],
            "(n_heads d_head) d_model -> n_heads d_head d_model",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]

        sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
        sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
        sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
        sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]

    return sd

def convert_to_torch_tensors(sd):
    """Converts state dict values to PyTorch tensors."""
    return {k: torch.tensor(np.array(v)) for k, v in sd.items()}

def create_model_input(input, input_encoder):
    """Creates model input tensor from raw input."""
    encoding = input_encoder.encode(input)
    return torch.tensor(encoding).unsqueeze(dim=0)

def decode_model_output(logits, output_encoder, bos_token):
    """Decodes model output logits."""
    max_output_indices = logits.squeeze(dim=0).argmax(dim=-1)
    decoded_output = output_encoder.decode(max_output_indices.tolist())
    decoded_output_with_bos = [bos_token] + decoded_output[1:]
    return decoded_output_with_bos

def convert_tracr_to_tl(tracr_model):
    """Converts a Tracr model to a TransformerLens model."""
    cfg = extract_model_config(tracr_model)
    tl_model = HookedTransformer(cfg)

    sd = extract_state_dict(
        tracr_model,
        d_model=cfg.d_model,
        d_vocab_out=cfg.d_vocab_out,
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        n_layers=cfg.n_layers
    )
    sd = convert_to_torch_tensors(sd)
    tl_model.load_state_dict(sd, strict=False)

    return tl_model

def verify_layer_outputs(tl_model, tracr_output, cache):
    """Verifies that TransformerLens and Tracr outputs match."""
    for layer in range(tl_model.cfg.n_layers):
        attn_match = np.isclose(
            cache["attn_out", layer].detach().cpu().numpy(),
            np.array(tracr_output.layer_outputs[2*layer])
        ).all()
        mlp_match = np.isclose(
            cache["mlp_out", layer].detach().cpu().numpy(),
            np.array(tracr_output.layer_outputs[2*layer+1])
        ).all()
        print(f"Layer {layer} Attn Out Equality Check:", attn_match)
        print(f"Layer {layer} MLP Out Equality Check:", mlp_match)

def plot_residual_stream(cache, input_sequence):
    """Plots the final residual stream."""
    px.imshow(
        cache["resid_post", -1].detach().cpu().numpy()[0],
        color_continuous_scale="Blues",
        labels={"x": "Residual Stream", "y": "Position"},
        y=[str(i) for i in input_sequence]
    ).show()

# %%

if __name__ == "__main__":
    from tracr_models import compile_reverse_model
    """
    Demonstrates analyzing a Tracr model that reverses sequences using TransformerLens.
    
    Example:
        Input: [BOS, 1, 2, 3]
        Output: [BOS, 3, 2, 1]
    """
    tracr_model = compile_reverse_model(
        max_seq_len=5,
        vocab={1, 2, 3, 4, 5},
    )
    
    tl_model = convert_tracr_to_tl(tracr_model)
    
    input_seq = ["BOS", 1, 3, 3, 1, 5]
    
    tracr_output = tracr_model.apply(input_seq)
    print(f"Tracr output: {tracr_output.decoded}")
    
    input_tokens = create_model_input(
        input_seq, 
        tracr_model.input_encoder
    )
    logits, cache = tl_model.run_with_cache(input_tokens)
    tl_output = decode_model_output(
        logits,
        tracr_model.output_encoder,
        tracr_model.input_encoder.bos_token
    )
    print(f"TransformerLens output: {tl_output}")
    
    # print("\nVerifying layer outputs match between models:")
    # verify_layer_outputs(tl_model, tracr_output, cache)
    
    # print("\nVisualizing final residual stream:")
    # plot_residual_stream(cache, input_seq)
    
# %%
