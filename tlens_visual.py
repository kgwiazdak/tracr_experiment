# %%
import torch
import plotly.express as px


def analyze_residual_stream(model, input_tokens):
    """Analyze and visualize the residual stream at each layer."""
    _, cache = model.run_with_cache(input_tokens)
    
    # Get all resid_pre activations
    resid_labels = [f"Layer {i}" for i in range(model.cfg.n_layers)]
    
    # Create frames for each layer
    frames = []
    for layer in range(model.cfg.n_layers):
        resid = cache[f"resid_post", layer][0].detach()
        # resid = cache["mlp_out", layer][0].detach()
        # resid = cache["attn_out", layer][0].detach()
 
        frames.append(dict(
            data=[dict(
                type='heatmap',
                z=resid.cpu().T,
                colorscale='RdBu',
                zmid=0
            )],
            name=f'Layer {layer}'
        ))
    
    # Create figure with slider
    fig = px.imshow(
        frames[0]['data'][0]['z'],
        title="Residual Stream Analysis",
        labels=dict(x="Position", y="Residual Dimension"),
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0
    )
    
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
            }]
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Layer: '},
            'steps': [{'args': [[f'Layer {i}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'label': str(i),
                        'method': 'animate'} for i in range(model.cfg.n_layers)]
        }]
    )
        
    fig.frames = frames
    fig.show()


def plot_attention_patterns(model, input_tokens, layer=0, head=0):
    """Visualize attention patterns for a given input."""
    _, cache = model.run_with_cache(input_tokens)
    
    pattern = cache["pattern", layer, head][0].detach()
    pattern = pattern.squeeze(0)
    
    token_labels = [f"{i}" for i in range(input_tokens.shape[1])]
    
    fig = px.imshow(
        pattern.cpu(),
        labels=dict(x="Key", y="Query"),
        x=token_labels,
        y=token_labels,
        title=f"Attention Pattern Layer {layer} Head {head}",
        color_continuous_scale="viridis"
    )
    fig.show()

def analyze_neurons(model, input_tokens, layer=0):
        """Analyze neuron activations in MLP layers."""
        _, cache = model.run_with_cache(input_tokens)
        
        mlp_acts = cache["mlp_out", layer][0].detach()
        
        fig = px.imshow(
            mlp_acts.cpu(),
            title=f"MLP Activations Layer {layer}",
            labels=dict(x="Position", y="Neuron"),
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0
        )
        fig.show()


def plot_parameters(model):
    """Plot the parameters of the model."""
    for name, param in model.named_parameters():
        param = param.detach().cpu()
        param = param.squeeze(0)
        # Only plot if the parameter is 2D
        if len(param.shape) == 2:
            fig = px.imshow(param)
            fig.update_layout(title=name)
            fig.show()
        else:
            print(f"Skipping {name} because it is not 2D, but shape {param.shape}")


def plot_activations(model, input_tokens, layer=0):
    """Plot the activations of the model."""
    _, cache = model.run_with_cache(input_tokens)
    activations = cache["mlp_out", layer][0].detach()
    fig = px.imshow(activations.cpu())
    fig.show()


if __name__ == "__main__":
    # analyze_neurons(tl_model, input_tokens, layer=0)
    # analyze_residual_stream(trained_model, input_tokens)

    # plot_weights(tl_model, layer=0)
    plot_parameters(trained_model)
    # plot_activations(tl_model, input_tokens, layer=0)
