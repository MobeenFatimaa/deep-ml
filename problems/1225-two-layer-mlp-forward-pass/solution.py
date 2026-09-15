import torch
import torch.nn as nn

def two_layer_mlp_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor
) -> float:
    """
    Constructs a 2-layer MLP (Linear -> ReLU -> Linear), assigns given weights/biases,
    and returns the forward output as a Python float.
    
    Args:
        x: Input tensor of shape (1, 2)
        w1: Weight tensor for first linear layer, shape (2, 2)
        b1: Bias tensor for first linear layer, shape (2,)
        w2: Weight tensor for second linear layer, shape (1, 2)
        b2: Bias tensor for second linear layer, shape (1,)
        
    Returns:
        Scalar output as a Python float
    """
    # 1. Define model architecture
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )
    
    # 2. Copy weights and biases into module parameters in-place without tracking gradients
    with torch.no_grad():
        model[0].weight.copy_(w1)
        model[0].bias.copy_(b1)
        model[2].weight.copy_(w2)
        model[2].bias.copy_(b2)
        
    # 3. Perform forward pass
    output = model(x)
    
    # 4. Return as Python float
    return output.item()