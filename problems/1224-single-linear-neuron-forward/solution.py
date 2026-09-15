import torch
import torch.nn as nn

def single_neuron_forward(x: torch.Tensor) -> float:
    """
    Creates a linear neuron with 3 inputs and 1 output, sets fixed weights and bias,
    and returns the scalar output for a batch-1 input.
    
    Args:
        x: Input tensor of shape (1, 3)
        
    Returns:
        Scalar output as a Python float
    """
    # 1. Instantiate linear layer with 3 input features and 1 output feature
    layer = nn.Linear(3, 1)
    
    # 2. Set weights and bias in-place under torch.no_grad()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.5, -0.2, 0.3]]))
        layer.bias.copy_(torch.tensor([0.1]))
    
    # 3. Perform forward pass
    out = layer(x)
    
    # 4. Extract scalar float value
    return out.item()