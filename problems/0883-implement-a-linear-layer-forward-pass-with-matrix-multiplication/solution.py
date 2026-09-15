import torch

def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the forward pass of a linear layer: y = x @ W^T + b
    
    Args:
        x: Tensor of shape (N, in_features)
        W: Tensor of shape (out_features, in_features)
        b: Tensor of shape (out_features,)
        
    Returns:
        Tensor of shape (N, out_features)
    """
    return torch.matmul(x, W.t()) + b