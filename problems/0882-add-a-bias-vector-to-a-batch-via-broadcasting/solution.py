import torch

def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Adds a bias vector b of shape (D,) to each row of a batch tensor x of shape (N, D).
    
    Args:
        x: Input batch tensor of shape (N, D)
        b: Bias vector tensor of shape (D,)
        
    Returns:
        Tensor of shape (N, D) containing (x + b)
    """
    return x + b