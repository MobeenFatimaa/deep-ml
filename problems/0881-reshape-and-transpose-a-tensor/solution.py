import torch

def flatten_then_reshape(x: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    """
    Flattens a tensor to 1-D and reshapes it into new_shape.
    
    Args:
        x: Input torch.Tensor
        new_shape: Tuple specifying the output shape
        
    Returns:
        Reshaped torch.Tensor
    """
    # .reshape(-1) flattens to 1D, then .reshape(new_shape) lays it out into the new dimensions
    return x.reshape(-1).reshape(new_shape)

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    """
    Swaps the last two dimensions of a tensor.
    
    Args:
        x: Input torch.Tensor of at least 2 dimensions
        
    Returns:
        Tensor with the last two dimensions swapped
    """
    # Negative indices reference dimensions relative to the end (-2: second to last, -1: last)
    return x.transpose(-2, -1)