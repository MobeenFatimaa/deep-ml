import torch

def to_float_tensor(values):
    """
    Converts a Python list (or list of lists) into a torch.float32 tensor.
    
    Args:
        values: Python list or nested list of numbers
        
    Returns:
        torch.Tensor with dtype torch.float32
    """
    return torch.tensor(values, dtype=torch.float32)