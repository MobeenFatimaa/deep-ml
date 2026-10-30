import torch
import torch.nn as nn


def build_rope_cache(seq_len, head_dim, base=10000.0):
    """
    Builds the RoPE cache containing cosine and sine tensors.
    
    Args:
        seq_len: Maximum sequence length
        head_dim: Dimensionality of each attention head
        base: Frequency base for RoPE (default: 10000.0)
        
    Returns:
        tuple of (cos, sin), each of shape (seq_len, head_dim // 2)
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # Position indices
    t = torch.arange(seq_len).float()
    
    # Outer product to get theta angles: (seq_len, head_dim // 2)
    theta = torch.outer(t, inv_freq)
    
    return torch.cos(theta), torch.sin(theta)


def apply_rope(x, cos, sin):
    """
    Applies rotary position embeddings to tensor x using the rotate-half convention.
    
    Args:
        x: Tensor of shape (..., seq_len, head_dim)
        cos: Cosine tensor of shape (seq_len, head_dim // 2) or broadcastable
        sin: Sine tensor of shape (seq_len, head_dim // 2) or broadcastable
        
    Returns:
        Rotated tensor of the same shape as x
    """
    d = x.shape[-1]
    
    # Split x into halves along the last dimension
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    
    # Rotate half convention: (-x2, x1)
    rotated = torch.cat((-x2, x1), dim=-1)
    
    # Concatenate cos and sin along the last dimension to match head_dim
    cos_cat = torch.cat((cos, cos), dim=-1)
    sin_cat = torch.cat((sin, sin), dim=-1)
    
    return x * cos_cat + rotated * sin_cat