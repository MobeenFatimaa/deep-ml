import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, key_padding_mask=None, causal=False):
    """
    Computes scaled dot-product attention with optional key padding and causal masks.
    
    Args:
        q: Query tensor of shape (batch, heads, seq_q, head_dim)
        k: Key tensor of shape (batch, heads, seq_k, head_dim)
        v: Value tensor of shape (batch, heads, seq_k, head_dim)
        key_padding_mask: Boolean tensor of shape (batch, seq_k) where True = real token, False = padding
        causal: Boolean flag for causal (autoregressive) masking
        
    Returns:
        Attention output tensor
    """
    b, h, t_q, d = q.shape
    _, _, t_k, _ = k.shape
    
    attn_mask = None
    
    # 1. Handle Key Padding Mask
    if key_padding_mask is not None:
        # Reshape from (batch, seq_k) to (batch, 1, 1, seq_k) to broadcast across heads and queries
        attn_mask = key_padding_mask.view(b, 1, 1, t_k)
        
    # 2. Handle Causal Mask
    if causal:
        # Create lower-triangular boolean mask of shape (t_q, t_k)
        causal_mask = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril()
        
        if attn_mask is not None:
            # Combine padding mask and causal mask using logical AND
            attn_mask = attn_mask & causal_mask
        else:
            attn_mask = causal_mask
            
    # 3. Pass to SDPA (where True means 'may attend' and False means 'masked out')
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)