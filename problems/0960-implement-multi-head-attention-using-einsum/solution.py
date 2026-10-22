import numpy as np

def multi_head_attention_einsum(X, W_q, W_k, W_v, W_o, num_heads: int, causal: bool = True):
    """
    Multi-head self-attention computed with numpy.einsum.

    Args:
        X: input array of shape (B, S, d_in)
        W_q, W_k, W_v: projection matrices of shape (d_in, d_out)
        W_o: output projection matrix of shape (d_out, d_out)
        num_heads: number of attention heads (must divide d_out)
        causal: if True, apply a causal mask so position t only attends to <= t

    Returns:
        Nested list of shape (B, S, d_out) representing the attention output.
    """
    X = np.array(X, dtype=float)
    W_q = np.array(W_q, dtype=float)
    W_k = np.array(W_k, dtype=float)
    W_v = np.array(W_v, dtype=float)
    W_o = np.array(W_o, dtype=float)
    
    B, S, d_in = X.shape
    d_out = W_q.shape[1]
    d_head = d_out // num_heads
    
    # 1. Project X to Q, K, V using einsum: (B, S, d_in) x (d_in, d_out) -> (B, S, d_out)
    Q = np.einsum('bsd,do->bso', X, W_q)
    K = np.einsum('bsd,do->bso', X, W_k)
    V = np.einsum('bsd,do->bso', X, W_v)
    
    # 2. Reshape into heads: (B, S, num_heads, d_head) -> permute to (B, num_heads, S, d_head)
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    
    # 3. Compute per-head attention scores scaled by 1/sqrt(d_head)
    # Q: (B, h, S, d_head), K: (B, h, S_k, d_head) -> scores: (B, h, S, S_k)
    scale = 1.0 / np.sqrt(d_head)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale
    
    # 4. Apply causal masking when requested
    if causal:
        # Create lower-triangular mask of shape (S, S)
        mask = np.triu(np.ones((S, S), dtype=bool), k=1)
        scores[:, :, mask] = -1e9
        
    # 5. Apply softmax along the key axis (last axis)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 6. Aggregate values: (B, h, S, S_k) x (B, h, S_k, d_head) -> (B, h, S, d_head)
    context = np.einsum('bhqk,bhkd->bhqd', attn_weights, V)
    
    # Concatenate heads: transpose to (B, S, h, d_head) and reshape to (B, S, d_out)
    context = context.transpose(0, 2, 1, 3).reshape(B, S, d_out)
    
    # 7. Apply output projection using einsum: (B, S, d_out) x (d_out, d_out) -> (B, S, d_out)
    output = np.einsum('bso,oo->bso', context, W_o)
    
    return output.tolist()