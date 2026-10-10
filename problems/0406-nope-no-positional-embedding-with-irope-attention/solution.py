import numpy as np

def irope_attention(Q: list, K: list, V: list, positions: list, layer_index: int, rope_layers: list, base: float = 10000.0) -> dict:
    """
    Compute attention for a single layer in an iRoPE transformer.
    
    Args:
        Q: Query matrix, shape (seq_len, d_head)
        K: Key matrix, shape (seq_len, d_head)
        V: Value matrix, shape (seq_len, d_head)
        positions: Position indices, shape (seq_len,)
        layer_index: Current layer index (0-indexed)
        rope_layers: List of layer indices that use RoPE
        base: Base frequency for RoPE
    
    Returns:
        Dictionary with 'output', 'attention_weights', and 'uses_rope'
    """
    Q = np.array(Q, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)
    
    seq_len, d_head = Q.shape
    uses_rope = layer_index in rope_layers
    
    def apply_rope(x, pos):
        # x shape: (seq_len, d_head), pos shape: (seq_len,)
        # Rotate consecutive pairs of dimensions (2i, 2i+1)
        x_out = x.copy()
        half_dim = d_head // 2
        
        # Compute frequencies
        inv_freq = 1.0 / (base ** (np.arange(0, d_head, 2, dtype=np.float64) / d_head))
        
        for i in range(seq_len):
            p = pos[i]
            angles = p * inv_freq  # shape (half_dim,)
            cos_val = np.cos(angles)
            sin_val = np.sin(angles)
            
            row = x[i]
            x1 = row[0::2]
            x2 = row[1::2]
            
            x_out[i, 0::2] = x1 * cos_val - x2 * sin_val
            x_out[i, 1::2] = x1 * sin_val + x2 * cos_val
            
        return x_out

    if uses_rope:
        Q_out = apply_rope(Q, positions)
        K_out = apply_rope(K, positions)
    else:
        Q_out = Q
        K_out = K
        
    # Compute scaled dot-product attention
    scale = np.sqrt(d_head)
    scores = np.matmul(Q_out, K_out.T) / scale
    
    # Numerically stable softmax
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Attention output
    output = np.matmul(attention_weights, V)
    
    # Format outputs to rounded nested lists
    output_list = [[round(float(val), 4) for val in row] for row in output]
    weights_list = [[round(float(val), 4) for val in row] for row in attention_weights]
    
    return {
        'output': output_list,
        'attention_weights': weights_list,
        'uses_rope': uses_rope
    }