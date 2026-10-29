import numpy as np

def cross_layer_kv_attention(x, layer_weights, n_kv_producing_layers):
    """
    Multi-layer causal self-attention with cross-layer KV sharing.

    Args:
        x: list of lists, shape (seq_len, d_model)
        layer_weights: list of dicts, each containing 'Wq', 'Wk', 'Wv'
                       (each d_model x d_model)
        n_kv_producing_layers: int, number of initial layers that compute
                               their own K and V; later layers reuse the
                               most recent shared K, V.

    Returns:
        Final hidden state as a nested list, rounded to 4 decimal places.
    """
    h = np.array(x, dtype=float)
    seq_len, d_model = h.shape
    
    # Causal mask: shape (seq_len, seq_len)
    # Add -1e9 to positions j > i (upper triangle above the main diagonal)
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    
    shared_K = None
    shared_V = None
    
    for i, weights in enumerate(layer_weights):
        Wq = np.array(weights['Wq'], dtype=float)
        Wk = np.array(weights['Wk'], dtype=float)
        Wv = np.array(weights['Wv'], dtype=float)
        
        # 1. Compute Q from current hidden state
        Q = h @ Wq
        
        # 2. KV production or sharing
        if i < n_kv_producing_layers:
            shared_K = h @ Wk
            shared_V = h @ Wv
            
        K = shared_K
        V = shared_V
        
        # 3. Compute scaled dot-product causal self-attention
        scores = (Q @ K.T) / np.sqrt(d_model) + mask
        
        # Numerically stable softmax
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        attn = attn_weights @ V
        
        # 4. Apply a residual connection
        h = h + attn
        
    return [[round(float(val), 4) for val in row] for row in h]