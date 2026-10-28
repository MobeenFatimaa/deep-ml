import numpy as np

def spatiotemporal_transformer_block(X: np.ndarray, params: dict) -> np.ndarray:
    """
    Forward pass of a spatiotemporal transformer block.
    
    Args:
        X: Input tensor of shape (T, N, D).
        params: Dict with attention and FFN weight matrices (see problem description).
    
    Returns:
        Output tensor of shape (T, N, D).
    """
    T, N, D = X.shape
    
    # Helper for LayerNorm: over feature dimension D, no affine params, eps = 1e-5
    def layer_norm(arr):
        mean = np.mean(arr, axis=-1, keepdims=True)
        var = np.var(arr, axis=-1, keepdims=True)
        return (arr - mean) / np.sqrt(var + 1e-5)

    # Helper for single-head scaled dot-product attention
    def self_attention(x_in, Wq, Wk, Wv, Wo):
        # x_in shape: (seq_len, D)
        Q = x_in @ Wq
        K = x_in @ Wk
        V = x_in @ Wv
        
        # Scaled dot-product attention: softmax(Q @ K^T / sqrt(D)) @ V
        scores = (Q @ K.T) / np.sqrt(D)
        
        # Stable softmax along the last dimension
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        attn_out = attn_weights @ V
        return attn_out @ Wo

    # --- Sub-Layer 1: Spatial Self-Attention ---
    # For each frame t, attend over its N tokens
    x = X.copy()
    spatial_out = np.zeros_like(x)
    for t in range(T):
        frame_input = x[t]  # shape (N, D)
        normed = layer_norm(frame_input)
        spatial_out[t] = self_attention(
            normed, 
            params['Wqs'], 
            params['Wks'], 
            params['Wvs'], 
            params['Wos']
        )
    x = x + spatial_out

    # --- Sub-Layer 2: Temporal Self-Attention ---
    # For each spatial position n, attend over its T tokens
    temporal_out = np.zeros_like(x)
    for n in range(N):
        spatial_input = x[:, n, :]  # shape (T, D)
        normed = layer_norm(spatial_input)
        temporal_out[:, n, :] = self_attention(
            normed, 
            params['Wqt'], 
            params['Wkt'], 
            params['Wvt'], 
            params['Wot']
        )
    x = x + temporal_out

    # --- Sub-Layer 3: Feed-Forward Network ---
    # ReLU(x @ W1) @ W2
    normed_ffn = layer_norm(x)
    ffn_hidden = np.maximum(0, normed_ffn @ params['W1'])
    ffn_out = ffn_hidden @ params['W2']
    x = x + ffn_out

    return x