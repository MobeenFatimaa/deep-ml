import numpy as np

def multi_head_latent_attention(
    X: np.ndarray,
    W_dkv: np.ndarray,
    W_uk: np.ndarray,
    W_uv: np.ndarray,
    W_dq: np.ndarray,
    W_uq: np.ndarray,
    W_o: np.ndarray,
    n_heads: int
) -> tuple:
    """
    Perform Multi-Head Latent Attention (MLA).
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_dkv: Down-projection for KV compression (d_model, d_c_kv)
        W_uk: Up-projection for keys (d_c_kv, d_model)
        W_uv: Up-projection for values (d_c_kv, d_model)
        W_dq: Down-projection for query compression (d_model, d_c_q)
        W_uq: Up-projection for queries (d_c_q, d_model)
        W_o: Output projection (d_model, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Tuple of (output, c_kv) where:
        - output: shape (seq_len, d_model)
        - c_kv: compressed KV latent of shape (seq_len, d_c_kv)
    """
    seq_len, d_model = X.shape
    if d_model % n_heads != 0:
        raise ValueError("d_model must be evenly divisible by n_heads.")
    
    d_k = d_model // n_heads
    
    # 1. Compress KV into low-dimensional latent representation
    c_kv = np.dot(X, W_dkv)
    
    # 2. Reconstruct keys and values from the compressed latent
    K = np.dot(c_kv, W_uk)
    V = np.dot(c_kv, W_uv)
    
    # 3. Compress and reconstruct queries
    c_q = np.dot(X, W_dq)
    Q = np.dot(c_q, W_uq)
    
    # 4. Reshape Q, K, V for multi-head attention: (seq_len, n_heads, d_k) -> transpose to (n_heads, seq_len, d_k)
    Q_heads = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    
    # Scaled dot-product attention per head
    scale = np.sqrt(d_k)
    scores = np.matmul(Q_heads, K_heads.transpose(0, 2, 1)) / scale
    
    # Numerically stable softmax (subtracting maximum per row)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Compute head outputs: shape (n_heads, seq_len, d_k)
    attn_output = np.matmul(attn_weights, V_heads)
    
    # Concatenate heads back: transpose to (seq_len, n_heads, d_k) then flatten to (seq_len, d_model)
    concat_output = attn_output.transpose(1, 0, 2).reshape(seq_len, d_model)
    
    # 5. Apply output projection
    output = np.dot(concat_output, W_o)
    
    # 6. Return the final output and the compressed KV latent
    return output, c_kv