import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = Q.shape[-1]
    
    # Calculate raw attention scores: (seq_len, seq_len)
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # Numerically stable softmax along the last axis (across keys for each query)
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Compute output weighted sum: (seq_len, d_k)
    output = attention_weights @ V
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    
    d_k = d_model // n_heads
    
    # Split feature dimension into multiple heads
    # Reshape (seq_len, d_model) -> (seq_len, n_heads, d_k)
    # Transpose to bring heads to the front: (n_heads, seq_len, d_k)
    Q_heads = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    
    # Compute scaled dot-product attention per head
    head_outputs = []
    for h in range(n_heads):
        out_h = self_attention(Q_heads[h], K_heads[h], V_heads[h])
        head_outputs.append(out_h)
        
    # Stack head outputs along the head dimension: (n_heads, seq_len, d_k)
    # Transpose back: (seq_len, n_heads, d_k)
    # Concatenate back to original model dimension: (seq_len, d_model)
    stacked = np.stack(head_outputs, axis=0)  # (n_heads, seq_len, d_k)
    transposed = stacked.transpose(1, 0, 2)   # (seq_len, n_heads, d_k)
    output = transposed.reshape(seq_len, d_model)
    
    return output