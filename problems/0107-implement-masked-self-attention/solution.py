import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_model)
        K: Key matrix of shape (seq_len, d_model)
        V: Value matrix of shape (seq_len, d_model)
        mask: Mask matrix of shape (seq_len, seq_len) with 0 at allowed and -inf at blocked positions
        
    Returns:
        np.ndarray of shape (seq_len, d_model)
    """
    d_k = Q.shape[-1]
    
    # 1. Compute scaled dot-product attention scores: (Q @ K^T) / sqrt(d_k)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # 2. Apply the mask (adds -inf to blocked positions)
    if mask is not None:
        scores = scores + mask
        
    # 3. Compute stable softmax over the rows
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 4. Multiply attention weights by V
    output = np.dot(attention_weights, V)
    
    return output