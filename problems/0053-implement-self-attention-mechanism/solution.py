import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


def self_attention(Q, K, V):
    """Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Step 1: Get the key vector dimension (d_k)
    d_k = K.shape[1]

    # Step 2: Compute scaled dot-product scores: (seq_len, seq_len)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # Step 3: Apply softmax row-wise (with numerical stability adjustment)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 4: Compute context vectors by weighting Values
    output = np.dot(attention_weights, V)

    return output