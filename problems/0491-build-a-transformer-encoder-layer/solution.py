import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = X.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model must be evenly divisible by num_heads.")
    d_k = d_model // num_heads

    W_q = weights['W_q']
    W_k = weights['W_k']
    W_v = weights['W_v']
    W_o = weights['W_o']
    W1 = weights['W1']
    b1 = weights['b1']
    W2 = weights['W2']
    b2 = weights['b2']
    gamma1 = weights['gamma1']
    beta1 = weights['beta1']
    gamma2 = weights['gamma2']
    beta2 = weights['beta2']

    def layer_norm(tensor, gamma, beta):
        mean = np.mean(tensor, axis=-1, keepdims=True)
        var = np.var(tensor, axis=-1, keepdims=True)  # population variance (ddof=0)
        normed = (tensor - mean) / np.sqrt(var + eps)
        return gamma * normed + beta

    # 1. Multi-Head Self-Attention
    # Project queries, keys, values: shape (batch_size, seq_len, d_model)
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)

    # Reshape for multi-head attention: (batch_size, seq_len, num_heads, d_k) -> transpose to (batch_size, num_heads, seq_len, d_k)
    Q_heads = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scale = np.sqrt(d_k)
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / scale

    # Numerically stable softmax
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Compute attention output per head: (batch_size, num_heads, seq_len, d_k)
    attn_output = np.matmul(attn_weights, V_heads)

    # Concatenate heads: transpose back to (batch_size, seq_len, num_heads, d_k) and reshape to (batch_size, seq_len, d_model)
    concat_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # Output projection
    attn_proj = np.dot(concat_output, W_o)

    # 2. Add & Norm (first)
    x1 = layer_norm(X + attn_proj, gamma1, beta1)

    # 3. Position-wise Feed-Forward Network
    ffn_hidden = np.maximum(0.0, np.dot(x1, W1) + b1)  # ReLU activation
    ffn_output = np.dot(ffn_hidden, W2) + b2

    # 4. Add & Norm (second)
    out = layer_norm(x1 + ffn_output, gamma2, beta2)

    return out