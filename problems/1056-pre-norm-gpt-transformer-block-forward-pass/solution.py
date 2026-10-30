import numpy as np

def pre_norm_transformer_block(x, params, num_heads):
    """
    Pre-norm Transformer block forward pass.

    Args:
        x: numpy array of shape (batch, seq_len, emb_dim)
        params: dict with keys 'ln1_gamma','ln1_beta','ln2_gamma','ln2_beta',
                'W_q','W_k','W_v','W_o','W_ff1','b_ff1','W_ff2','b_ff2'
        num_heads: int, number of attention heads (emb_dim must be divisible by num_heads)

    Returns:
        numpy array of shape (batch, seq_len, emb_dim)
    """
    batch, seq_len, emb_dim = x.shape
    d_head = emb_dim // num_heads
    eps = 1e-5

    # 1 & 2. First LayerNorm
    ln1_gamma = params['ln1_gamma']
    ln1_beta = params['ln1_beta']
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    h_ln1 = ln1_gamma * (x - mean) / np.sqrt(var + eps) + ln1_beta

    # 3, 4, 5 & 6. Causal Multi-Head Self-Attention
    # W_q, W_k, W_v, W_o shapes:
    # W_q: (emb_dim, emb_dim) or similar depending on convention. Usually weight matrices in transformers are (in_dim, out_dim) or (out_dim, in_dim).
    # Let's check typical shapes. Standard linear layer: h @ W + b. Here weights have no bias. 
    # Let's inspect standard shape conventions or write it generally using matmul.
    # In standard linear projections: Q = h @ W_q, K = h @ W_k, V = h @ W_v
    
    W_q = params['W_q']
    W_k = params['W_k']
    W_v = params['W_v']
    W_o = params['W_o']

    # Project Q, K, V
    # Assuming W_q shape is (emb_dim, emb_dim)
    Q = h_ln1 @ W_q  # (batch, seq_len, emb_dim)
    K = h_ln1 @ W_k
    V = h_ln1 @ W_v

    # Reshape for multi-head attention: (batch, seq_len, num_heads, d_head) -> transpose to (batch, num_heads, seq_len, d_head)
    Q = Q.reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)

    # Scaled dot-product attention scores
    # Q @ K^T / sqrt(d_head)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_head)  # (batch, num_heads, seq_len, seq_len)

    # Causal mask: set future positions to -inf
    causal_mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    scores = scores + causal_mask

    # Softmax over the last dimension
    # Subtract max for numerical stability
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Context vector
    context = np.matmul(attn_weights, V)  # (batch, num_heads, seq_len, d_head)

    # Concatenate heads and project with W_o
    context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, emb_dim)
    h_mha = context @ W_o

    # Residual connection: x + h_mha
    x_post_mha = x + h_mha

    # 7, 8, 9, 10, 11, 12, 13 & 14. Second LayerNorm & FFN
    ln2_gamma = params['ln2_gamma']
    ln2_beta = params['ln2_beta']
    mean2 = np.mean(x_post_mha, axis=-1, keepdims=True)
    var2 = np.var(x_post_mha, axis=-1, keepdims=True)
    h_ln2 = ln2_gamma * (x_post_mha - mean2) / np.sqrt(var2 + eps) + ln2_beta

    W_ff1 = params['W_ff1']
    b_ff1 = params['b_ff1']
    W_ff2 = params['W_ff2']
    b_ff2 = params['b_ff2']

    # FFN: GELU(h @ W_ff1 + b_ff1) @ W_ff2 + b_ff2
    ffn_hidden = h_ln2 @ W_ff1 + b_ff1
    # GELU tanh approximation: 0.5 * z * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 * z^3)))
    gelu_out = 0.5 * ffn_hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (ffn_hidden + 0.044715 * (ffn_hidden ** 3))))
    
    h_ffn = gelu_out @ W_ff2 + b_ff2

    # Final residual connection
    out = x_post_mha + h_ffn

    return out