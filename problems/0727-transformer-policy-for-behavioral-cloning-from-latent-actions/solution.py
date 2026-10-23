import numpy as np

def bc_transformer_loss(obs: np.ndarray, targets: np.ndarray, weights: dict) -> float:
    """
    Compute the behavioral cloning loss of a single-block transformer policy
    predicting discrete latent actions from a sequence of observation embeddings.

    Args:
        obs: Array of shape (T, d_model) containing observation embeddings.
        targets: Integer array of shape (T,) with ground-truth latent action indices.
        weights: Dict with keys 'W_q', 'W_k', 'W_v', 'W_o', 'W1', 'b1', 'W2', 'b2',
                 'W_head', 'b_head'.

    Returns:
        The mean token-level cross-entropy behavioral cloning loss as a float.
    """
    obs = np.array(obs, dtype=np.float64)
    targets = np.array(targets, dtype=np.int64)
    T, d_model = obs.shape

    # 1. Single-head causal self-attention
    W_q = weights['W_q']
    W_k = weights['W_k']
    W_v = weights['W_v']
    W_o = weights['W_o']

    Q = obs @ W_q
    K = obs @ W_k
    V = obs @ W_v

    scale = np.sqrt(d_model)
    scores = (Q @ K.T) / scale

    # Causal mask (upper triangle above diagonal is masked)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores[mask] = -1e9

    # Softmax
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    attn_out = attn_weights @ V
    attn_proj = attn_out @ W_o
    x1 = obs + attn_proj

    # 2. Position-wise two-layer MLP with ReLU
    W1 = weights['W1']
    b1 = weights['b1']
    W2 = weights['W2']
    b2 = weights['b2']

    h = np.maximum(0.0, x1 @ W1 + b1)
    mlp_out = h @ W2 + b2
    x2 = x1 + mlp_out

    # 3. Final linear head
    W_head = weights['W_head']
    b_head = weights['b_head']

    logits = x2 @ W_head + b_head

    # 4. Cross-entropy loss
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    log_sum_exp = max_logits.squeeze(-1) + np.log(np.sum(exp_logits, axis=-1))
    target_logits = logits[np.arange(T), targets]
    loss_per_token = log_sum_exp - target_logits

    return float(np.mean(loss_per_token))