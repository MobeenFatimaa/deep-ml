import numpy as np

def gated_deltanet(q, k, v, a, b, g, A_log, rms_weight, eps=1e-6):
    """
    Simplified Gated DeltaNet linear attention forward pass.

    Args:
        q, k, v: lists of shape (T, d)
        a, b: lists of shape (T,) -- pre-activation gate logits
        g: list of shape (T, d) -- output gate logits
        A_log: float -- learned log time-scale
        rms_weight: list of shape (d,) -- RMSNorm scale
        eps: float -- numerical stability constant

    Returns:
        Nested list of shape (T, d), rounded to 4 decimals.
    """
    q = np.array(q, dtype=np.float64)
    k = np.array(k, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    g = np.array(g, dtype=np.float64)
    rms_weight = np.array(rms_weight, dtype=np.float64)

    T, d = q.shape
    S = np.zeros((d, d), dtype=np.float64)

    def sigmoid(x):
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def softplus(x):
        x_clipped = np.clip(x, -500, 500)
        return np.log(1.0 + np.exp(x_clipped))

    exp_A_log = np.exp(A_log)
    output = []

    for t in range(T):
        # 1. L2 normalize q and k
        q_t = q[t]
        k_t = k[t]
        q_norm_t = q_t / (np.linalg.norm(q_t) + eps)
        k_norm_t = k_t / (np.linalg.norm(k_t) + eps)

        # 2. Decay gate alpha
        alpha_t = np.exp(-softplus(a[t]) * exp_A_log)

        # 3. Update gate beta
        beta_t = sigmoid(b[t])

        # 4a. Decay S
        S = alpha_t * S

        # 4b. Prediction error delta
        # S^T k_norm_t
        s_k = S.T @ k_norm_t
        delta_t = (v[t] - s_k) * beta_t

        # 4c. Memory update S <- S + outer(k_norm_t, delta_t)
        S = S + np.outer(k_norm_t, delta_t)

        # 4d. Output o_t = S^T q_norm_t
        o_t = S.T @ q_norm_t

        # 5. RMSNorm
        rms = np.sqrt(np.mean(o_t**2) + eps)
        o_norm_t = (o_t / rms) * rms_weight

        # 6. SiLU output gate
        g_t = g[t]
        silu_g = g_t * sigmoid(g_t)
        y_t = o_norm_t * silu_g

        output.append([round(float(val), 4) for val in y_t])

    return output