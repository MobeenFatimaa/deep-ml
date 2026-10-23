import numpy as np

def anticipatory_moe(features, router_weights, expert_weights, delta):
    """
    Top-1 MoE forward pass with decoupled (anticipatory) routing.

    Args:
        features: list of T arrays, each of shape (N, d)
        router_weights: list of T arrays, each of shape (d, E)
        expert_weights: list of T arrays, each of shape (E, d, d_out)
        delta: non-negative integer, routing lookback offset

    Returns:
        List of T outputs, each a nested list of shape (N, d_out).
    """
    features = [np.array(f, dtype=np.float64) for f in features]
    router_weights = [np.array(r, dtype=np.float64) for r in router_weights]
    expert_weights = [np.array(e, dtype=np.float64) for e in expert_weights]

    T = len(features)
    outputs = []

    for t in range(T):
        if t >= delta:
            W_r = router_weights[t - delta]
        else:
            W_r = router_weights[t]

        feat_t = features[t]
        scores = feat_t @ W_r
        idxs = np.argmax(scores, axis=1)

        N = feat_t.shape[0]
        step_outputs = []
        curr_experts = expert_weights[t]

        for i in range(N):
            expert_idx = idxs[i]
            token_out = feat_t[i] @ curr_experts[expert_idx]
            step_outputs.append(token_out.tolist())

        outputs.append(step_outputs)

    return outputs