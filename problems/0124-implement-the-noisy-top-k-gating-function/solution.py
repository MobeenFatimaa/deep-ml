import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    H_base = X @ W_g
    H_noise = X @ W_noise
    softplus = np.log1p(np.exp(H_noise))
    H = H_base + N * softplus
    
    batch_size, num_experts = H.shape
    topk_indices = np.argsort(H, axis=1)[:, -k:]
    
    masked_H = np.full_like(H, -np.inf)
    for i in range(batch_size):
        masked_H[i, topk_indices[i]] = H[i, topk_indices[i]]
        
    max_vals = np.max(masked_H, axis=1, keepdims=True)
    exp_H = np.exp(masked_H - max_vals)
    exp_H[masked_H == -np.inf] = 0.0
    sum_exp = np.sum(exp_H, axis=1, keepdims=True)
    
    return exp_H / sum_exp