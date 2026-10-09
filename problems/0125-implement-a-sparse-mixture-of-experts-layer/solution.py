import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    n_batch, l_seq, d_model = x.shape
    
    # Flatten batch and sequence dimensions for token-level routing
    N = n_batch * l_seq
    x_flat = x.reshape(N, d_model)  # Shape: (N, d_model)
    
    # 1. Compute gating logits and apply numerically stable softmax
    logits = x_flat @ Wg  # Shape: (N, n_experts)
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=-1, keepdims=True)
    
    # 2. Select top-k experts per token
    top_k_indices = np.argsort(-probs, axis=-1)[:, :top_k]  # Shape: (N, top_k)
    top_k_probs = np.take_along_axis(probs, top_k_indices, axis=-1)  # Shape: (N, top_k)
    
    # 3. Renormalize top-k probabilities to sum to 1.0 per token
    top_k_probs = top_k_probs / np.sum(top_k_probs, axis=-1, keepdims=True)
    
    # 4. Compute weighted expert output
    out_flat = np.zeros_like(x_flat)
    
    for i in range(N):
        token = x_flat[i]
        for k in range(top_k):
            expert_idx = top_k_indices[i, k]
            gate_weight = top_k_probs[i, k]
            
            # Linear transformation: x_i @ W_expert
            expert_out = token @ We[expert_idx]
            out_flat[i] += gate_weight * expert_out
            
    return out_flat.reshape(n_batch, l_seq, d_model)