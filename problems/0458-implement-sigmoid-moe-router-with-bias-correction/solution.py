import numpy as np

def sigmoid_moe_router(hidden_states: np.ndarray, gate_weight: np.ndarray, score_bias: np.ndarray, top_k: int) -> tuple:
    """
    Implement sigmoid-based MoE routing with bias correction.

    Args:
        hidden_states (np.ndarray): Token representations, shape (num_tokens, hidden_dim).
        gate_weight (np.ndarray): Gate projection weights, shape (num_experts, hidden_dim).
        score_bias (np.ndarray): Learned bias for load balancing, shape (num_experts,).
        top_k (int): Number of experts to select per token.

    Returns:
        tuple: (top_k_weights, top_k_indices)
            - top_k_weights: Normalized routing weights, shape (num_tokens, top_k).
            - top_k_indices: Selected expert indices, shape (num_tokens, top_k).
    """
    # 1. Compute router logits via matrix multiplication
    logits = np.dot(hidden_states, gate_weight.T)
    
    # 2. Apply sigmoid activation to get routing weights
    sigmoid_weights = 1.0 / (1.0 + np.exp(-logits))
    
    # 3. Add the score bias to determine expert selection
    selection_scores = sigmoid_weights + score_bias
    
    # 4. Select the top-k experts per token in descending order
    num_tokens = hidden_states.shape[0]
    top_k_indices = np.argsort(-selection_scores, axis=-1)[:, :top_k]
    
    # 5. Gather the actual sigmoid weights (without bias) for selected experts
    row_indices = np.arange(num_tokens)[:, None]
    top_k_weights = sigmoid_weights[row_indices, top_k_indices]
    
    # 6. Normalize the selected weights to sum to 1
    weight_sums = np.sum(top_k_weights, axis=-1, keepdims=True)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    top_k_weights = top_k_weights / weight_sums
    
    return top_k_weights, top_k_indices