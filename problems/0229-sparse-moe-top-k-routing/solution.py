import numpy as np

def moe_topk_routing(
    router_logits: np.ndarray,
    expert_outputs: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Perform top-k expert routing for a Mixture-of-Experts layer.
    
    For each token:
    1. Select the top-k experts based on router_logits
    2. Compute softmax weights over only the selected experts
    3. Return weighted combination of the selected expert outputs
    
    Args:
        router_logits: Shape (batch_size, num_experts)
                      Raw scores from the router for each expert
        expert_outputs: Shape (batch_size, num_experts, hidden_dim)
                       Output from each expert for each input
        k: Number of experts to select per token
        
    Returns:
        Shape (batch_size, hidden_dim) - weighted combination of expert outputs
    """
    batch_size, num_experts = router_logits.shape
    hidden_dim = expert_outputs.shape[2]
    
    # Get the indices of the top-k logits for each token in the batch
    # np.argsort sorts in ascending order, so we take the last k elements and reverse them
    topk_indices = np.argsort(router_logits, axis=-1)[:, -k:][:, ::-1]
    
    # Initialize output array and routing weights matrix
    output = np.zeros((batch_size, hidden_dim))
    routing_weights = np.zeros((batch_size, num_experts))
    
    for i in range(batch_size):
        # Extract logits for the top-k experts of token i
        indices = topk_indices[i]
        logits = router_logits[i, indices]
        
        # Compute softmax over the top-k logits
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability trick
        weights = exp_logits / np.sum(exp_logits)
        
        # Store weights for reference or use directly
        routing_weights[i, indices] = weights
        
        # Compute the weighted combination of the selected expert outputs
        # expert_outputs[i] has shape (num_experts, hidden_dim)
        selected_outputs = expert_outputs[i, indices]  # Shape (k, hidden_dim)
        output[i] = np.sum(weights[:, np.newaxis] * selected_outputs, axis=0)
        
    return output