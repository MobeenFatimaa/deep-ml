import numpy as np

def deterministic_moe_backward(contributions: list, num_ranks: int, num_tokens: int, dim: int) -> list:
    """
    Deterministically accumulate MoE backward-pass token gradients.

    Args:
        contributions: list of (rank, token_id, grad_list) tuples.
        num_ranks: number of source ranks.
        num_tokens: number of destination tokens.
        dim: gradient dimensionality.

    Returns:
        Nested list of shape (num_tokens, dim) with the accumulated gradient.
    """
    # 1. Group contributions by rank
    rank_contributions = {r: [] for r in range(num_ranks)}
    for rank, token_id, grad in contributions:
        rank_contributions[rank].append((token_id, grad))
        
    # Allocate a separate buffer for each rank: shape (num_ranks, num_tokens, dim)
    buffers = np.zeros((num_ranks, num_tokens, dim), dtype=float)
    
    # 2. Token order pre-processing & buffer isolation per rank
    for rank in range(num_ranks):
        conts = rank_contributions[rank]
        if not conts:
            continue
            
        # Stable sort by token_id (Python's sorted is stable)
        sorted_conts = sorted(conts, key=lambda x: x[0])
        
        # Accumulate into rank's own buffer in order
        for token_id, grad in sorted_conts:
            buffers[rank, token_id] += np.array(grad, dtype=float)
            
    # 3. Deterministic reduction: Sum per-rank buffers in ascending rank order
    final_grads = np.sum(buffers, axis=0)
    
    return final_grads.tolist()