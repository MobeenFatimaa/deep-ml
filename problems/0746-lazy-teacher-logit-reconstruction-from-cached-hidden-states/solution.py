import numpy as np

def reconstruct_teacher_logits(hidden_states, teacher_ids, teacher_heads):
    """
    Reconstruct per-sample teacher logits from cached hidden states.

    Args:
        hidden_states: array-like of shape (N, H)
        teacher_ids: array-like of length N, integer teacher indices
        teacher_heads: list of dicts with 'W' (V, H) and 'b' (V,)

    Returns:
        2D list of shape (N, V) with logits in original sample order.
    """
    hidden_states = np.asarray(hidden_states)
    teacher_ids = np.asarray(teacher_ids)
    
    N = len(hidden_states)
    if N == 0:
        return []

    # Process samples grouped by teacher index in ascending order
    unique_teachers = np.unique(teacher_ids)
    unique_teachers.sort()

    reconstructed_logits = [None] * N

    for t in unique_teachers:
        # Find indices corresponding to teacher t
        indices = np.where(teacher_ids == t)[0]
        
        # Extract weight matrix (V, H) and bias vector (V,)
        W = np.asarray(teacher_heads[t]['W'])
        b = np.asarray(teacher_heads[t]['b'])
        
        # Extract subset of hidden states: shape (N_t, H)
        h_t = hidden_states[indices]
        
        # Compute logits for batch: h_t @ W.T + b -> shape (N_t, V)
        logits_t = h_t @ W.T + b
        
        # Place rows back into their original index positions
        for idx, row in zip(indices, logits_t):
            reconstructed_logits[idx] = row.tolist()

    return reconstructed_logits