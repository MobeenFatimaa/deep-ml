import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution
        Q: Predicted probability distribution
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    p_arr = np.array(P, dtype=float)
    q_arr = np.array(Q, dtype=float)
    
    # Calculate Entropy H(P)
    # Using np.where to handle p_i = 0 safely (0 * ln(0) -> 0)
    log_p = np.where(p_arr > 0, np.log(p_arr), 0.0)
    h_p = -np.sum(p_arr * log_p)
    
    # Calculate Cross-Entropy H(P, Q)
    # Check if any p_i > 0 where q_i == 0
    if np.any((p_arr > 0) & (q_arr == 0)):
        h_pq = float('inf')
    else:
        # Filter terms where p_arr == 0 to avoid log(0) issues when q_arr > 0
        valid = p_arr > 0
        log_q = np.zeros_like(q_arr)
        log_q[valid] = np.log(q_arr[valid])
        h_pq = -np.sum(p_arr * log_q)
        
    return float(h_p), float(h_pq)