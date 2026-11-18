import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        P: First probability distribution
        Q: Second probability distribution
    
    Returns:
        Jensen-Shannon Divergence value
    """
    p_arr = np.array(P, dtype=float)
    q_arr = np.array(Q, dtype=float)
    
    # Compute midpoint distribution M
    m_arr = 0.5 * (p_arr + q_arr)
    
    def kl_divergence(a, b):
        # Handle cases where a_i = 0 (0 * log(...) = 0)
        # Also handle division safely where b_i > 0
        mask = (a > 0)
        term = np.zeros_like(a)
        term[mask] = a[mask] * np.log(a[mask] / b[mask])
        return np.sum(term)
    
    # Calculate JSD as the average of KL(P || M) and KL(Q || M)
    kl_p_m = kl_divergence(p_arr, m_arr)
    kl_q_m = kl_divergence(q_arr, m_arr)
    
    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
    
    return float(jsd)