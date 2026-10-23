import math
import numpy as np


def fit_cost_model(samples):
    """Least-squares fit of t = a q^2 + b q c + d q; return (a, b, d) rounded to 6 decimals."""
    A = []
    y = []
    for q, c, t_ns in samples:
        A.append([q**2, q * c, q])
        y.append(t_ns)
    A = np.array(A, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return tuple(round(float(v), 6) for v in coeffs)


def chunk_for(remaining_ns, c, coeffs, cap=32768):
    """Largest integer q whose predicted cost fits in remaining_ns, capped."""
    if remaining_ns <= 0:
        return 0
    
    a, b, d = coeffs
    B_q = b * c + d
    
    if abs(a) < 1e-12:
        if B_q <= 0:
            q = cap
        else:
            q = remaining_ns / B_q
    else:
        discriminant = B_q**2 + 4.0 * a * remaining_ns
        if discriminant < 0:
            q = 0.0
        else:
            q = (-B_q + math.sqrt(discriminant)) / (2.0 * a)
            
    if q < 0:
        return 0
        
    q_int = math.floor(q)
    return int(min(q_int, cap))


def plan_step(queue, budgets, coeffs, cap=32768):
    """FIFO chunk plan for one step against the DP-group minimum budget."""
    remaining_ns = float(min(budgets))
    a, b, d = coeffs
    
    chunks = []
    initial_budget = remaining_ns
    
    for idx, (tokens_left, prefix_len) in enumerate(queue):
        if remaining_ns <= 0:
            break
        q = min(chunk_for(remaining_ns, prefix_len, coeffs, cap), tokens_left)
        if q <= 0:
            break
            
        cost = a * (q**2) + b * q * prefix_len + d * q
        remaining_ns -= cost
        chunks.append((idx, int(q)))
            
    used_ns = initial_budget - remaining_ns
    left_ns = remaining_ns
    
    return {
        "chunks": chunks,
        "used_ns": int(round(used_ns)),
        "left_ns": int(round(left_ns))
    }