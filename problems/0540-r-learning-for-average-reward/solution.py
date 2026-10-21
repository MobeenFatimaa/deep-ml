import numpy as np

def r_learning(transitions: list, n_states: int, n_actions: int, alpha: float, beta: float, initial_rho: float = 0.0) -> tuple:
    """
    R-Learning algorithm for average reward MDPs.
    
    Args:
        transitions: List of (state, action, reward, next_state) tuples.
        n_states: Number of states.
        n_actions: Number of actions.
        alpha: Learning rate for Q-values.
        beta: Learning rate for average reward estimate.
        initial_rho: Initial average reward estimate.
    
    Returns:
        Tuple of (Q, rho) where Q is a nested list and rho is a float,
        both rounded to 4 decimal places.
    """
    Q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]
    rho = float(initial_rho)
    
    for s, a, r, s_prime in transitions:
        max_q_next = max(Q[s_prime])
        delta = r - rho + max_q_next - Q[s][a]
        
        Q[s][a] += alpha * delta
        
        greedy_a = min(range(n_actions), key=lambda x: (-Q[s][x], x))
        
        if a == greedy_a:
            rho += beta * delta
            
    Q_rounded = [[round(val, 4) for val in row] for row in Q]
    rho_rounded = round(rho, 4)
    
    return Q_rounded, rho_rounded