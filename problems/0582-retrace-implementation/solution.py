import numpy as np

def retrace_targets(rewards: np.ndarray, Q_values: np.ndarray, expected_next_Q: np.ndarray, target_pi: np.ndarray, behavior_mu: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """
    Compute Retrace(lambda) targets for a trajectory of experience.
    
    Args:
        rewards: (T,) rewards at each step
        Q_values: (T,) Q(s_k, a_k) estimates
        expected_next_Q: (T,) E_pi[Q(s_{k+1}, .)] (0 if terminal)
        target_pi: (T,) target policy action probabilities
        behavior_mu: (T,) behavior policy action probabilities
        dones: (T,) 1.0 if episode terminates, 0.0 otherwise
        gamma: discount factor
        lam: trace decay parameter
    
    Returns:
        (T,) array of Retrace targets, rounded to 4 decimal places
    """
    T = len(rewards)
    
    # Compute trace coefficients: c_t = lambda * min(1, pi / mu)
    c = lam * np.minimum(1.0, target_pi / behavior_mu)
    
    # Compute TD errors: delta_t = r_t + gamma * (1 - d_t) * E_pi[Q(s_{t+1}, .)] - Q(s_t, a_t)
    delta = rewards + gamma * (1.0 - dones) * expected_next_Q - Q_values
    
    Q_ret = np.zeros(T)
    
    # Backward recursion
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            term = 0.0
        else:
            term = gamma * (1.0 - dones[t]) * c[t + 1] * (Q_ret[t + 1] - Q_values[t + 1])
            
        Q_ret[t] = Q_values[t] + delta[t] + term
        
    return np.round(Q_ret, 4)