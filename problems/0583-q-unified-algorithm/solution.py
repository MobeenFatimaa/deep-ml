import numpy as np

def q_sigma_return(
    states: list,
    actions: list,
    rewards: list,
    sigmas: list,
    Q: np.ndarray,
    pi: np.ndarray,
    b: np.ndarray,
    gamma: float
) -> float:
    """
    Compute the n-step Q(sigma) return.
    
    Args:
        states: List of n+1 state indices [S_t, S_{t+1}, ..., S_{t+n}]
        actions: List of n+1 action indices [A_t, A_{t+1}, ..., A_{t+n}]
        rewards: List of n rewards [R_{t+1}, R_{t+2}, ..., R_{t+n}]
        sigmas: List of n+1 sigma values [sigma_t, sigma_{t+1}, ..., sigma_{t+n}]
        Q: Q-value table, shape (n_states, n_actions)
        pi: Target policy probabilities, shape (n_states, n_actions)
        b: Behavior policy probabilities, shape (n_states, n_actions)
        gamma: Discount factor
    
    Returns:
        The n-step Q(sigma) return as a float.
    """
    n = len(rewards)
    G = Q[states[n], actions[n]]

    for k in range(n - 1, -1, -1):
        r = rewards[k]
        s_next = states[k + 1]
        a_next = actions[k + 1]
        sigma_next = sigmas[k + 1]
        
        rho = pi[s_next, a_next] / b[s_next, a_next] if b[s_next, a_next] > 0 else 0.0
        v_bar = np.sum(pi[s_next] * Q[s_next])
        
        weight = sigma_next * rho + (1.0 - sigma_next) * pi[s_next, a_next]
        G = r + gamma * (weight * (G - Q[s_next, a_next]) + v_bar)

    return float(G)