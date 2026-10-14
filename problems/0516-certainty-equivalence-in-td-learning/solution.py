import numpy as np

def certainty_equivalence_td(episodes, n_states, gamma):
    """
    Compute the certainty-equivalence value function from observed transition data.
    
    Args:
        episodes: list of episodes, each episode is a list of (state, reward, next_state) tuples
        n_states: int, number of states in the MDP
        gamma: float, discount factor
    
    Returns:
        np.ndarray: value function of shape (n_states,)
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    rewards_sum = np.zeros(n_states, dtype=float)
    visits = np.zeros(n_states, dtype=float)
    
    # Aggregate counts and rewards from observed transitions
    for episode in episodes:
        for s, r, next_s in episode:
            counts[s, next_s] += 1.0
            rewards_sum[s] += r
            visits[s] += 1.0
            
    # Estimate transition probabilities and expected rewards
    P_hat = np.zeros((n_states, n_states), dtype=float)
    R_hat = np.zeros(n_states, dtype=float)
    
    for s in range(n_states):
        if visits[s] > 0:
            P_hat[s, :] = counts[s, :] / visits[s]
            R_hat[s] = rewards_sum[s] / visits[s]
            
    # Solve the linear system (I - gamma * P_hat) * V = R_hat
    I = np.eye(n_states)
    A = I - gamma * P_hat
    V = np.linalg.solve(A, R_hat)
    
    return V