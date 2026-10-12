import numpy as np

def td_lambda_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float,
    lambd: float,
    alpha: float
) -> np.ndarray:
    """
    Estimate state values using TD(λ) with accumulating eligibility traces.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
        n_states: Number of states (states are integers 0 to n_states-1)
        gamma: Discount factor
        lambd: Trace decay parameter (λ). Use 'lambd' to avoid Python keyword.
        alpha: Learning rate
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    V = np.zeros(n_states, dtype=float)
    
    for episode in episodes:
        e = np.zeros(n_states, dtype=float)
        T = len(episode)
        
        for t in range(T):
            s_t, r = episode[t]
            
            # Get value of next state, or 0 if terminal
            if t + 1 < T:
                s_next = episode[t + 1][0]
                v_next = V[s_next]
            else:
                v_next = 0.0
                
            # Compute TD error
            delta = r + gamma * v_next - V[s_t]
            
            # Update eligibility traces (decay and accumulate)
            e *= gamma * lambd
            e[s_t] += 1.0
            
            # Update value estimates for all states
            V += alpha * delta * e
            
    return V