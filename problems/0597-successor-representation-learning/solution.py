import numpy as np

def learn_successor_representation(
    experience: list,
    n_states: int,
    gamma: float,
    alpha_sr: float,
    alpha_w: float
) -> tuple:
    """
    Learn the Successor Representation from a stream of experience.
    
    Args:
        experience: List of (state, reward, next_state, done) tuples.
        n_states: Number of states in the environment.
        gamma: Discount factor.
        alpha_sr: Learning rate for SR matrix updates.
        alpha_w: Learning rate for reward weight updates.
    
    Returns:
        Tuple of (M, w, V) where:
            M: SR matrix as list of lists, shape (n_states, n_states)
            w: Reward weights as list, length n_states
            V: Value function as list, length n_states
        All values rounded to 4 decimal places.
    """
    M = np.zeros((n_states, n_states))
    w = np.zeros(n_states)
    
    for state, reward, next_state, done in experience:
        # 1. Update the reward weight for the current state
        w[state] += alpha_w * (reward - w[state])
        
        # 2. Compute the TD target for the SR matrix row
        target = np.zeros(n_states)
        target[state] = 1.0
        if not done:
            target += gamma * M[next_state]
            
        # Update the SR matrix row
        M[state] += alpha_sr * (target - M[state])
        
    # 3. Compute the value function V = M * w
    V = M @ w
    
    return (
        np.round(M, 4).tolist(),
        np.round(w, 4).tolist(),
        np.round(V, 4).tolist()
    )