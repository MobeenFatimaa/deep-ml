import numpy as np

def semi_markov_q_learning(
    episodes: list,
    n_states: int,
    n_actions: int,
    gamma: float,
    alpha: float
) -> list:
    """
    Implement Semi-Markov Q-Learning with variable action durations.
    
    Args:
        episodes: List of episodes, each a list of
                    (state, action, reward, next_state, duration, done) tuples
        n_states: Number of states
        n_actions: Number of actions
        gamma: Per-unit-time discount factor
        alpha: Learning rate
    
    Returns:
        Q-value table as nested list of shape (n_states, n_actions),
        rounded to 4 decimal places
    """
    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))
    
    for episode in episodes:
        for state, action, reward, next_state, duration, done in episode:
            if done:
                target = reward
            else:
                # Discount future value using gamma raised to the power of duration (tau)
                max_next_q = np.max(Q[next_state])
                target = reward + (gamma ** duration) * max_next_q
            
            # Standard Q-learning update step
            Q[state, action] += alpha * (target - Q[state, action])
            
    # Round to 4 decimal places and convert to a standard nested list
    return np.round(Q, 4).tolist()