import numpy as np

def td_prediction(episodes: list, n_states: int, gamma: float, alpha: float, mode: str) -> list:
    """
    Perform TD(0) prediction in online or offline mode.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of states
        gamma: Discount factor
        alpha: Learning rate
        mode: 'online' or 'offline'
    
    Returns:
        List of estimated state values rounded to 4 decimal places
    """
    V = np.zeros(n_states, dtype=float)
    
    for episode in episodes:
        if mode == 'online':
            for state, reward, next_state, done in episode:
                target = reward if done else reward + gamma * V[next_state]
                td_error = target - V[state]
                V[state] += alpha * td_error
        elif mode == 'offline':
            V_start = V.copy()
            delta_V = np.zeros(n_states, dtype=float)
            for state, reward, next_state, done in episode:
                target = reward if done else reward + gamma * V_start[next_state]
                td_error = target - V_start[state]
                delta_V[state] += alpha * td_error
            V += delta_V
            
    return [round(float(v), 4) for v in V]