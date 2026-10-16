import numpy as np

def linear_sarsa(
    episodes: list,
    features: dict,
    n_features: int,
    alpha: float,
    gamma: float
) -> list:
    """
    Episodic semi-gradient Sarsa with linear function approximation.
    
    Args:
        episodes: List of episodes, each a list of (state, action, reward) tuples.
        features: Dict mapping (state, action) -> feature vector (list of floats).
        n_features: Number of features.
        alpha: Learning rate.
        gamma: Discount factor.
    
    Returns:
        Final weight vector as a list of floats rounded to 4 decimal places.
    """
    w = np.zeros(n_features, dtype=np.float64)
    
    for episode in episodes:
        T = len(episode)
        for t in range(T):
            state, action, reward = episode[t]
            x_current = np.array(features[(state, action)], dtype=np.float64)
            q_current = np.dot(w, x_current)
            
            if t + 1 < T:
                next_state, next_action, _ = episode[t + 1]
                x_next = np.array(features[(next_state, next_action)], dtype=np.float64)
                q_next = np.dot(w, x_next)
            else:
                q_next = 0.0
                
            # TD error: delta = R + gamma * Q(S', A') - Q(S, A)
            delta = reward + gamma * q_next - q_current
            
            # Update weights: w = w + alpha * delta * x
            w += alpha * delta * x_current
            
    return [round(float(val), 4) for val in w]