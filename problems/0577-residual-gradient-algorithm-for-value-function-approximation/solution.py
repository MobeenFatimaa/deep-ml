import numpy as np

def residual_gradient_td(episodes, features, gamma, alpha, n_passes):
    """
    Run the Residual Gradient algorithm for policy evaluation
    with linear function approximation.
    
    Args:
        episodes: list of episodes, each is a list of (state, reward, next_state) tuples.
                    next_state = -1 indicates terminal (value 0).
        features: np.ndarray of shape (n_states, d), feature vectors per state.
        gamma: float, discount factor.
        alpha: float, step size.
        n_passes: int, number of passes through the data.
    
    Returns:
        np.ndarray of shape (d,): learned weight vector.
    """
    d = features.shape[1]
    w = np.zeros(d, dtype=float)
    
    for _ in range(n_passes):
        for episode in episodes:
            for state, reward, next_state in episode:
                phi_s = features[state]
                if next_state == -1:
                    phi_next = np.zeros(d, dtype=float)
                else:
                    phi_next = features[next_state]
                    
                v_s = np.dot(w, phi_s)
                v_next = np.dot(w, phi_next)
                
                delta = reward + gamma * v_next - v_s
                w += alpha * delta * (phi_s - gamma * phi_next)
                
    return w