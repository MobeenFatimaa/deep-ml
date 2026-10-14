import numpy as np

def semi_gradient_td_lambda(episodes: list, n_features: int, alpha: float, gamma: float, lam: float, initial_w: list = None) -> list:
    """
    Implement semi-gradient TD(lambda) with accumulating eligibility traces
    and linear function approximation.
    
    Args:
        episodes: List of episodes. Each episode is a list of tuples
                    (state_features, reward, next_state_features).
                    next_state_features is None for terminal transitions.
        n_features: Dimensionality of feature vectors
        alpha: Learning rate
        gamma: Discount factor
        lam: Lambda parameter for eligibility trace decay
        initial_w: Optional initial weight vector (defaults to zeros)
    
    Returns:
        Final weight vector as a list of floats rounded to 4 decimal places.
    """
    if initial_w is not None:
        w = np.array(initial_w, dtype=float)
    else:
        w = np.zeros(n_features, dtype=float)
        
    for episode in episodes:
        # Reset eligibility trace vector to zeros at the start of each episode
        z = np.zeros(n_features, dtype=float)
        
        for state_features, reward, next_state_features in episode:
            x = np.array(state_features, dtype=float)
            
            # Value estimate for current state: V(s) = w^T * x
            v_s = np.dot(w, x)
            
            # Value estimate for next state: V(s') = w^T * x' (0 if terminal)
            if next_state_features is None:
                v_next = 0.0
            else:
                x_next = np.array(next_state_features, dtype=float)
                v_next = np.dot(w, x_next)
                
            # TD error: delta = r + gamma * V(s') - V(s)
            delta = reward + gamma * v_next - v_s
            
            # Accumulating eligibility trace update: z = gamma * lambda * z + grad(V(s))
            # For linear function approximation, grad(V(s)) = x
            z = gamma * lam * z + x
            
            # Weight update: w = w + alpha * delta * z
            w += alpha * delta * z
            
    return [round(float(val), 4) for val in w]