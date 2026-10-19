import numpy as np

def gtd2_prediction(
    features: np.ndarray,
    transitions: list,
    gamma: float,
    alpha_w: float,
    alpha_v: float,
    w_init: np.ndarray,
    v_init: np.ndarray
) -> tuple:
    """
    Perform GTD2 (Gradient TD with second-order correction) for policy evaluation.
    
    Args:
        features: Feature matrix of shape (num_states, d)
        transitions: List of (state, reward, next_state, done) tuples
        gamma: Discount factor
        alpha_w: Learning rate for primary weights
        alpha_v: Learning rate for secondary weights
        w_init: Initial primary weight vector of shape (d,)
        v_init: Initial secondary weight vector of shape (d,)
    
    Returns:
        Tuple of (w, v) as lists of floats rounded to 4 decimal places
    """
    features = np.array(features, dtype=float)
    w = np.array(w_init, dtype=float)
    v = np.array(v_init, dtype=float)
    d = features.shape[1]
    
    for s, r, s_next, done in transitions:
        x = features[s]
        if done:
            x_next = np.zeros(d)
        else:
            x_next = features[s_next]
            
        # TD error: delta = r + gamma * w^T * x_next - w^T * x
        v_x = np.dot(w, x)
        v_x_next = np.dot(w, x_next)
        delta = r + gamma * v_x_next - v_x
        
        # Inner product x^T * v
        x_dot_v = np.dot(x, v)
        
        # Primary weights update: w += alpha_w * (x - gamma * x_next) * (x^T * v)
        w += alpha_w * (x - gamma * x_next) * x_dot_v
        
        # Secondary weights update: v += alpha_v * (delta - x^T * v) * x
        # Note: Depending on standard formulation, let's check the example reasoning:
        # Reasoning: v += 0.01 * (0.95 - 0.1) * [1,0] where delta = 0.95 and x^T v = 0.1
        v += alpha_v * (delta - x_dot_v) * x
        
    w_list = [round(float(val), 4) for val in w]
    v_list = [round(float(val), 4) for val in v]
    
    return w_list, v_list