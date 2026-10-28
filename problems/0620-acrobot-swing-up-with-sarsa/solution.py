import numpy as np

def sarsa_lambda_acrobot(
    features: list,
    rewards: list,
    num_weights: int,
    alpha: float,
    gamma: float,
    lam: float,
    initial_weights: np.ndarray,
    trace_type: str = "accumulating"
) -> list:
    """
    Run one episode of Sarsa(lambda) with linear function approximation.
    
    Args:
        features: List of binary feature vectors x(S_t, A_t) for t=0,...,T-1
        rewards: List of rewards R_{t+1} for t=0,...,T-1
        num_weights: Dimension of weight vector
        alpha: Learning rate
        gamma: Discount factor
        lam: Trace decay parameter (lambda)
        initial_weights: Initial weight vector (numpy array)
        trace_type: 'accumulating' or 'replacing'
    
    Returns:
        Updated weight vector as a list of floats
    """
    w = np.array(initial_weights, dtype=float)
    e = np.zeros(num_weights, dtype=float)
    
    T = len(features)
    
    for t in range(T):
        x_t = np.array(features[t], dtype=float)
        r_next = rewards[t]
        
        # Current Q-value: dot product of weights and feature vector
        q_current = np.dot(w, x_t)
        
        # Next Q-value (0 if terminal at t = T-1)
        if t + 1 < T:
            x_next = np.array(features[t+1], dtype=float)
            q_next = np.dot(w, x_next)
        else:
            q_next = 0.0
            
        # TD error computation
        delta = r_next + gamma * q_next - q_current
        
        # Update eligibility traces
        if trace_type == 'accumulating':
            e = gamma * lam * e + x_t
        elif trace_type == 'replacing':
            # For binary features with replacing traces, active features are set to 1, others decay
            e = gamma * lam * e
            e[x_t == 1] = 1.0
        else:
            raise ValueError(f"Unknown trace_type: {trace_type}")
            
        # Update weights
        w += alpha * delta * e
        
    return w.tolist()