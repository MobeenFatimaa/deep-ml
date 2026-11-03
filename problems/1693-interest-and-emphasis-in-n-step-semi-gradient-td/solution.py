import numpy as np

def emphatic_nstep_td(episode, n, gamma, alpha, n_episodes, use_emphasis=True, w0=None):
    """
    Implements n-step semi-gradient TD with Interest and Emphasis.
    
    Args:
        episode (list): List of (x_t, R_{t+1}, I_t) tuples for t = 0, ..., T-1.
        n (int): Number of steps for n-step TD.
        gamma (float): Discount factor.
        alpha (float): Step-size parameter.
        n_episodes (int): Number of passes over the episode.
        use_emphasis (bool): If True, use computed M_t; otherwise M_t = 1 for all t.
        w0 (array-like, optional): Initial weight vector. Defaults to zeros.
        
    Returns:
        np.ndarray: The final learned weight vector.
    """
    T = len(episode)
    d = len(episode[0][0])
    
    # Initialize weights
    if w0 is None:
        w = np.zeros(d, dtype=float)
    else:
        w = np.array(w0, dtype=float)
        
    # Extract interest values
    I = [episode[t][2] for t in range(T)]
    
    # Compute emphasis M_t
    M = np.zeros(T, dtype=float)
    if use_emphasis:
        for t in range(T):
            it = I[t]
            m_prev = M[t - n] if t - n >= 0 else 0.0
            M[t] = it + (gamma ** n) * m_prev
    else:
        M = np.ones(T, dtype=float)
        
    # Replay the episode n_episodes times
    for ep in range(n_episodes):
        for t in range(T):
            h = min(n, T - t)
            G = 0.0
            
            # Accumulate n-step rewards
            for k in range(h):
                G += (gamma ** k) * episode[t + k][1]
                
            # Add bootstrap value if within horizon
            if t + h < T:
                x_bootstrap = np.array(episode[t + h][0], dtype=float)
                G += (gamma ** h) * np.dot(w, x_bootstrap)
                
            # State feature and value estimate at time t
            x_t = np.array(episode[t][0], dtype=float)
            v_t = np.dot(w, x_t)
            
            # TD error
            delta = G - v_t
            
            # Semi-gradient update scaled by emphasis M_t
            w += alpha * M[t] * delta * x_t
            
    return w