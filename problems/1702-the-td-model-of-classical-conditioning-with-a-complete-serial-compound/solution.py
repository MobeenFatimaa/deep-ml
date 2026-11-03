import numpy as np

def td_model_csc(trials, n_cs, trial_len, alpha, gamma, lam, w=None):
    """
    Implements the TD model of classical conditioning with a Complete Serial Compound (CSC).
    
    Args:
        trials (list): List of (cs_list, us) tuples per trial.
                       cs_list: list of (i, onset, offset) triples.
                       us: None or (u, magnitude).
        n_cs (int): Number of conditioned stimuli.
        trial_len (int): Number of time steps per trial (L).
        alpha (float): Step-size parameter.
        gamma (float): Discount factor.
        lam (float): Trace decay parameter (lambda).
        w (array-like, optional): Initial weights of shape (n_cs, trial_len).
        
    Returns:
        np.ndarray: Final weight vector of shape (n_cs, trial_len).
    """
    if w is None:
        weights = np.zeros((n_cs, trial_len), dtype=float)
    else:
        weights = np.array(w, dtype=float, copy=True)
        
    for cs_list, us in trials:
        # At the start of every trial, reset the trace z to zero
        z = np.zeros((n_cs, trial_len), dtype=float)
        
        for t in range(trial_len):
            # Construct feature vector x(S_t) for step t
            x_t = np.zeros((n_cs, trial_len), dtype=float)
            for i, onset, offset in cs_list:
                if onset <= t < offset:
                    x_t[i, t] = 1.0
                    
            # 1. Update eligibility trace: z <- gamma * lam * z + x(S_t)
            z = gamma * lam * z + x_t
            
            # Determine reward R_{t+1} on transition from step t to t+1
            r_next = 0.0
            if us is not None:
                u, magnitude = us
                if u == t + 1:
                    r_next = magnitude
                    
            # Compute value estimate at current state S_t
            v_t = np.sum(weights * x_t)
            
            # Compute value estimate at next state S_{t+1}
            if t + 1 < trial_len:
                x_next = np.zeros((n_cs, trial_len), dtype=float)
                for i, onset, offset in cs_list:
                    if onset <= (t + 1) < offset:
                        x_next[i, t + 1] = 1.0
                v_next = np.sum(weights * x_next)
            else:
                # Inter-trial state after step L-1 has no features, so v^(S_L) = 0
                v_next = 0.0
                
            # 2. Compute TD error delta_t
            delta_t = r_next + gamma * v_next - v_t
            
            # 3. Update weights: w <- w + alpha * delta_t * z
            weights += alpha * delta_t * z
            
    return weights