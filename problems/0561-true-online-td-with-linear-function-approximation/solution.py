import numpy as np

def true_online_td_lambda(episodes, n_features, alpha, gamma, lam):
    """
    True Online TD(lambda) for policy evaluation with linear function approximation.
    
    Args:
        episodes: List of episodes, each a list of (state_features, reward, next_state_features, done)
        n_features: Dimensionality of feature vectors
        alpha: Step size
        gamma: Discount factor
        lam: Trace decay parameter (lambda)
    
    Returns:
        Final weight vector as numpy array of shape (n_features,)
    """
    w = np.zeros(n_features)
    
    for episode in episodes:
        e = np.zeros(n_features)
        if not episode:
            continue
            
        phi = np.array(episode[0][0], dtype=float)
        v = np.dot(w, phi)
        v_old = v
        
        for state_features, reward, next_state_features, done in episode:
            phi = np.array(state_features, dtype=float)
            v = np.dot(w, phi)
            
            if done:
                v_prime = 0.0
            else:
                phi_prime = np.array(next_state_features, dtype=float)
                v_prime = np.dot(w, phi_prime)
                
            delta = reward + gamma * v_prime - v
            
            e = gamma * lam * e + phi - alpha * gamma * lam * np.dot(e, phi) * phi
            w = w + alpha * (delta + v - v_old) * e - alpha * (v - v_old) * phi
            
            v_old = v_prime
            
    return w