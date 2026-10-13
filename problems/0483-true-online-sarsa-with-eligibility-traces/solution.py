import numpy as np

def true_online_sarsa_lambda(episodes, n_features, alpha, gamma, lam):
    """
    Implement True Online SARSA(lambda) with linear function approximation.

    Args:
        episodes: list of episodes. Each episode is a list of tuples
                    (x, reward, x_next, done) where x and x_next are
                    numpy arrays of shape (n_features,).
        n_features: int, dimension of feature vectors
        alpha: float, step size
        gamma: float, discount factor
        lam: float, trace decay parameter

    Returns:
        list of floats: learned weight vector rounded to 5 decimal places
    """
    w = np.zeros(n_features, dtype=float)
    
    for episode in episodes:
        if not episode:
            continue
            
        z = np.zeros(n_features, dtype=float)
        q_old = 0.0  # Initialize q_old to 0.0 at the start of each episode
        
        for t in range(len(episode)):
            x, reward, x_next, done = episode[t]
            
            q = float(np.dot(w, x))
            if done:
                q_next = 0.0
            else:
                q_next = float(np.dot(w, x_next))
                
            # TD error
            delta = reward + gamma * q_next - q
            
            # Eligibility trace update (Dutch trace)
            z = gamma * lam * z + (1.0 - alpha * gamma * lam * np.dot(z, x)) * x
            
            # Weight vector update
            w += alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x
            
            if done:
                break
                
            # Prepare for next step
            q_old = q_next
            
    return [round(float(val), 5) for val in w]