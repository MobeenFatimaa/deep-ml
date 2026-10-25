import numpy as np

def kanerva_coding_td(prototypes: np.ndarray, threshold: float, episodes: list,
                      gamma: float, alpha: float, query_states: np.ndarray) -> tuple:
    """
    Kanerva coding with semi-gradient TD(0) for value function approximation.
    
    Args:
        prototypes: Array of shape (k, d) - prototype points
        threshold: Euclidean distance threshold for activation
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        gamma: Discount factor
        alpha: Learning rate
        query_states: Array of shape (m, d) - states to evaluate
    
    Returns:
        Tuple of (weights, values) as lists of floats rounded to 4 decimal places
    """
    prototypes = np.array(prototypes)
    k, d = prototypes.shape
    w = np.zeros(k)
    
    def get_features(state):
        state_arr = np.array(state)
        distances = np.linalg.norm(prototypes - state_arr, axis=1)
        return (distances <= threshold).astype(float)
        
    for episode in episodes:
        for state, reward, next_state, done in episode:
            x = get_features(state)
            v_current = np.dot(w, x)
            
            if done:
                target = reward
            else:
                x_next = get_features(next_state)
                v_next = np.dot(w, x_next)
                target = reward + gamma * v_next
                
            td_error = target - v_current
            w += alpha * td_error * x
            
    # Evaluate query states
    values = []
    for q_state in query_states:
        x_q = get_features(q_state)
        values.append(float(np.dot(w, x_q)))
        
    return (
        np.round(w, 4).tolist(),
        np.round(values, 4).tolist()
    )