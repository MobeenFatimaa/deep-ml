import numpy as np

def td_lambda_convergence(
    episodes: list,
    n_states: int,
    gamma: float,
    lam: float,
    alpha: float,
    true_values: np.ndarray,
    tolerance: float
) -> tuple:
    """
    Run TD(lambda) with eligibility traces and monitor convergence.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of states
        gamma: Discount factor
        lam: Trace decay parameter (lambda)
        alpha: Learning rate
        true_values: True value function for RMSE computation
        tolerance: Convergence threshold on RMSE
    
    Returns:
        Tuple of (v_final, rmse_history, convergence_episode)
    """
    V = np.zeros(n_states, dtype=float)
    rmse_history = []
    convergence_episode = -1
    
    for ep_idx, episode in enumerate(episodes):
        e = np.zeros(n_states, dtype=float)
        
        for state, reward, next_state, done in episode:
            # TD error
            v_next = 0.0 if done else V[next_state]
            delta = reward + gamma * v_next - V[state]
            
            # Eligibility trace update (accumulating traces)
            e = gamma * lam * e
            e[state] += 1.0
            
            # Value update
            V += alpha * delta * e
            
        # Compute RMSE after each complete episode
        rmse = float(np.sqrt(np.mean((V - true_values) ** 2)))
        rmse_history.append(round(rmse, 4))
        
        if convergence_episode == -1 and rmse < tolerance:
            convergence_episode = ep_idx
            
    v_final = [round(float(v), 4) for v in V]
    
    return v_final, rmse_history, convergence_episode