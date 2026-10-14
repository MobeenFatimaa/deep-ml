import numpy as np

def td_lambda_forward_view(episodes: list, n_states: int, gamma: float, alpha: float, lam: float) -> list:
    """
    Implement forward-view TD(lambda) prediction.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of states
        gamma: Discount factor
        alpha: Learning rate
        lam: Lambda parameter for trace decay
    
    Returns:
        List of estimated state values rounded to 4 decimal places
    """
    V = np.zeros(n_states, dtype=float)
    
    for episode in episodes:
        T = len(episode)
        if T == 0:
            continue
            
        # Snapshot of V at the start of the episode
        V_old = V.copy()
        
        # Extract states, rewards, and dones
        states = [step[0] for step in episode]
        rewards = [step[1] for step in episode]
        dones = [step[3] for step in episode]
        
        lambda_returns = np.zeros(T, dtype=float)
        
        for t in range(T):
            H = T - t
            g_lambda = 0.0
            sum_weights = 0.0
            
            for n in range(1, H + 1):
                # Compute n-step return starting from time t
                G_n = 0.0
                terminated = False
                
                for i in range(n):
                    idx = t + i
                    G_n += (gamma ** i) * rewards[idx]
                    if dones[idx]:
                        terminated = True
                        break
                
                # Bootstrap if not terminated within these n steps
                if not terminated:
                    next_state = episode[t + n - 1][2]
                    G_n += (gamma ** n) * V_old[next_state]
                
                # Weight for n-step return
                if n < H:
                    weight = (1.0 - lam) * (lam ** (n - 1))
                else:
                    weight = lam ** (n - 1)
                    
                g_lambda += weight * G_n
                sum_weights += weight
                
            if sum_weights > 0:
                lambda_returns[t] = g_lambda / sum_weights
            else:
                lambda_returns[t] = V_old[states[t]]
                
        # Apply updates sequentially using the snapshot values
        for t in range(t if 't' in locals() else 0): # wait, let's just loop properly
            pass
            
        for t in range(T):
            s = states[t]
            V[s] += alpha * (lambda_returns[t] - V_old[s])
            
    return [round(float(v), 4) for v in V]