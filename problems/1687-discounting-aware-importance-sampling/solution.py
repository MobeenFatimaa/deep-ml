import numpy as np

def discounting_aware_is(episodes, pi, b, gamma, state, first_visit=True):
    """
    Computes ordinary and weighted discounting-aware importance sampling estimates.
    
    Args:
        episodes: List of episodes, each a list of (S_k, A_k, R_{k+1}) tuples.
        pi: Target policy array of shape (n_states, n_actions).
        b: Behavior policy array of shape (n_states, n_actions).
        gamma: Discount factor.
        state: Target state to evaluate.
        first_visit: If True, consider only the first visit per episode; otherwise every visit.
        
    Returns:
        tuple: (ordinary, weighted) estimators as floats.
    """
    pi = np.asarray(pi, dtype=float)
    b = np.asarray(b, dtype=float)
    
    total_N = 0.0
    total_D = 0.0
    num_visits = 0
    
    for episode in episodes:
        T = len(episode)
        visited_in_episode = False
        
        for t in range(T):
            S_t, A_t, R_next = episode[t]
            if S_t == state:
                if first_visit and visited_in_episode:
                    continue
                visited_in_episode = True
                num_visits += 1
                
                N_t = 0.0
                D_t = 0.0
                
                rho = 1.0
                G_bar = 0.0
                
                # Iterate over horizons h from t+1 to T
                for h in range(t + 1, T + 1):
                    k = h - 1
                    s_k, a_k, r_next = episode[k]
                    
                    # Incrementally update the importance sampling ratio and flat partial return
                    rho *= pi[s_k, a_k] / b[s_k, a_k]
                    G_bar += r_next
                    
                    # Weight depends on whether h reaches the episode termination T
                    if h < T:
                        w_h = (1.0 - gamma) * (gamma ** (h - t - 1))
                    else:
                        w_h = gamma ** (T - t - 1)
                        
                    N_t += w_h * rho * G_bar
                    D_t += w_h * rho
                    
                total_N += N_t
                total_D += D_t
                
    ordinary = float(total_N / num_visits) if num_visits > 0 else 0.0
    weighted = float(total_N / total_D) if total_D > 0.0 else 0.0
    
    return (ordinary, weighted)