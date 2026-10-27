import numpy as np

def variance_reduced_td(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    alpha: float,
    gamma: float,
    c_bar: float
) -> dict:
    """
    Off-policy TD(0) prediction with standard and truncated importance sampling.

    Args:
        episodes: List of episodes, each a list of (state, action, reward) tuples.
        behavior_policy: b(a|s) as 2D list of shape (num_states, num_actions).
        target_policy: pi(a|s) as 2D list of shape (num_states, num_actions).
        num_states: Number of states.
        num_actions: Number of actions.
        alpha: Learning rate.
        gamma: Discount factor.
        c_bar: Truncation threshold for importance sampling ratios.

    Returns:
        Dictionary with V_standard, V_truncated, var_standard, var_truncated.
    """
    V_std = np.zeros(num_states, dtype=float)
    V_trunc = np.zeros(num_states, dtype=float)
    
    std_errors_per_state = [[] for _ in range(num_states)]
    trunc_errors_per_state = [[] for _ in range(num_states)]
    
    for episode in episodes:
        T = len(episode)
        for t in range(T):
            s_t, a_t, r_t = episode[t]
            if t < T - 1:
                s_next = episode[t + 1][0]
                v_next_std = V_std[s_next]
                v_next_trunc = V_trunc[s_next]
            else:
                v_next_std = 0.0
                v_next_trunc = 0.0
                
            rho = target_policy[s_t][a_t] / behavior_policy[s_t][a_t] if behavior_policy[s_t][a_t] > 0 else 0.0
            c = min(rho, c_bar)
            
            delta_std = r_t + gamma * v_next_std - V_std[s_t]
            weighted_std = rho * delta_std
            V_std[s_t] += alpha * weighted_std
            std_errors_per_state[s_t].append(weighted_std)
            
            delta_trunc = r_t + gamma * v_next_trunc - V_trunc[s_t]
            weighted_trunc = c * delta_trunc
            V_trunc[s_t] += alpha * weighted_trunc
            trunc_errors_per_state[s_t].append(weighted_trunc)
            
    var_std = np.zeros(num_states, dtype=float)
    var_trunc = np.zeros(num_states, dtype=float)
    
    for s in range(num_states):
        if std_errors_per_state[s]:
            var_std[s] = np.var(std_errors_per_state[s])
        if trunc_errors_per_state[s]:
            var_trunc[s] = np.var(trunc_errors_per_state[s])
            
    return {
        "V_standard": np.round(V_std, 4),
        "V_truncated": np.round(V_trunc, 4),
        "var_standard": np.round(var_std, 4),
        "var_truncated": np.round(var_trunc, 4)
    }