import numpy as np

def horde_parallel_gvfs(
    experience: list,
    demons: list,
    behavior_policy: np.ndarray,
    n_states: int,
    n_actions: int,
    alpha: float
) -> list:
    """
    Learn multiple General Value Functions in parallel from a single
    experience stream using off-policy TD(lambda) with importance sampling.
    
    Args:
        experience: List of (state, action, next_state) tuples.
        demons: List of demon configs, each a dict with keys:
            'cumulant': np.ndarray of shape (n_states, n_actions, n_states)
            'gamma': float, discount factor
            'target_policy': np.ndarray of shape (n_states, n_actions)
            'lambd': float, eligibility trace decay
        behavior_policy: np.ndarray of shape (n_states, n_actions) with behavior probabilities
        n_states: Number of states.
        n_actions: Number of actions.
        alpha: Learning rate.
    
    Returns:
        List of lists, one per demon, each of length n_states,
        with values rounded to 4 decimal places.
    """
    num_demons = len(demons)
    demon_values = []
    
    for demon in demons:
        cumulant = demon['cumulant']
        gamma = demon['gamma']
        target_policy = demon['target_policy']
        lambd = demon['lambd']
        
        V = np.zeros(n_states, dtype=float)
        e = np.zeros(n_states, dtype=float)
        
        for s, a, next_s in experience:
            # Importance sampling ratio
            b_prob = behavior_policy[s, a]
            rho = (target_policy[s, a] / b_prob) if b_prob > 0 else 0.0
            
            # Cumulant (pseudo-reward)
            r = cumulant[s, a, next_s]
            
            # TD error
            v_next = V[next_s]
            delta = r + gamma * v_next - V[s]
            
            # Eligibility trace update
            e = gamma * lambd * e
            e[s] += 1.0
            e = rho * e
            
            # Value update
            V += alpha * delta * e
            
        demon_values.append(np.round(V, 4).tolist())
        
    return demon_values