import numpy as np

def off_policy_mc_control(episodes: list, behavior_policy: dict, n_states: int, n_actions: int, gamma: float = 1.0) -> list:
    """
    Off-policy Monte Carlo control using weighted importance sampling.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, action, reward) tuples.
        behavior_policy: Dict mapping (state, action) -> probability under behavior policy.
        n_states: Number of states.
        n_actions: Number of actions.
        gamma: Discount factor.
    
    Returns:
        Q-values as a nested list of shape (n_states, n_actions), rounded to 4 decimal places.
    """
    Q = [[0.0] * n_actions for _ in range(n_states)]
    C = [[0.0] * n_actions for _ in range(n_states)]
    
    for episode in episodes:
        G = 0.0
        W = 1.0
        T = len(episode)
        
        for t in reversed(range(T)):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            
            # Determine greedy action for the current state (ties broken by lowest action index)
            best_action = max(range(n_actions), key=lambda a: (Q[state][a], -a))
            
            if action != best_action:
                break
            
            prob_b = behavior_policy.get((state, action), 1.0)
            if prob_b == 0.0:
                break
            W /= prob_b
            
    return [[round(float(val), 4) for val in row] for row in Q]