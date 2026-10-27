import numpy as np

def intra_option_q_learning(
    num_states: int,
    num_actions: int,
    num_options: int,
    option_policies: list,
    option_terminations: list,
    initiation_sets: list,
    transitions: list,
    alpha: float,
    gamma: float
) -> list:
    """
    Learn option-value function Q(s, o) using intra-option Q-learning.

    Args:
        num_states: Total number of states
        num_actions: Total number of primitive actions
        num_options: Total number of options
        option_policies: option_policies[o][s][a] = P(a | s, o)
        option_terminations: option_terminations[o][s] = P(terminate | s, o)
        initiation_sets: initiation_sets[o] = list of states where option o can start
        transitions: List of (state, action, reward, next_state) tuples
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        2D list of Q-values, shape (num_states, num_options), rounded to 4 decimals.
    """
    Q = np.zeros((num_states, num_options), dtype=float)
    
    for s, a, r, s_prime in transitions:
        available_options = [o for o in range(num_options) if s_prime in initiation_sets[o]]
        if available_options:
            max_q_prime = max(Q[s_prime, o_prime] for o_prime in available_options)
        else:
            max_q_prime = 0.0
            
        for o in range(num_options):
            if option_policies[o][s][a] > 0.0:
                beta = option_terminations[o][s_prime]
                u = (1.0 - beta) * Q[s_prime, o] + beta * max_q_prime
                target = r + gamma * u
                Q[s, o] += alpha * (target - Q[s, o])
                
    return np.round(Q, 4).tolist()