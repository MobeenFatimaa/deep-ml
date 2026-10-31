import numpy as np

def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:
    """
    Compute Q^pi(s, a) for all state-action pairs using iterative policy evaluation
    based on the Bellman expectation equation for action-values.
    """
    # Initialize Q-values for all state-action pairs to 0.0
    Q = {(s, a): 0.0 for s in states for a in actions}
    
    while True:
        delta = 0.0
        new_Q = Q.copy()
        
        for s in states:
            for a in actions:
                q_val = 0.0
                for s_prime in states:
                    p = transition_probs.get((s, a, s_prime), 0.0)
                    if p == 0.0:
                        continue
                    r = rewards.get((s, a, s_prime), 0.0)
                    
                    # Compute expected value of next state under policy pi
                    next_expected_val = 0.0
                    for a_prime in actions:
                        pi_prob = policy.get((s_prime, a_prime), 0.0)
                        next_expected_val += pi_prob * Q.get((s_prime, a_prime), 0.0)
                    
                    q_val += p * (r + gamma * next_expected_val)
                
                new_Q[(s, a)] = q_val
                delta = max(delta, abs(q_val - Q[(s, a)]))
        
        Q = new_Q
        if delta < theta:
            break
            
    # Round final Q-values to 4 decimal places
    return {k: round(v, 4) for k, v in Q.items()}