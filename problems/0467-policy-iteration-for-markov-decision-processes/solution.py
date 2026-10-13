import numpy as np

def policy_iteration(num_states: int, num_actions: int, transitions: list, gamma: float, theta: float = 1e-8) -> tuple:
    """
    Implement the Policy Iteration algorithm for solving MDPs.
    
    Args:
        num_states: Number of states in the MDP
        num_actions: Number of actions available in each state
        transitions: transitions[s][a] = [(prob, next_state, reward), ...]
        gamma: Discount factor
        theta: Convergence threshold for policy evaluation
    
    Returns:
        Tuple of (policy, values) where policy is a list of ints
        and values is a list of floats rounded to 4 decimal places
    """
    policy = [0] * num_states
    values = [0.0] * num_states
    
    while True:
        # 1. Policy Evaluation
        while True:
            delta = 0.0
            new_values = list(values)
            for s in range(num_states):
                a = policy[s]
                v = 0.0
                for prob, next_state, reward in transitions[s][a]:
                    v += prob * (reward + gamma * values[next_state])
                delta = max(delta, abs(v - values[s]))
                new_values[s] = v
            values = new_values
            if delta < theta:
                break
                
        # 2. Policy Improvement
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s]
            
            # Compute Q-values for all actions in state s
            action_values = []
            for a in range(num_actions):
                q = 0.0
                for prob, next_state, reward in transitions[s][a]:
                    q += prob * (reward + gamma * values[next_state])
                action_values.append(q)
                
            # Find best action, breaking ties by choosing the smallest index
            best_action = max(range(num_actions), key=lambda a: (action_values[a], -a))
            
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False
                
        if policy_stable:
            break
            
    # Round values to 4 decimal places
    values = [round(float(v), 4) for v in values]
    
    return policy, values