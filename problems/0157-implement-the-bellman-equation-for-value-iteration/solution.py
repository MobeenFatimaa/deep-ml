import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    n_states = len(transitions)
    new_V = np.zeros(n_states)
    
    for s in range(n_states):
        action_values = []
        # If a state has no actions available, its value defaults to 0 (or V[s])
        if not transitions[s]:
            new_V[s] = V[s]
            continue
            
        for a, outcomes in transitions[s].items():
            q_sa = 0.0
            for prob, next_state, reward, done in outcomes:
                # If done is True, the future value V(s') is omitted
                future_value = 0.0 if done else V[next_state]
                q_sa += prob * (reward + gamma * future_value)
            action_values.append(q_sa)
            
        # The Bellman optimality equation takes the maximum over all actions
        new_V[s] = max(action_values)
        
    return new_V