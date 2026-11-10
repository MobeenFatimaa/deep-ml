import numpy as np

def expected_action_value(state, action, P, R, V, gamma):
    """
    Computes the expected value of taking `action` in `state` for the given MDP.
    Args:
      state: int or str, the current state
      action: str, the chosen action
      P: dict of dicts, P[s][a][s'] = prob of next state s' if a in s
      R: dict of dicts, R[s][a][s'] = reward for (s, a, s')
      V: np.ndarray, the value function vector, indexed by state
      gamma: float, discount factor
    Returns:
      float: expected value
    """
    expected_val = 0.0
    
    # Iterate over all possible next states for the given state and action
    transitions = P[state][action]
    rewards = R[state][action]
    
    for next_state, prob in transitions.items():
        reward = rewards[next_state]
        # Bellman equation component: prob * (reward + gamma * V[next_state])
        expected_val += prob * (reward + gamma * V[next_state])
        
    return float(expected_val)