def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float = 1e-3) -> list[list[float]]:
    """
    Perform policy evaluation for a 5x5 gridworld.
    
    Args:
        policy: Dictionary mapping each state (i, j) to action probabilities (dict of action -> prob).
        gamma: Discount factor.
        threshold: Convergence threshold for stopping iteration.
        
    Returns:
        List of lists (5x5) representing the state-value function V(s).
    """
    # Initialize state values to 0.0
    V = [[0.0 for _ in range(5)] for _ in range(5)]
    
    # Terminal states (four corners)
    terminal_states = {(0, 0), (0, 4), (4, 0), (4, 4)}
    
    # Action coordinate offsets
    actions_delta = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    
    while True:
        delta = 0.0
        V_new = [row[:] for row in V]
        
        for i in range(5):
            for j in range(5):
                state = (i, j)
                
                # Terminal states remain 0
                if state in terminal_states:
                    V_new[i][j] = 0.0
                    continue
                
                v = 0.0
                action_probs = policy.get(state, {})
                
                for action, prob in action_probs.items():
                    if prob == 0.0:
                        continue
                        
                    di, dj = actions_delta[action]
                    ni, nj = i + di, j + dj
                    
                    # If move goes out of bounds, agent stays in the current state
                    if not (0 <= ni < 5 and 0 <= nj < 5):
                        next_state = state
                    else:
                        next_state = (ni, nj)
                    
                    reward = -1.0
                    next_val = 0.0 if next_state in terminal_states else V[next_state[0]][next_state[1]]
                    
                    v += prob * (reward + gamma * next_val)
                
                V_new[i][j] = v
                delta = max(delta, abs(v - V[i][j]))
                
        V = V_new
        if delta < threshold:
            break
            
    return V