import numpy as np

def model_based_value_expansion(Q, transitions, rewards_model, V, experiences, H, gamma, alpha, terminal_states):
    """
    Perform model-based value expansion updates on Q-values.
    
    Args:
        Q: np.ndarray of shape (num_states, num_actions), initial Q-values
        transitions: dict mapping (state, action) -> next_state
        rewards_model: dict mapping (state, action) -> reward
        V: np.ndarray of shape (num_states,), value function for bootstrapping
        experiences: list of (state, action) tuples to update
        H: int, model rollout horizon
        gamma: float, discount factor
        alpha: float, learning rate
        terminal_states: list of int, terminal state indices
    
    Returns:
        Q: np.ndarray of shape (num_states, num_actions), updated Q-values rounded to 4 decimals
    """
    Q_updated = Q.copy()
    terminal_set = set(terminal_states)
    
    for s, a in experiences:
        G = 0.0
        discount = 1.0
        current_state = s
        current_action = a
        
        reached_terminal = False
        
        for step in range(H):
            # Check if model has transition and reward for current (state, action)
            if (current_state, current_action) not in transitions or (current_state, current_action) not in rewards_model:
                # If model is undefined, break out of rollout
                break
                
            r = rewards_model[(current_state, current_action)]
            next_state = transitions[(current_state, current_action)]
            
            G += discount * r
            discount *= gamma
            
            if next_state in terminal_set:
                reached_terminal = True
                break
                
            if step < H - 1:
                # Select greedy action at next_state using current Q-values
                current_action = int(np.argmax(Q_updated[next_state]))
                current_state = next_state
            else:
                current_state = next_state
                
        # Bootstrap if terminal state was not reached during the rollout
        if not reached_terminal:
            G += discount * V[current_state]
            
        # Update Q-value
        Q_updated[s, a] = (1.0 - alpha) * Q_updated[s, a] + alpha * G
        
    return np.round(Q_updated, 4)