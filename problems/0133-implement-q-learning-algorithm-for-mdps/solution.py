import numpy as np

def q_learning(num_states: int, num_actions: int, P: np.ndarray, R: np.ndarray, terminal_states: list, alpha: float, gamma: float, epsilon: float, num_episodes: int) -> np.ndarray:
    """
    Implement the Q-learning algorithm for tabular MDPs.
    
    Args:
        num_states: Total number of states
        num_actions: Total number of actions
        P: Transition probability array of shape (num_states, num_actions, num_states)
        R: Reward matrix of shape (num_states, num_actions)
        terminal_states: List of terminal state indices
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration probability for epsilon-greedy policy
        num_episodes: Number of training episodes
        
    Returns:
        np.ndarray: Learned Q-table of shape (num_states, num_actions)
    """
    Q = np.zeros((num_states, num_actions))
    non_terminal_states = sorted([s for s in range(num_states) if s not in terminal_states])
    
    for _ in range(num_episodes):
        if not non_terminal_states:
            break
            
        # Start episode at a randomly chosen non-terminal state
        state = np.random.choice(non_terminal_states)
        
        while state not in terminal_states:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state])  # np.argmax picks the lowest index on ties
                
            # Sample next state using transition probabilities
            next_state = np.random.choice(num_states, p=P[state, action])
            
            # Get reward
            r = R[state, action]
            
            # Compute target value
            if next_state in terminal_states:
                target = r
            else:
                target = r + gamma * np.max(Q[next_state])
                
            # Update Q-value
            Q[state, action] += alpha * (target - Q[state, action])
            
            # Transition to the next state
            state = next_state
            
    return Q