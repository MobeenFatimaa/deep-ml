import numpy as np

def simulate_markov_chain(transition_matrix: np.ndarray, initial_state: int, num_steps: int) -> np.ndarray:
    """
    Simulate a Markov Chain given a transition matrix, initial state, and number of steps.
    
    Args:
        transition_matrix: 2D numpy array of shape (n_states, n_states) where rows sum to 1.
        initial_state: Integer index of the starting state.
        num_steps: Number of steps to simulate.
        
    Returns:
        np.ndarray: Sequence of state indices of length (num_steps + 1), including the initial state.
    """
    n_states = transition_matrix.shape[0]
    states = np.zeros(num_steps + 1, dtype=int)
    states[0] = initial_state
    
    current_state = initial_state
    for i in range(1, num_steps + 1):
        # Sample the next state based on the probability distribution of the current state
        probabilities = transition_matrix[current_state]
        current_state = np.random.choice(n_states, p=probabilities)
        states[i] = current_state
        
    return states