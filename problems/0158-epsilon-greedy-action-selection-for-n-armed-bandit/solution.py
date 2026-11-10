import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    """
    Selects an action using epsilon-greedy policy.
    Q: np.ndarray of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """
    n = len(Q)
    # With probability epsilon, explore by selecting a random action uniformly
    if np.random.rand() < epsilon:
        return int(np.random.choice(n))
    # With probability 1 - epsilon, exploit by selecting the greedy action
    else:
        return int(np.argmax(Q))