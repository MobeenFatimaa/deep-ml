import numpy as np

def bellman_error_minimization(
    features: np.ndarray,
    transition_probs: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    state_distribution: np.ndarray,
    learning_rate: float,
    n_iterations: int,
    initial_weights: np.ndarray = None
) -> tuple:
    """
    Minimize the Mean Squared Bellman Error using gradient descent
    dengan linear function approximation.
    
    Args:
        features: (n_states, n_features) feature matrix
        transition_probs: (n_states, n_states) transition matrix under the policy
        rewards: (n_states,) expected rewards under the policy
        gamma: discount factor
        state_distribution: (n_states,) weighting distribution over states
        learning_rate: gradient descent step size
        n_iterations: number of gradient descent updates
        initial_weights: (n_features,) initial weight vector, defaults to zeros
    
    Returns:
        Tuple of (weights, msbe, values)
    """
    n_states, n_features = features.shape
    
    if initial_weights is None:
        w = np.zeros(n_features)
    else:
        w = np.array(initial_weights, dtype=float)
        
    # Define matrix A = (I - gamma * P) * Phi
    I = np.eye(n_states)
    A = (I - gamma * transition_probs) @ features
    
    D = state_distribution
    
    # Gradient descent optimization on MSBE
    for _ in range(n_iterations):
        delta = A @ w - rewards
        # Gradient of MSBE: 2 * A^T @ D @ delta
        grad = 2.0 * A.T @ (D * delta)
        w -= learning_rate * grad
        
    # Compute final metrics
    delta_final = A @ w - rewards
    msbe = float(np.sum(D * (delta_final ** 2)))
    values = features @ w
    
    weights_list = [round(float(x), 4) for x in w]
    msbe_rounded = round(msbe, 4)
    values_list = [round(float(x), 4) for x in values]
    
    return weights_list, msbe_rounded, values_list