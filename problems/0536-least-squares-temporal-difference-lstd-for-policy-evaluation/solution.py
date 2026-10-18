import numpy as np


def lstd(features, next_features, rewards, gamma, epsilon=0.01):
    """Least-Squares Temporal Difference (LSTD) for policy evaluation.

    Args:
        features: array-like of shape (T, d) - feature vectors for each state
        next_features: array-like of shape (T, d) - feature vectors for next states
        rewards: array-like of length T - observed rewards
        gamma: float - discount factor
        epsilon: float - regularization constant

    Returns:
        List of floats - weight vector w of length d
    """
    phi = np.array(features, dtype=float)
    phi_next = np.array(next_features, dtype=float)
    r = np.array(rewards, dtype=float)

    d = phi.shape[1]

    diff = phi - gamma * phi_next
    A = epsilon * np.eye(d) + np.dot(phi.T, diff)
    b = np.dot(phi.T, r)

    w = np.linalg.solve(A, b)
    return w.tolist()