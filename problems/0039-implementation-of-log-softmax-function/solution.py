import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    """Computes numerically stable log-softmax for a 1D NumPy array of scores."""
    # Subtract max for numerical stability (prevents overflow during exp)
    c = np.max(scores)
    # log(exp(z - c) / sum(exp(z - c))) = (z - c) - log(sum(exp(z - c)))
    return (scores - c) - np.log(np.sum(np.exp(scores - c)))


# Example usage
A = np.array([1, 2, 3])
print(np.round(log_softmax(A), 4))
# Output: [-2.4076 -1.4076 -0.4076]