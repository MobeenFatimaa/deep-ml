import numpy as np


def train_softmaxreg(
    X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int
) -> tuple[list[list[float]], list[float]]:
    """Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss."""
    N, D = X.shape
    C = int(np.max(y)) + 1

    # Add a bias column of ones as the first column of X
    X_bias = np.hstack([np.ones((N, 1)), X])
    M = D + 1

    # Initialize all coefficients to zero (C x M matrix)
    W = np.zeros((C, M))

    losses = []

    for _ in range(iterations):
        # Compute logits (N x C)
        logits = X_bias @ W.T

        # Softmax computation with numerical stability (subtract max per row)
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Compute sum-based Cross Entropy loss
        correct_probs = probs[np.arange(N), y]
        loss = -np.sum(np.log(np.clip(correct_probs, 1e-15, 1.0)))
        losses.append(float(loss))

        # Compute sum-based gradient (no division by N)
        Y_one_hot = np.zeros((N, C))
        Y_one_hot[np.arange(N), y] = 1.0

        grad_logits = probs - Y_one_hot
        grad_W = grad_logits.T @ X_bias

        # Update weights using gradient descent
        W -= learning_rate * grad_W

    # Round coefficients and losses for output consistency
    coeffs = [[float(val) for val in row] for row in W.tolist()]
    losses = [float(l) for l in losses]

    return coeffs, losses