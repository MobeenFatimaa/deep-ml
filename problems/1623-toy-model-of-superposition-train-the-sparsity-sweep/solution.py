import numpy as np


def toy_forward(X, W, b):
    """(H, Z, Xhat) with H = X W^T, Z = H W + b, Xhat = relu(Z)."""
    H = X @ W.T
    Z = H @ W + b
    Xhat = np.maximum(Z, 0.0)
    return H, Z, Xhat


def toy_loss_and_grads(X, W, b, importance):
    """Importance-weighted reconstruction loss and exact gradients (loss, dW, db)."""
    B = X.shape[0]
    H, Z, Xhat = toy_forward(X, W, b)
    
    diff = Xhat - X
    weighted_diff_sq = importance * (diff ** 2)
    loss = np.sum(weighted_diff_sq) / B
    
    G = -2.0 * importance * (X - Xhat) * (Z > 0)
    db = np.mean(G, axis=0)
    dW = (1.0 / B) * (H.T @ G + (G @ W.T).T @ X)
    
    return loss, dW, db


def sample_features(rng, B, n, sparsity):
    values = rng.random((B, n))
    mask = rng.random((B, n)) < sparsity
    return values * (~mask)


def train_toy(rng, n, m, sparsity, importance, steps, lr, batch=256):
    W = rng.normal(size=(m, n)) * 0.1
    b = np.zeros(n)
    final_loss = 0.0
    for _ in range(steps):
        X = sample_features(rng, batch, n, sparsity)
        loss, dW, db = toy_loss_and_grads(X, W, b, importance)
        W -= lr * dW
        b -= lr * db
        final_loss = loss
    return W, b, final_loss


def dims_per_feature(W):
    m = W.shape[0]
    return m / np.sum(W ** 2)