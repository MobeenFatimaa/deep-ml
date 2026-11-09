import numpy as np

def _sigmoid(x):
    # Numerically stable sigmoid function
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def autorec_forward(Y, V, mu, W, b):
    """
    Computes the Item-based AutoRec forward pass.
    Y: (M, N) user-by-item ratings matrix.
    V: (K, M) encoder weight matrix.
    mu: (K,) encoder bias vector.
    W: (K, M) decoder weight matrix.
    b: (M,) decoder bias vector.
    Returns: (M, N) reconstructed output matrix.
    """
    h = _sigmoid(np.dot(V, Y) + mu[:, np.newaxis])
    out = np.dot(W.T, h) + b[:, np.newaxis]
    return out

def autorec_loss(Y, mask, out, W, V, lam):
    """
    Computes the regularized reconstruction loss over observed entries.
    """
    diff = (Y - out) * mask
    recon_loss = np.sum(diff ** 2)
    reg_loss = 0.5 * lam * (np.sum(W ** 2) + np.sum(V ** 2))
    return float(recon_loss + reg_loss)

def autorec_grads(Y, mask, V, mu, W, b, lam):
    """
    Computes exact gradients of the loss with respect to V, mu, W, and b.
    Returns a dictionary with keys 'V', 'mu', 'W', 'b'.
    """
    M, N = Y.shape
    K = V.shape[0]
    
    # Forward pass intermediates
    z = np.dot(V, Y) + mu[:, np.newaxis]  # (K, N)
    h = _sigmoid(z)                      # (K, N)
    out = np.dot(W.T, h) + b[:, np.newaxis]  # (M, N)
    
    # Derivative of loss w.r.t. out: d/d(out) sum((Y - out)^2 * mask) = -2 * (Y - out) * mask = 2 * (out - Y) * mask
    # To match standard loss scaling without the factor of 2, we define E = (out - Y) * mask
    E = (out - Y) * mask  # (M, N)
    
    # Gradients w.r.t. b and W (scaled by 2 to match sum of squared errors gradient)
    grad_b = 2.0 * np.sum(E, axis=1)  # (M,)
    grad_W = 2.0 * np.dot(h, E.T) + lam * W  # (K, M)
    
    # Backpropagate through decoder to hidden layer
    d_h = np.dot(W, 2.0 * E)  # (K, N)
    
    dh_dz = h * (1.0 - h)  # (K, N)
    d_z = d_h * dh_dz  # (K, N)
    
    # Gradients w.r.t. mu and V
    grad_mu = np.sum(d_z, axis=1)  # (K,)
    grad_V = np.dot(d_z, Y.T) + lam * V  # (K, M)
    
    return {
        "V": grad_V,
        "mu": grad_mu,
        "W": grad_W,
        "b": grad_b
    }

def autorec_n_params(M, K):
    """
    Computes total parameter count: 2MK + M + K.
    """
    return 2 * M * K + M + K