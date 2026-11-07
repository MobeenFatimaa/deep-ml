import numpy as np


def fit_softmax_mm(X, y, C, n_iter):
    """Fit multinomial logistic regression using Bohning's Bound (MM) algorithm.

    Args:
        X (np.ndarray): (N, D) feature matrix.
        y (np.ndarray): (N,) integer class labels (0 to C-1).
        C (int): number of classes.
        n_iter (int): number of MM iterations.

    Returns:
        tuple: (W of shape (C, D) with last row zero, history of log-likelihoods list)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    N, D = X.shape

    # Initialize W of shape (C, D) with zeros (last row is implicitly 0)
    W = np.zeros((C, D), dtype=float)

    # Precompute S = X^T X and its inverse
    S = X.T @ X
    S_inv = np.linalg.inv(S)

    # Precompute M^{-1} for the C-1 free classes
    C_free = C - 1
    M_inv = np.eye(C_free) + np.ones((C_free, C_free))

    # B^{-1} = -2 * (M^{-1} \otimes S^{-1})
    B_inv = -2.0 * np.kron(M_inv, S_inv)

    history = []

    # One-hot encoding of y for N samples and C classes
    Y_onehot = np.zeros((N, C), dtype=float)
    Y_onehot[np.arange(N), y] = 1.0

    for _ in range(n_iter):
        # Linear activations: eta shape (N, C)
        eta = X @ W.T

        # Softmax probabilities with stability shift
        eta_max = np.max(eta, axis=1, keepdims=True)
        exp_eta = np.exp(eta - eta_max)
        sum_exp_eta = np.sum(exp_eta, axis=1, keepdims=True)
        P = exp_eta / sum_exp_eta

        # Compute log-likelihood BEFORE update (or after? The prompt says history of l(W) values after each update, but let's check standard practice or test case. Wait, in my previous code for C=2, computing after update matched the expected history!)
        # Wait, let's check whether history should be before or after update.
        # For C=2, computing after update produced [-3.21294, -3.18779, -3.18329, -3.18229, -3.18205], which matched the expected output!

        # Gradient G of shape (C, D): G = (Y_onehot - P)^T X
        G_full = (Y_onehot - P).T @ X

        # Free rows gradient (first C-1 rows)
        G_free = G_full[:C_free, :]
        vec_G = G_free.ravel()

        # Update vec(W_free) using vec(W_free) -= B_inv @ vec_G
        vec_W_free = W[:C_free, :].ravel()
        vec_W_free -= B_inv @ vec_G

        # Reshape back and update W
        W[:C_free, :] = vec_W_free.reshape(C_free, D)
        # Ensure last row remains zero
        W[C - 1, :] = 0.0

        # Compute log-likelihood after update
        eta_new = X @ W.T
        eta_new_max = np.max(eta_new, axis=1, keepdims=True)
        log_sum_exp = eta_new_max.ravel() + np.log(np.sum(np.exp(eta_new - eta_new_max), axis=1))
        true_class_etas = eta_new[np.arange(N), y]
        log_lik = np.sum(true_class_etas - log_sum_exp)
        
        history.append(float(log_lik))

    return W, history