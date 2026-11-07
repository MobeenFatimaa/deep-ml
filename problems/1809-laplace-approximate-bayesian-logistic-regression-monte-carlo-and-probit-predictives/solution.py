import numpy as np


def laplace_logreg(X, y, prior_var, n_iter=50):
    """Approximate the posterior of a logistic regression model with a Gaussian using Laplace approximation.

    Args:
        X (np.ndarray): (n, D) design matrix.
        y (np.ndarray): (n,) binary labels (0 or 1).
        prior_var (float): variance sigma^2 for the zero-mean Gaussian prior on weights.
        n_iter (int): maximum number of Newton-Raphson iterations.

    Returns:
        tuple: (w_map, Sigma) where w_map is (D,) and Sigma is (D, D).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, D = X.shape

    w = np.zeros(D, dtype=float)
    inv_prior_var = 1.0 / prior_var

    for _ in range(n_iter):
        z_lin = X @ w
        # Stable sigmoid computation
        mu = np.where(z_lin >= 0, 1.0 / (1.0 + np.exp(-z_lin)), np.exp(z_lin) / (1.0 + np.exp(z_lin)))

        # Gradient of log posterior = X^T (y - mu) - w / prior_var
        grad = X.T @ (y - mu) - w * inv_prior_var

        # Hessian of negative log posterior = X^T S X + (1 / prior_var) * I
        # where S = diag(mu * (1 - mu))
        s_diag = mu * (1.0 - mu)
        H = X.T @ (s_diag[:, None] * X) + np.eye(D) * inv_prior_var

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break

        w += step

        if np.linalg.norm(step) < 1e-12:
            break

    # Final covariance matrix Sigma
    z_lin = X @ w
    mu = np.where(z_lin >= 0, 1.0 / (1.0 + np.exp(-z_lin)), np.exp(z_lin) / (1.0 + np.exp(z_lin)))
    s_diag = mu * (1.0 - mu)
    H = X.T @ (s_diag[:, None] * X) + np.eye(D) * inv_prior_var

    try:
        Sigma = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Sigma = np.linalg.pinv(H)

    return w, Sigma


def predictive(X_test, w_map, Sigma, method, n_samples=1000, seed=0):
    """Compute predictive probabilities p(y=1 | x) using plugin, Monte Carlo, or probit approximation."""
    X_test = np.asarray(X_test, dtype=float)
    w_map = np.asarray(w_map, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    if method == "plugin":
        z = X_test @ w_map
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    elif method == "mc":
        rng = np.random.default_rng(seed)
        # Cholesky factor of Sigma
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            # Fallback if numerically non-PD due to precision issues
            vals, vecs = np.linalg.eigh(Sigma)
            vals = np.maximum(vals, 0.0)
            L = vecs @ np.diag(np.sqrt(vals))

        D = len(w_map)
        zs = rng.standard_normal((n_samples, D))
        # ws shape: (n_samples, D)
        ws = w_map + zs @ L.T

        # X_test @ ws.T shape: (n_test, n_samples)
        z_matrix = X_test @ ws.T
        probs_matrix = np.where(z_matrix >= 0, 1.0 / (1.0 + np.exp(-z_matrix)), np.exp(z_matrix) / (1.0 + np.exp(z_matrix)))
        return np.mean(probs_matrix, axis=1)

    elif method == "probit":
        m = X_test @ w_map
        # v = diag(X_test @ Sigma @ X_test^T) computed efficiently row-wise
        v = np.sum((X_test @ Sigma) * X_test, axis=1)
        kappa = 1.0 / np.sqrt(1.0 + (np.pi / 8.0) * v)
        z = kappa * m
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    else:
        raise ValueError(f"Unknown prediction method: {method}")