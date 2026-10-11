import math
import numpy as np
import scipy.special


def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5):
    dist = np.linalg.norm(x - x_prime)
    if dist == 0:
        return 1.0
    if nu == 0.5:
        return np.exp(-dist / length_scale)
    elif nu == 1.5:
        sqrt3_dist = np.sqrt(3) * dist / length_scale
        return (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
    elif nu == 2.5:
        sqrt5_dist = np.sqrt(5) * dist / length_scale
        return (
            1 + sqrt5_dist + (5 * dist**2) / (3 * length_scale**2)
        ) * np.exp(-sqrt5_dist)
    else:
        sqrt_2nu_dist = np.sqrt(2 * nu) * dist / length_scale
        return (
            (2.0 ** (1.0 - nu) / scipy.special.gamma(nu))
            * (sqrt_2nu_dist**nu)
            * scipy.special.kv(nu, sqrt_2nu_dist)
        )


def rbf_kernel(x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0):
    dist_sq = np.sum((x - x_prime) ** 2)
    return (sigma**2) * np.exp(-0.5 * dist_sq / (length_scale**2))


def periodic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, period=1.0
):
    dist = np.linalg.norm(x - x_prime)
    return (sigma**2) * np.exp(
        -2.0 * (np.sin(np.pi * dist / period) ** 2) / (length_scale**2)
    )


def linear_kernel(x: np.ndarray, x_prime: np.ndarray, sigma_b=1.0, sigma_v=1.0):
    return (sigma_b**2) + (sigma_v**2) * np.dot(x, x_prime)


def rational_quadratic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, alpha=1.0
):
    dist_sq = np.sum((x - x_prime) ** 2)
    return (sigma**2) * (
        (1.0 + dist_sq / (2.0 * alpha * (length_scale**2))) ** (-alpha)
    )


# --- BASE CLASS -------------------------------------------------------------


class _GaussianProcessBase:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None):
        self.kernel_name = kernel
        self.noise = noise
        self.kernel_params = kernel_params or {}

        if kernel == "rbf":
            self.kernel_func = rbf_kernel
        elif kernel == "matern":
            self.kernel_func = matern_kernel
        elif kernel == "periodic":
            self.kernel_func = periodic_kernel
        elif kernel == "linear":
            self.kernel_func = linear_kernel
        elif kernel == "rational_quadratic":
            self.kernel_func = rational_quadratic_kernel
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    def _select_kernel(self, x1, x2):
        """Selects and computes the kernel value for two single data points."""
        return self.kernel_func(x1, x2, **self.kernel_params)

    def _compute_covariance(self, X1, X2):
        """Computes the covariance matrix between two sets of points."""
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._select_kernel(X1[i], X2[j])
        return K


# --- REGRESSION MODEL -------------------------------------------------------
class GaussianProcessRegression(_GaussianProcessBase):
    def fit(self, X, y):
        self.X_train = np.atleast_2d(X)
        self.y_train = np.atleast_1d(y)

        K = self._compute_covariance(self.X_train, self.X_train)
        self.K_y = K + self.noise * np.eye(len(self.X_train))

        try:
            self.L = np.linalg.cholesky(self.K_y)
            self.alpha = np.linalg.solve(
                self.L.T, np.linalg.solve(self.L, self.y_train)
            )
        except np.linalg.LinAlgError:
            self.L = None
            self.alpha = np.linalg.pinv(self.K_y) @ self.y_train

    def predict(self, X_test, return_std=False):
        X_test = np.atleast_2d(X_test)
        K_s = self._compute_covariance(self.X_train, X_test)
        mu = K_s.T @ self.alpha

        if return_std:
            K_ss = self._compute_covariance(X_test, X_test)
            if self.L is not None:
                v = np.linalg.solve(self.L, K_s)
                cov = K_ss - v.T @ v
            else:
                cov = K_ss - K_s.T @ np.linalg.pinv(self.K_y) @ K_s
            std = np.sqrt(np.clip(np.diagonal(cov), 1e-12, None))
            return mu, std
        return mu

    def log_marginal_likelihood(self):
        if self.L is not None:
            # - 0.5 * y^T * alpha - sum(log(diag(L))) - (N / 2) * log(2 * pi)
            term1 = -0.5 * np.dot(self.y_train, self.alpha)
            term2 = -np.sum(np.log(np.diagonal(self.L)))
            term3 = -0.5 * len(self.X_train) * np.log(2 * np.pi)
            return term1 + term2 + term3
        else:
            sign, logdet = np.linalg.slogdet(self.K_y)
            return -0.5 * (
                np.dot(self.y_train, self.alpha)
                + logdet
                + len(self.X_train) * np.log(2 * np.pi)
            )

    def optimize_hyperparameters(self):
        # Placeholder for hyperparameter optimization (e.g. via gradient descent on LML)
        pass