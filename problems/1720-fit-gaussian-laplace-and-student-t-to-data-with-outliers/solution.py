import math
import numpy as np


def robust_fits(x, nu=4.0, iters=200):
    """Fits Gaussian, Laplace, and Student-t models to 1-D data x.

    Returns a dict with keys 'gaussian', 'laplace', 'student',
    each mapped to a tuple (loc, scale, loglik).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    # 1. Gaussian MLE
    mu_g = np.mean(x)
    sigma_g = np.std(x)  # default ddof=0 uses divisor N
    if sigma_g == 0:
        sigma_g = 1e-8
    var_g = sigma_g ** 2
    loglik_gaussian = np.sum(-0.5 * np.log(2 * np.pi * var_g) - ((x - mu_g) ** 2) / (2 * var_g))

    # 2. Laplace MLE
    mu_l = np.median(x)
    b_l = np.mean(np.abs(x - mu_l))
    if b_l == 0:
        b_l = 1e-8
    loglik_laplace = np.sum(-np.log(2 * b_l) - np.abs(x - mu_l) / b_l)

    # 3. Student-t via EM algorithm
    mu_t = np.median(x)
    sigma_t = sigma_g
    if sigma_t == 0:
        sigma_t = 1.0

    for _ in range(iters):
        r = (x - mu_t) / sigma_t
        w = (nu + 1.0) / (nu + r ** 2)
        
        sum_w = np.sum(w)
        mu_t = np.sum(w * x) / sum_w
        
        sigma_sq = np.sum(w * (x - mu_t) ** 2) / N
        sigma_t = np.sqrt(max(sigma_sq, 1e-16))

    # Student-t log-likelihood computation
    term1 = math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0)
    term2 = -0.5 * np.log(nu * np.pi) - np.log(sigma_t)
    z = (x - mu_t) / sigma_t
    term3 = -0.5 * (nu + 1.0) * np.log(1.0 + (z ** 2) / nu)
    loglik_student = np.sum(term1 + term2 + term3)

    return {
        "gaussian": (float(mu_g), float(sigma_g), float(loglik_gaussian)),
        "laplace": (float(mu_l), float(b_l), float(loglik_laplace)),
        "student": (float(mu_t), float(sigma_t), float(loglik_student))
    }