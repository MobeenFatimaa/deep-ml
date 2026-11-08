import numpy as np
from scipy.optimize import linprog

def lad_fit(X, y):
    """
    Fits a linear model using Least Absolute Deviations (LAD) via Linear Programming.
    Solves: minimize sum(e_n) subject to w^T x_n - y_n <= e_n, -(w^T x_n - y_n) <= e_n, e_n >= 0.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = X.shape

    # Decision variables vector v = [w (size D), e (size N)]
    # Objective: minimize sum(e_n) -> c has zeros for w and ones for e
    c = np.zeros(D + N)
    c[D:] = 1.0

    # Constraints:
    # 1) w^T x_n - e_n <= y_n
    # 2) -w^T x_n - e_n <= -y_n
    # 3) e_n >= 0 (handled by bounds)
    
    A_ub = []
    b_ub = []

    for i in range(N):
        # Constraint 1: w^T x_n - e_n <= y_n  =>  w^T x_n - e_n <= y_n
        row1 = np.zeros(D + N)
        row1[:D] = X[i]
        row1[D + i] = -1.0
        A_ub.append(row1)
        b_ub.append(y[i])

        # Constraint 2: -w^T x_n - e_n <= -y_n
        row2 = np.zeros(D + N)
        row2[:D] = -X[i]
        row2[D + i] = -1.0
        A_ub.append(row2)
        b_ub.append(-y[i])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # w is unbounded (None), e_n >= 0 (0 to None)
    bounds = [(None, None)] * D + [(0.0, None)] * N

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    
    if not res.success:
        # Fallback or return zeros if optimization fails
        return np.zeros(D)

    w = res.x[:D]
    return w


def huber_fit(X, y, delta, n_iter):
    """
    Fits a linear model using Huber loss via Iteratively Reweighted Least Squares (IRLS).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = X.shape

    # Start from Ordinary Least Squares solution
    # (X^T X)^(-1) X^T y
    try:
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        w = np.zeros(D)

    for _ in range(n_iter):
        residuals = y - X @ w
        abs_r = np.abs(residuals)
        
        # Compute weights s_n = min(1, delta / |r_n|)
        # with s_n = 1 when r_n = 0
        s = np.ones_like(abs_r)
        nonzero_mask = abs_r > 1e-12
        s[nonzero_mask] = np.minimum(1.0, delta / abs_r[nonzero_mask])

        # Weighted least squares: minimize sum s_n (y_n - w^T x_n)^2
        # Equivalent to solving (X^T S X) w = X^T S y where S = diag(s)
        S = s[:, np.newaxis]
        XT_S = X.T * s  # Shape (D, N)
        XT_S_X = XT_S @ X
        XT_S_y = XT_S @ y

        try:
            w = np.linalg.solve(XT_S_X, XT_S_y)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if matrix is singular
            w, _, _, _ = np.linalg.lstsq(XT_S_X, XT_S_y, rcond=None)

    return w