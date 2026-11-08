import numpy as np


def normalized_laplacian(A):
    """Computes L = I - D^(-1/2) A D^(-1/2) for an undirected graph without self-loops."""
    A = np.asarray(A, dtype=float)
    d = np.sum(A, axis=1)
    
    # Handle zero-degree nodes safely
    d_inv_sqrt = np.zeros_like(d)
    nonzero_mask = d > 0
    d_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(d[nonzero_mask])
    
    D_inv_sqrt = np.diag(d_inv_sqrt)
    I = np.eye(A.shape[0])
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def scaled_laplacian(A, lam_max=None):
    """Computes L_tilde = 2 * L / lam_max - I."""
    L = normalized_laplacian(A)
    if lam_max is None:
        eigenvalues = np.linalg.eigvalsh(L)
        lam_max = np.max(eigenvalues) if len(eigenvalues) > 0 else 1.0
        if lam_max == 0:
            lam_max = 1.0
    L_tilde = (2.0 / lam_max) * L - np.eye(L.shape[0])
    return L_tilde


def chebyshev_basis(L_tilde, X, K):
    """Returns the list [T_0(L_tilde)X, ..., T_K(L_tilde)X] using the recurrence relation."""
    X = np.asarray(X, dtype=float)
    if K < 0:
        return []
    
    basis = []
    if K >= 0:
        basis.append(X.copy())
    if K >= 1:
        T1 = L_tilde @ X
        basis.append(T1)
        
    for k in range(1, K):
        Tk = basis[-1]
        Tk_minus_1 = basis[-2]
        Tk_plus_1 = 2.0 * (L_tilde @ Tk) - Tk_minus_1
        basis.append(Tk_plus_1)
        
    return basis


def chebyshev_filter(L_tilde, X, thetas):
    """Computes sum_{k=0}^K theta_k * T_k(L_tilde)X."""
    K = len(thetas) - 1
    if K < 0:
        return np.zeros_like(X)
    
    basis = chebyshev_basis(L_tilde, X, K)
    res = np.zeros_like(basis[0])
    for k, theta in enumerate(thetas):
        res += theta * basis[k]
    return res


def filter_gain(thetas, lam, lam_max=2.0):
    """Computes the scalar/array gain g(lam) = sum_k theta_k T_k(2 * lam / lam_max - 1)."""
    lam = np.asarray(lam, dtype=float)
    x = (2.0 * lam / lam_max) - 1.0
    
    K = len(thetas) - 1
    if K < 0:
        return np.zeros_like(lam)
        
    t_prev2 = np.ones_like(x)
    if K == 0:
        return thetas[0] * t_prev2
        
    t_prev1 = x.copy()
    res = thetas[0] * t_prev2 + thetas[1] * t_prev1
    
    for k in range(2, K + 1):
        t_curr = 2.0 * x * t_prev1 - t_prev2
        res += thetas[k] * t_curr
        t_prev2 = t_prev1
        t_prev1 = t_curr
        
    return res


def exact_spectral_filter(A, X, thetas, lam_max=None):
    """Eigendecomposes L and computes U diag(g(lambda)) U^T X."""
    L = normalized_laplacian(A)
    eigenvalues, U = np.linalg.eigh(L)
    
    if lam_max is None:
        lam_max = np.max(eigenvalues) if len(eigenvalues) > 0 else 1.0
        if lam_max == 0:
            lam_max = 1.0
            
    g_lam = filter_gain(thetas, eigenvalues, lam_max=lam_max)
    return (U * g_lam) @ (U.T @ X)