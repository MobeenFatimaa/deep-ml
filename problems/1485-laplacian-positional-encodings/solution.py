import numpy as np


def laplacian(A, normalized=True):
    """Computes the graph Laplacian (unnormalized or symmetric normalized)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    degrees = np.sum(A, axis=1)
    
    if normalized:
        inv_sqrt_deg = np.zeros_like(degrees)
        mask = degrees > 0
        inv_sqrt_deg[mask] = 1.0 / np.sqrt(degrees[mask])
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        D = np.diag(degrees)
        L = D - A
    return L


def canonical_sign(U):
    """Flips columns of U so their first non-negligible entry (> 1e-8) is positive."""
    U_adj = U.copy()
    for j in range(U_adj.shape[1]):
        col = U_adj[:, j]
        idx = np.where(np.abs(col) > 1e-8)[0]
        if len(idx) > 0:
            first_idx = idx[0]
            if col[first_idx] < 0:
                U_adj[:, j] = -col
    return U_adj


def laplacian_pe(A, k, normalized=True):
    """Computes Laplacian positional encodings by eigendecomposition."""
    L = laplacian(A, normalized=normalized)
    evals, evecs = np.linalg.eigh(L)
    
    # Discard eigenvalues below 1e-8 (zero eigenvalues / connected components)
    mask = evals >= 1e-8
    valid_evals = evals[mask]
    valid_evecs = evecs[:, mask]
    
    # Take the next k eigenvectors/eigenvalues
    selected_evals = valid_evals[:k]
    selected_evecs = valid_evecs[:, :k]
    
    # Apply canonical sign convention
    selected_evecs = canonical_sign(selected_evecs)
    
    return selected_evals, selected_evecs


def pe_distances(P):
    """Computes the (n, n) matrix of Euclidean distances between positional encodings."""
    P = np.asarray(P, dtype=float)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)


def random_sign_flip(P, signs):
    """Applies random sign augmentation per column during training."""
    P_aug = P.copy()
    signs = np.asarray(signs)
    return P_aug * signs[np.newaxis, :]