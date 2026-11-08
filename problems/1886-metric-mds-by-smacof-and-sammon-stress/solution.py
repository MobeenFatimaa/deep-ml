import numpy as np

def stress(D, Z):
    """Computes Kruskal's normalized stress (Stress-1)."""
    N = D.shape[0]
    i, j = np.triu_indices(N, k=1)
    diff = Z[:, None, :] - Z[None, :, :]
    d_hat = np.sqrt(np.sum(diff**2, axis=-1))
    
    num = np.sum((D[i, j] - d_hat[i, j])**2)
    den = np.sum(D[i, j]**2)
    return np.sqrt(num / den) if den != 0 else 0.0


def sammon_stress(D, Z):
    """Computes Sammon's stress."""
    N = D.shape[0]
    i, j = np.triu_indices(N, k=1)
    diff = Z[:, None, :] - Z[None, :, :]
    d_hat = np.sqrt(np.sum(diff**2, axis=-1))
    
    d_ij = D[i, j]
    d_hat_ij = d_hat[i, j]
    
    num = np.sum(((d_hat_ij - d_ij)**2) / d_ij)
    den = np.sum(d_ij)
    return num / den if den != 0 else 0.0


def smacof(D, n_iter, Z0):
    """Minimizes raw stress via SMACOF (Guttman transform)."""
    N = D.shape[0]
    i, j = np.triu_indices(N, k=1)
    
    def get_raw_stress(curr_Z):
        diff = curr_Z[:, None, :] - curr_Z[None, :, :]
        d_hat = np.sqrt(np.sum(diff**2, axis=-1))
        return np.sum((D[i, j] - d_hat[i, j])**2)

    Z = np.array(Z0, dtype=float)
    history = [get_raw_stress(Z)]

    for _ in range(n_iter):
        diff = Z[:, None, :] - Z[None, :, :]
        d_hat = np.sqrt(np.sum(diff**2, axis=-1))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            B = -D / d_hat
            B[d_hat == 0] = 0.0
            
        np.fill_diagonal(B, 0.0)
        row_sums = np.sum(B, axis=1)
        np.fill_diagonal(B, -row_sums)
        
        Z = (1.0 / N) * (B @ Z)
        history.append(get_raw_stress(Z))

    return Z, history