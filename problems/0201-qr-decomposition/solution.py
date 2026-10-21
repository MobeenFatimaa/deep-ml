import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix represented as list of lists
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    A_arr = np.array(A, dtype=float)
    m, n = A_arr.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A_arr[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A_arr[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] != 0:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v
            
    return Q.tolist(), R.tolist()