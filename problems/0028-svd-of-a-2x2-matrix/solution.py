import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    """Compute SVD of a 2x2 matrix without using built-in SVD functions.

    Returns U, s, V such that A = U @ diag(s) @ V.
    """
    AtA = A.T @ A
    a, b = AtA[0, 0], AtA[0, 1]
    c = AtA[1, 1]

    trace = a + c
    det = a * c - b * b
    disc = np.sqrt(max(0.0, trace**2 - 4 * det))

    lam1 = (trace + disc) / 2
    lam2 = (trace - disc) / 2

    s1 = np.sqrt(max(0.0, lam1))
    s2 = np.sqrt(max(0.0, lam2))

    # Sort in descending order of singular values
    if s1 < s2:
        s1, s2 = s2, s1
        lam1, lam2 = lam2, lam1

    s = np.array([s1, s2])

    def get_v(lam):
        if abs(b) > 1e-12:
            x = np.array([lam - c, b])
        else:
            if abs(a - lam) < 1e-12:
                x = np.array([1.0, 0.0])
            else:
                x = np.array([0.0, 1.0])
        norm = np.linalg.norm(x)
        return x / norm if norm > 1e-12 else np.array([1.0, 0.0])

    v1 = get_v(lam1)
    v2 = get_v(lam2)
    V = np.column_stack((v1, v2)).T

    u1 = (A @ v1) / s1 if s1 > 1e-12 else np.array([1.0, 0.0])
    if s2 > 1e-12:
        u2 = (A @ v2) / s2
    else:
        u2 = np.array([-u1[1], u1[0]])

    U = np.column_stack((u1, u2))

    return U, s, V