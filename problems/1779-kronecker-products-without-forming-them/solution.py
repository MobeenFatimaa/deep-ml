import numpy as np


def kron_matvec(A, B, C):
    """Computes (A \otimes B) vec(C) efficiently as vec(B C A^T) without forming the Kronecker product."""
    # C has shape (q, n) where B is (p, q) and A is (m, n)
    # BCA^T has shape (p, m)
    # vec() stacks columns in column-major order (order='F')
    return (B @ C @ A.T).reshape(-1, order='F')


def kron_solve(A, B, y):
    """Solves (A \otimes B) x = y for x using (A \otimes B)^{-1} = A^{-1} \otimes B^{-1},
    leveraging factor solves without forming the big matrix or inverting directly.
    """
    n = A.shape[0]
    q = B.shape[0]
    
    # Reshape y of length nq into a (q, n) matrix Y (column-major layout)
    Y = y.reshape((q, n), order='F')
    
    # Using (A \otimes B)^{-1} vec(C) = vec(B^{-1} C A^{-T})
    # Here x = vec(X), so Y = B X A^T => B X = Y A^{-T} => X = B^{-1} Y A^{-T}
    # Equivalently, solve A^T X^T = Y^T B^{-T} or use np.linalg.solve on factors.
    # Let's solve A^T Z^T = Y^T => Z A^T = Y => Z = Y (A^T)^{-1} -> Z = (A^{-1} Y^T)^T ... wait:
    # Y = B X A^T => B^{-1} Y = X A^T => (B^{-1} Y) A^{-T} = X
    # Let's solve B Z = Y for Z, then solve X A^T = Z for X (or X^T A = Z^T).
    
    Z = np.linalg.solve(B, Y)
    # X A^T = Z  =>  A X^T = Z^T  =>  X^T = solve(A, Z^T)  =>  X = (solve(A, Z^T))^T
    X_T = np.linalg.solve(A, Z.T)
    X = X_T.T
    
    return X.reshape(-1, order='F')


def bow_logits(S, W, V):
    """Computes bag-of-words logits L_{nc} = \sum_{t,k,d} S_{ntk} W_{kd} V_{dc}
    using a single np.einsum call.
    """
    # S: (N, T, K)
    # W: (K, D)
    # V: (D, C)
    return np.einsum('ntk,kd,dc->nc', S, W, V)