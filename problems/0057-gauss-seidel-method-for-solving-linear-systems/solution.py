import numpy as np


def gauss_seidel(A, b, n, x_ini=None):
    """Solve a linear system Ax = b using the Gauss-Seidel method.

    Args:
        A: Square matrix of coefficients (2D array-like).
        b: Right-hand side vector (1D array-like).
        n: Number of iterations.
        x_ini: Optional initial guess for x (1D array-like).

    Returns:
        Approximated solution vector x after n iterations.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    dim = len(b)

    if x_ini is None:
        x = np.zeros(dim, dtype=float)
    else:
        x = np.array(x_ini, dtype=float)

    for _ in range(n):
        x_new = np.copy(x)
        for i in range(dim):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        x = x_new

    return x


# Example verification
if __name__ == "__main__":
    A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)
    b = np.array([4, 7, 3], dtype=float)
    n = 100
    print(gauss_seidel(A, b, n))