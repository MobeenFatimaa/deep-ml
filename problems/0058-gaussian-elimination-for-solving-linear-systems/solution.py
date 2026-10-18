import numpy as np


def gaussian_elimination(A, b):
    """Solve a linear system Ax = b using Gaussian Elimination with partial pivoting.

    Args:
        A: Square matrix of coefficients (2D array-like).
        b: Right-hand side vector (1D array-like).

    Returns:
        Solution vector x as a NumPy array.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Create an augmented matrix [A | b]
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find the row with the maximum absolute value in the current column
        max_row = k + np.argmax(np.abs(Ab[k:, k]))

        # Swap the current row with the max_row if needed
        if max_row != k:
            Ab[[k, max_row]] = Ab[[max_row, k]]

        # Eliminate entries below the pivot
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]

    # Backward substitution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1 : n], x[i + 1 : n])) / Ab[i, i]

    return x


# Example verification
if __name__ == "__main__":
    A = np.array([[2, 8, 4], [2, 5, 1], [4, 10, -1]], dtype=float)
    b = np.array([2, 5, 1], dtype=float)

    print(gaussian_elimination(A, b))