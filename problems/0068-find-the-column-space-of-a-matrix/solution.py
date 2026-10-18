import numpy as np


def matrix_image(A):
    """Calculate the column space (image) of a given matrix A.

    Args:
        A: 2D array-like representing the matrix.

    Returns:
        A NumPy array containing the basis vectors (independent columns extracted
        from A) that span the column space of A.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    if m == 0 or n == 0:
        return np.zeros((m, 0))

    M = A.copy()
    row = 0
    pivot_cols = []

    for col in range(n):
        if row >= m:
            break

        # Find pivot in the current column from the current row downwards
        max_row = row + np.argmax(np.abs(M[row:, col]))
        if np.isclose(M[max_row, col], 0.0, atol=1e-12):
            continue

        # Swap rows to put the pivot in place
        if max_row != row:
            M[[row, max_row]] = M[[max_row, row]]

        # Normalize pivot row
        pivot_val = M[row, col]
        M[row] = M[row] / pivot_val

        # Eliminate entries in other rows for this column
        for r in range(m):
            if r != row:
                factor = M[r, col]
                if not np.isclose(factor, 0.0, atol=1e-12):
                    M[r] -= factor * M[row]

        pivot_cols.append(col)
        row += 1

    if not pivot_cols:
        return np.zeros((m, 0))

    return A[:, pivot_cols]


# Example verification
if __name__ == "__main__":
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(matrix_image(matrix))