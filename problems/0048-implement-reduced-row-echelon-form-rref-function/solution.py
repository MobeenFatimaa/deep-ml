import numpy as np


def rref(matrix):
    """Computes the Reduced Row Echelon Form (RREF) of a matrix using Gauss-Jordan elimination.

    Args:
        matrix: A 2D numpy array or list of numerical values.

    Returns:
        A 2D numpy array of floats representing the RREF of the input matrix.
    """
    # Convert matrix to float type to prevent integer division truncation
    A = np.array(matrix, dtype=float).copy()
    rows, cols = A.shape

    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        # Step 1: Find the absolute maximum entry in current column (at or below pivot_row) for numerical stability
        max_row = pivot_row + np.argmax(np.abs(A[pivot_row:, col]))

        # Check if the pivot element is effectively zero
        if np.isclose(A[max_row, col], 0.0):
            continue

        # Step 2: Swap the current pivot row with the row containing the largest pivot element
        if max_row != pivot_row:
            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]

        # Step 3: Scale the pivot row to make the leading entry equal to 1
        pivot_val = A[pivot_row, col]
        A[pivot_row] = A[pivot_row] / pivot_val

        # Step 4: Eliminate all other entries in the current column (both above and below the pivot)
        for r in range(rows):
            if r != pivot_row:
                factor = A[r, col]
                A[r] -= factor * A[pivot_row]

        pivot_row += 1

    # Clean up negative zeros (e.g., -0.0 -> 0.0)
    A[np.isclose(A, 0.0)] = 0.0

    return A