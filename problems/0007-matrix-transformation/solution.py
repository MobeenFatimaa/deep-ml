import numpy as np

def transform_matrix(A: list[list[float]], T: list[list[float]], S: list[list[float]]):
    """
    Transforms matrix A using the operation T^(-1) * A * S.
    
    Parameters:
    ----------
    A : list[list[float]]
        The input matrix to transform.
    T : list[list[float]]
        Invertible matrix on the left.
    S : list[list[float]]
        Invertible matrix on the right.
        
    Returns:
    -------
    list[list[float]] or int
        The resulting transformed matrix as nested lists, or -1 if T or S is singular.
    """
    try:
        A_arr = np.array(A, dtype=float)
        T_arr = np.array(T, dtype=float)
        S_arr = np.array(S, dtype=float)

        # Check invertibility by checking determinants
        if np.isclose(np.linalg.det(T_arr), 0) or np.isclose(np.linalg.det(S_arr), 0):
            return -1

        # Compute inverse of T
        T_inv = np.linalg.inv(T_arr)

        # Perform transformation: T^(-1) * A * S
        result = T_inv @ A_arr @ S_arr

        return result.tolist()

    except (np.linalg.LinAlgError, ValueError):
        # Catch singular matrix or incompatible dimension errors
        return -1