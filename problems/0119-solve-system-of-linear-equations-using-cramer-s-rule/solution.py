import numpy as np

def cramers_rule(A, b):
    """
    Solve a system of linear equations Ax = b using Cramer's Rule.
    
    Args:
        A: Square coefficient matrix (list of lists or np.ndarray)
        b: Constant vector (list or np.ndarray)
        
    Returns:
        np.ndarray or int: Solution vector x, or -1 if no unique solution exists.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Compute the determinant of the coefficient matrix A
    det_A = np.linalg.det(A)
    
    # Check if the system has no unique solution (det(A) is nearly zero)
    if np.isclose(det_A, 0.0):
        return -1
        
    n = A.shape[0]
    x = np.zeros(n)
    
    # Apply Cramer's Rule by replacing each column of A with b in turn
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        det_Ai = np.linalg.det(Ai)
        x[i] = det_Ai / det_A
        
    return x