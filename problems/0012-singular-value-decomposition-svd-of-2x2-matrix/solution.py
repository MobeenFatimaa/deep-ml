import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ np.diag(S) @ Vt
    """
    A = np.array(A, dtype=float)
    
    # Step 1: Compute M = A^T @ A
    M = A.T @ A
    a, b = M[0, 0], M[0, 1]
    c = M[1, 1]
    
    # Step 2: Compute rotation angle theta to diagonalize M
    # Jacobi angle formula: tan(2 * theta) = 2 * b / (a - c)
    if np.abs(b) < 1e-15:
        theta = 0.0
    else:
        theta = 0.5 * np.arctan2(2 * b, a - c)
        
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Right singular vector matrix V
    V = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    
    # Step 3: Compute diagonal matrix D = V^T @ M @ V
    D = V.T @ M @ V
    
    # Extract eigenvalues (diagonal elements) and singular values
    # Ensure values are non-negative due to float precision
    eigs = np.maximum(0.0, np.diag(D))
    
    # Sort eigenvalues/singular values in descending order
    if eigs[0] < eigs[1]:
        eigs = eigs[::-1]
        V = V[:, ::-1]
        
    s1 = np.sqrt(eigs[0])
    s2 = np.sqrt(eigs[1])
    S = np.array([s1, s2])
    
    # Step 4: Compute U = A @ V @ Sigma^(-1)
    AV = A @ V
    U = np.zeros((2, 2))
    
    # Normalize columns of AV to get orthogonal columns of U
    if s1 > 1e-15:
        U[:, 0] = AV[:, 0] / s1
    else:
        U[:, 0] = np.array([1.0, 0.0])
        
    if s2 > 1e-15:
        U[:, 1] = AV[:, 1] / s2
    else:
        # Construct an orthogonal column if s2 is close to zero
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])
        
    # Transpose of V
    Vt = V.T
    
    return U, S, Vt