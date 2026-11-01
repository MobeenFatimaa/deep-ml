import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    """
    Computes an orthonormal basis for the subspace spanned by a list of vectors using Gram-Schmidt.
    
    Args:
        vectors: List of input vectors.
        tol: Tolerance value to determine linear independence.
        
    Returns:
        list[np.ndarray]: List of orthonormal vectors spanning the same subspace.
    """
    basis = []
    for v in vectors:
        v = np.array(v, dtype=float)
        
        # Subtract the projection of v onto each existing basis vector
        for u in basis:
            v -= np.dot(v, u) * u
            
        # Check if the resulting vector is significantly non-zero (linearly independent)
        norm = np.linalg.norm(v)
        if norm > tol:
            basis.append(v / norm)
            
    return basis