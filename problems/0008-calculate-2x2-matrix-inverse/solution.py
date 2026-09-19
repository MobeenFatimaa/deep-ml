def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculates the inverse of a 2x2 matrix.
    
    Parameters:
    -----------
    matrix : list of list of floats
        A 2x2 matrix represented as [[a, b], [c, d]].
        
    Returns:
    --------
    inverse : list of list of floats or None
        The 2x2 inverse matrix, or None if det == 0.
    """
    a, b = matrix[0]
    c, d = matrix[1]
    
    # Calculate determinant
    det = a * d - b * c
    
    # Return None if non-invertible
    if det == 0:
        return None
    
    # Apply standard 2x2 inverse formula
    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ]

# Verification
if __name__ == "__main__":
    print(inverse_2x2([[4, 7], [2, 6]]))
    # Output: [[0.6, -0.7], [-0.2, 0.4]]