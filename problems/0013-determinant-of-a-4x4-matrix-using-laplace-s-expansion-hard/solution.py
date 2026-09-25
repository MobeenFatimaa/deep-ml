def determinant(matrix: list[list[int | float]]) -> float:
    """
    Recursively computes the determinant of an n x n matrix using Laplace's Expansion.
    Supports 4x4, 3x3, 2x2, and 1x1 matrices.
    """
    n = len(matrix)
    
    # Base cases
    if n == 1:
        return float(matrix[0][0])
    
    if n == 2:
        return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
    
    det = 0.0
    
    # Expand along the first row (row 0)
    for j in range(n):
        # Create submatrix by excluding row 0 and column j
        submatrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        
        # Cofactor sign (-1)^(0 + j) -> alternating signs (+, -, +, -)
        sign = 1.0 if j % 2 == 0 else -1.0
        
        # Recursive Laplace expansion
        det += sign * matrix[0][j] * determinant(submatrix)
        
    return det


def determinant_4x4(matrix: list[list[int | float]]) -> float:
    """Calculates the determinant of a 4x4 matrix."""
    return determinant(matrix)