import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    num_equations = len(b)
    # Initialize solution vector x to zeros
    x = np.zeros(num_equations, dtype=float)
    
    # Extract diagonal and off-diagonal components
    # A = D + R  =>  D*x^(k+1) = b - R*x^(k)
    D = np.diag(A)
    R = A - np.diag(D)
    
    # Perform exactly n iterations without intermediate rounding
    for _ in range(n):
        x = (b - np.dot(R, x)) / D
        
    # Round final result to 4 decimal places
    return list(np.round(x, 4))