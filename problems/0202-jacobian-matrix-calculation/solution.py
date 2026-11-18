import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as list of lists
    """
    x_arr = np.array(x, dtype=float)
    fx = np.array(f(list(x_arr)), dtype=float)
    
    n = len(x_arr)
    m = len(fx)
    
    # Initialize an m x n matrix
    jacobian = np.zeros((m, n))
    
    for j in range(n):
        # Create a perturbed copy of x
        x_perturbed = x_arr.copy()
        x_perturbed[j] += h
        
        # Evaluate function at the perturbed point
        fx_perturbed = np.array(f(list(x_perturbed)), dtype=float)
        
        # Approximate partial derivative using forward difference
        jacobian[:, j] = (fx_perturbed - fx) / h
        
    return jacobian.tolist()