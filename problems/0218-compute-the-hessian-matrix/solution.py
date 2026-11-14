from typing import Callable
import copy

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point using finite differences.
    
    Args:
        f: A scalar function that takes a list of floats and returns a float
        point: The point at which to compute the Hessian (list of coordinates)
        h: Step size for finite differences (default: 1e-5)
        
    Returns:
        The Hessian matrix as a list of lists (n x n where n = len(point))
    """
    n = len(point)
    hessian = [[0.0] * n for _ in range(n)]
    
    # Precompute base point value for diagonal terms if needed, 
    # though standard formulas evaluate the perturbed points directly.
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Pure second derivative: d^2f / dx_i^2
                p_plus = list(point)
                p_minus = list(point)
                
                p_plus[i] += h
                p_minus[i] -= h
                
                val_center = f(point)
                val_plus = f(p_plus)
                val_minus = f(p_minus)
                
                deriv = (val_plus - 2.0 * val_center + val_minus) / (h ** 2)
                hessian[i][j] = deriv
                
            else:
                # Mixed partial derivative: d^2f / dx_i dx_j
                p_pp = list(point)
                p_pm = list(point)
                p_mp = list(point)
                p_mm = list(point)
                
                p_pp[i] += h; p_pp[j] += h
                p_pm[i] += h; p_pm[j] -= h
                p_mp[i] -= h; p_mp[j] += h
                p_mm[i] -= h; p_mm[j] -= h
                
                deriv = (f(p_pp) - f(p_pm) - f(p_mp) + f(p_mm)) / (4.0 * h ** 2)
                
                hessian[i][j] = deriv
                hessian[j][i] = deriv  # Hessian is symmetric by Clairaut's theorem
                
    return hessian