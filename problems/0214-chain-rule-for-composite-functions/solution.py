import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    derivative_product = 1.0
    u = x
    
    for func in reversed(functions):
        if func == 'square':
            derivative_product *= 2 * u
            u = u ** 2
        elif func == 'sin':
            derivative_product *= np.cos(u)
            u = np.sin(u)
        elif func == 'exp':
            derivative_product *= np.exp(u)
            u = np.exp(u)
        elif func == 'log':
            derivative_product *= 1.0 / u
            u = np.log(u)
        else:
            raise ValueError(f"Unknown function: {func}")
            
    return float(derivative_product)