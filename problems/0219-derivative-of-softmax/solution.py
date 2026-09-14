import numpy as np

def softmax_derivative(x: list[float] | np.ndarray) -> list[list[float]]:
    """Computes the Jacobian matrix of the Softmax function for an input vector x."""
    x = np.array(x, dtype=np.float64)
    
    # 1. Compute stable softmax vector s
    c = np.max(x)
    exp_x = np.exp(x - c)
    s = exp_x / np.sum(exp_x)
    
    # 2. Compute Jacobian matrix J: diag(s) - outer(s, s)
    J = np.diag(s) - np.outer(s, s)
    
    # Return as nested list
    return J.tolist()


# Test Case verification
if __name__ == "__main__":
    result = softmax_derivative([1.0, 2.0, 3.0])
    print([[round(v, 4) for v in row] for row in result])
    # Output: [[0.0819, -0.022, -0.0599], [-0.022, 0.1848, -0.1628], [-0.0599, -0.1628, 0.2227]]