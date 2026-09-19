import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    """
    Computes linear regression coefficients using the Normal Equation.
    
    Parameters:
    -----------
    X : list of list of floats
        Feature matrix (N samples, D features).
    y : list of floats
        Target vector (N samples).
        
    Returns:
    --------
    theta : list of floats
        Coefficients rounded to 4 decimal places.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    # Calculate theta = (X^T * X)^(-1) * X^T * y
    X_transpose = X.T
    theta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
    
    # Round to 4 decimal places
    theta_rounded = np.round(theta, 4).tolist()
    
    return theta_rounded

# Example Test Case
if __name__ == "__main__":
    X = [[1, 1], [1, 2], [1, 3]]
    y = [1, 2, 3]
    
    output = linear_regression_normal_equation(X, y)
    print("Output:", output)  # Expected: [0.0, 1.0]