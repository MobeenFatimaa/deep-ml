import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Performs linear regression using batch gradient descent.
    
    Parameters:
    -----------
    X : np.ndarray of shape (m, n)
        Feature matrix (includes a column of ones for the bias/intercept term).
    y : np.ndarray of shape (m,)
        Target values.
    alpha : float
        Learning rate.
    iterations : int
        Number of gradient descent iterations.
        
    Returns:
    --------
    theta : np.ndarray of shape (n,)
        Learned model weights/coefficients.
    """
    m, n = X.shape
    
    # Initialize all weights to zero
    theta = np.zeros(n, dtype=np.float64)
    
    for _ in range(iterations):
        # 1. Compute predictions: h_theta(X) = X @ theta
        predictions = X @ theta
        
        # 2. Compute the error vector: h_theta(X) - y
        errors = predictions - y
        
        # 3. Compute gradient: (1 / m) * X^T @ errors
        gradient = (1 / m) * (X.T @ errors)
        
        # 4. Update parameters: theta = theta - alpha * gradient
        theta -= alpha * gradient
        
    return theta

# Example Test Case
if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([3, 5, 7])
    alpha = 0.1
    iterations = 1000
    
    theta = linear_regression_gradient_descent(X, y, alpha, iterations)
    print("Learned coefficients:", np.round(theta, 4))
    # Output: [1. 2.]