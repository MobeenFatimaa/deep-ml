import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    """
    Fits linear regression with Elastic Net regularization using full-batch gradient descent.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
        alpha1: L1 regularization strength
        alpha2: L2 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum number of iterations
        tol: Convergence tolerance for the L1 norm of the weight gradient
        
    Returns:
        tuple: (weights, bias) where weights is a numpy array of shape (n_features,) and bias is a float.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(max_iter):
        # 1. Compute predictions and residual e = Xw + b - y
        e = np.dot(X, w) + b - y
        
        # 2. Compute gradients
        grad_w = (1.0 / n_samples) * np.dot(X.T, e) + alpha1 * np.sign(w) + 2.0 * alpha2 * w
        grad_b = (1.0 / n_samples) * np.sum(e)
        
        # 3. Update parameters
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        
        # 4. Check convergence after update based on L1 norm of grad_w
        if np.sum(np.abs(grad_w)) < tol:
            break
            
    return w, float(b)