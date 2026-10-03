import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    
    Args:
        w: Input array
        threshold: Threshold value λ
    
    Returns:
        Soft-thresholded array where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias)
    
    Note: The bias term is NOT regularized.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    
    for _ in range(max_iter):
        # 1. Predictions
        predictions = X @ weights + bias
        error = predictions - y
        
        # 2. Compute gradients of smooth MSE loss: J_mse = (1 / (2 * n_samples)) * sum(error^2)
        dw = (1 / n_samples) * (X.T @ error)
        db = (1 / n_samples) * np.sum(error)
        
        # 3. Save old weights to check for convergence
        weights_old = weights.copy()
        
        # 4. Gradient descent step on weights and bias (bias is NOT regularized)
        w_temp = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
        # 5. Proximal step: soft-thresholding on weights with threshold = learning_rate * alpha
        weights = soft_threshold(w_temp, learning_rate * alpha)
        
        # 6. Check convergence criteria based on change in weights
        if np.linalg.norm(weights - weights_old, ord=np.inf) < tol:
            break
            
    return weights, bias