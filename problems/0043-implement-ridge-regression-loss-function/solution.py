import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    """
    Returns the Ridge regression loss.
    
    Args:
        X: Feature matrix of shape (n, d)
        w: Coefficient vector of length d
        y_true: Target vector of length n
        alpha: Regularization strength
        
    Returns:
        The computed Ridge loss as a float.
    """
    n = X.shape[0]
    y_pred = X @ w
    mse = np.mean((y_true - y_pred) ** 2)
    reg = alpha * np.sum(w ** 2)
    return float(mse + reg)