import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    X = np.asarray(X)
    weights = np.asarray(weights)
    
    # Step 1: Compute the linear combination z = Xw + b
    z = np.dot(X, weights) + bias
    
    # Step 2: Compute the sigmoid probabilities
    probabilities = 1 / (1 + np.exp(-z))
    
    # Step 3: Apply threshold at 0.5 to get binary predictions (0 or 1)
    predictions = (probabilities >= 0.5).astype(int)
    
    return predictions