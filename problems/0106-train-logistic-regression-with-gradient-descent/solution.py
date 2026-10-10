import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    n_samples, n_features = X.shape
    
    # Initialize weights and bias to zeros
    weights = np.zeros(n_features)
    bias = 0.0
    
    losses = []
    
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    for _ in range(iterations):
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)
        
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Use sum instead of mean for total loss matching the expected scale
        loss = -np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        losses.append(round(float(loss), 4))
        
        # Gradients scaled accordingly
        dw = np.dot(X.T, (y_pred - y))
        db = np.sum(y_pred - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
    coefficients = [round(float(bias), 4)] + [round(float(w), 4) for w in weights]
    
    return coefficients, losses