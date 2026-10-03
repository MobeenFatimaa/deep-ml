import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Create a copy of weights to avoid in-place modification of caller arguments
    w = np.array(weights, dtype=float).copy()
    m = X.shape[0]

    for epoch in range(n_epochs):
        if method == 'batch':
            # Use all samples to calculate gradient
            predictions = X @ w
            error = predictions - y
            gradient = (2 / m) * (X.T @ error)
            w -= learning_rate * gradient

        elif method == 'stochastic':
            # Process samples sequentially one by one
            for i in range(m):
                X_i = X[i:i+1]  # Keep 2D shape (1, n)
                y_i = y[i:i+1]
                
                prediction = X_i @ w
                error = prediction - y_i
                gradient = 2 * (X_i.T @ error)  # k = 1 sample
                w -= learning_rate * gradient

        elif method == 'mini_batch':
            # Process consecutive non-overlapping batches
            for i in range(0, m, batch_size):
                X_b = X[i:i + batch_size]
                y_b = y[i:i + batch_size]
                k = X_b.shape[0]  # Handles potential smaller last batch
                
                predictions = X_b @ w
                error = predictions - y_b
                gradient = (2 / k) * (X_b.T @ error)
                w -= learning_rate * gradient

        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'batch', 'stochastic', or 'mini_batch'.")

    return w