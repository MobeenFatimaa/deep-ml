import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    X = np.array(X)
    y = np.array(y, dtype=float)
    current_predictions = np.array(current_predictions, dtype=float)
    
    # 1. Compute residuals (negative gradient for MSE loss: y - preds)
    residuals = y - current_predictions
    
    n_samples, n_features = X.shape
    
    best_mse = float('inf')
    best_stump_preds = np.full(n_samples, np.mean(residuals))
    
    # 2. Fit a decision stump (depth 1 tree) minimizing MSE
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        sorted_indices = np.argsort(feature_values)
        sorted_X = feature_values[sorted_indices]
        sorted_res = residuals[sorted_indices]
        
        # Try all possible thresholds (midpoints between consecutive unique values)
        for i in range(1, n_samples):
            if sorted_X[i] == sorted_X[i-1]:
                continue
            
            threshold = (sorted_X[i] + sorted_X[i-1]) / 2.0
            
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            if not np.any(left_mask) or not np.any(right_mask):
                continue
                
            left_mean = np.mean(residuals[left_mask])
            right_mean = np.mean(residuals[right_mask])
            
            stump_preds = np.empty(n_samples)
            stump_preds[left_mask] = left_mean
            stump_preds[right_mask] = right_mean
            
            # Compute MSE
            mse = np.mean((residuals - stump_preds) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_stump_preds = stump_preds
                
    # 3. Update predictions using the stump's predictions scaled by the learning rate
    updated_predictions = current_predictions + learning_rate * best_stump_preds
    
    return [round(float(val), 4) for val in updated_predictions]