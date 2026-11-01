import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """
    Return the (feature_index, threshold) that minimises weighted Gini impurity.
    
    Args:
        X: n x d NumPy array of numeric features.
        y: Length-n NumPy array of 0/1 binary labels.
        
    Returns:
        tuple: (best_feature_index, best_threshold)
    """
    n_samples, n_features = X.shape
    best_impurity = float('inf')
    best_feature = 0
    best_threshold = 0.0

    def gini(labels: np.ndarray) -> float:
        """Calculate Gini impurity for a set of binary labels."""
        if len(labels) == 0:
            return 0.0
        p = np.mean(labels)
        return 2.0 * p * (1.0 - p)

    for feature_idx in range(n_features):
        # Evaluate unique values of the feature as candidate thresholds
        thresholds = np.unique(X[:, feature_idx])
        
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            # Skip splits that result in empty child nodes
            if not np.any(left_mask) or not np.any(right_mask):
                continue
                
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            n_left = len(y_left)
            n_right = len(y_right)
            
            # Compute weighted Gini impurity
            weighted_gini = (n_left / n_samples) * gini(y_left) + (n_right / n_samples) * gini(y_right)
            
            # Update best split (strictly less ensures first-encountered tie-breaking)
            if weighted_gini < best_impurity:
                best_impurity = weighted_gini
                best_feature = feature_idx
                best_threshold = threshold

    return (best_feature, float(best_threshold))