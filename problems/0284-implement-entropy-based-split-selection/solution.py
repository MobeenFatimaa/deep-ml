import numpy as np

def _calculate_entropy(y: np.ndarray) -> float:
    """Calculate the Shannon entropy (base-2) of a label array."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # Using base-2 logarithm, filtering out zero probabilities to avoid log2(0)
    return -np.sum(probabilities * np.log2(probabilities + 1e-12))


def entropy_split_selection(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Find the best feature and threshold for splitting based on information gain.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
    
    Returns:
        Tuple of (best_feature_index, best_threshold, best_info_gain)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    n_samples, n_features = X.shape
    parent_entropy = _calculate_entropy(y)
    
    best_feature_index = 0
    best_threshold = 0.0
    best_info_gain = -1.0  # Initialize with a low value to capture 0.0 gain splits
    
    found_any_split = False
    
    for feat_idx in range(n_features):
        feature_values = X[:, feat_idx]
        sorted_unique_vals = np.unique(feature_values)
        
        # A feature with only one unique value has no candidate split
        if len(sorted_unique_vals) <= 1:
            continue
            
        # Consider midpoints between consecutive unique values as candidate thresholds
        thresholds = (sorted_unique_vals[:-1] + sorted_unique_vals[1:]) / 2.0
        
        for thresh in thresholds:
            left_mask = feature_values <= thresh
            right_mask = ~left_mask
            
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            
            # Skip invalid splits where one child is empty
            if n_left == 0 or n_right == 0:
                continue
                
            found_any_split = True
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            # Weighted average entropy of child nodes
            child_entropy = (n_left / n_samples) * _calculate_entropy(y_left) + \
                            (n_right / n_samples) * _calculate_entropy(y_right)
                            
            info_gain = parent_entropy - child_entropy
            
            # Check if this split is better (with strict tie-breaking: lowest feature index, then lowest threshold)
            if (best_info_gain == -1.0) or \
               (info_gain > best_info_gain) or \
               (np.isclose(info_gain, best_info_gain) and feat_idx < best_feature_index) or \
               (np.isclose(info_gain, best_info_gain) and feat_idx == best_feature_index and thresh < best_threshold):
                
                best_feature_index = feat_idx
                best_threshold = round(float(thresh), 4)
                best_info_gain = round(float(info_gain), 4)
                
    # If no feature has a valid split (e.g., all features have only one unique value)
    if not found_any_split:
        return (0, 0.0, 0.0)
        
    return (best_feature_index, best_threshold, best_info_gain)