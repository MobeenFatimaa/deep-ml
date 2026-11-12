import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes, normalize=None, round_decimals=4):
    """
    Compute a KxK confusion matrix with optional normalization.

    Args:
        y_true: Iterable of true labels in [0, K-1]
        y_pred: Iterable of predicted labels in [0, K-1]
        num_classes: K, number of classes
        normalize: None | 'true' | 'pred' | 'all'
        round_decimals: decimals to round when normalization is applied

    Returns:
        list[list[int|float]] confusion matrix
    """
    # Initialize the K x K confusion matrix with zeros
    cm = np.zeros((num_classes, num_classes), dtype=float)
    
    # Populate the raw confusion matrix counts: cm[true_label][pred_label]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1.0
            
    if normalize is None:
        # Return raw integer counts as a list of lists
        return cm.astype(int).tolist()
    
    # Apply normalization modes
    if normalize == 'true':
        # Normalize by true labels (row sums)
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero for rows with sum 0
        with np.errstate(all='ignore'):
            cm = np.where(row_sums > 0, cm / row_sums, 0.0)
    elif normalize == 'pred':
        # Normalize by predicted labels (column sums)
        col_sums = cm.sum(axis=0, keepdims=True)
        with np.errstate(all='ignore'):
            cm = np.where(col_sums > 0, cm / col_sums, 0.0)
    elif normalize == 'all':
        # Normalize by total sample count
        total = cm.sum()
        if total > 0:
            cm = cm / total
        else:
            cm = np.zeros_like(cm)
    else:
        raise ValueError(f"Invalid normalization mode: {normalize}")
        
    # Round outputs to specified decimal places and convert to list of lists
    return np.round(cm, decimals=round_decimals).tolist()