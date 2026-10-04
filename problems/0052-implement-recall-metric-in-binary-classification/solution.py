import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Convert inputs to NumPy arrays in case lists are passed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # True Positives: actual is 1 AND predicted is 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Actual Positives: TP + FN (all cases where actual is 1)
    actual_positives = np.sum(y_true == 1)
    
    # Avoid division by zero if there are no actual positive instances
    if actual_positives == 0:
        return 0.0
    
    return float(tp / actual_positives)