import numpy as np

def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 score based on true and predicted labels.

    Args:
        y_true (list or np.ndarray): True labels (ground truth).
        y_pred (list or np.ndarray): Predicted labels.

    Returns:
        float: The F1 score rounded to three decimal places.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Compute metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Handle edge case where precision + recall is 0 to avoid division by zero
    if (2 * tp + fp + fn) == 0:
        return 0.0
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(float(f1), 3)