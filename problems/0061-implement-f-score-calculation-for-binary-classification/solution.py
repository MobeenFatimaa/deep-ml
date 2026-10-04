import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of recall relative to precision
    :return: F-Score rounded to three decimal places
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate True Positives, False Positives, and False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Avoid division by zero if both precision and recall are zero
    if precision + recall == 0:
        return 0.0
    
    # Calculate F-beta score: (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    
    if denominator == 0:
        return 0.0
    
    score = numerator / denominator
    return round(float(score), 3)