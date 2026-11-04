import numpy as np

def dice_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Intersection: where both y_true and y_pred are 1 (True Positives)
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    
    # Total count of ones in each array
    sum_true = np.sum(y_true == 1)
    sum_pred = np.sum(y_pred == 1)
    
    # Denominator is the sum of the cardinalities of both arrays
    denominator = sum_true + sum_pred
    
    # Handle edge case where either or both arrays have no ones
    if denominator == 0:
        return 0.0
        
    score = (2.0 * intersection) / denominator
    return round(float(score), 3)