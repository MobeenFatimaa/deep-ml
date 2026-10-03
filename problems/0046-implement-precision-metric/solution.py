import numpy as np

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    # Handle division by zero edge case when no positive predictions are made
    if tp + fp == 0:
        return 0.0
        
    return float(tp / (tp + fp))