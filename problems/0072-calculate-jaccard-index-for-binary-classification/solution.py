import numpy as np

def jaccard_index(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Intersection: where both y_true and y_pred are 1
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    
    # Union: where either y_true or y_pred is 1
    union = np.sum((y_true == 1) | (y_pred == 1))
    
    # Handle edge case where union is empty (all zeros)
    if union == 0:
        return 0.0
        
    result = intersection / union
    return round(float(result), 3)