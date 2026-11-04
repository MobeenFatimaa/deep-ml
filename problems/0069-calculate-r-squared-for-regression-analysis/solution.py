import numpy as np

def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate Residual Sum of Squares (SS_res)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate Total Sum of Squares (SS_tot)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Compute R-squared
    return 1 - (ss_res / ss_tot)