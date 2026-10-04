import numpy as np

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between two arrays.

    Parameters:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values

    Returns:
        float: Mean Absolute Error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return float(np.mean(np.abs(y_true - y_pred)))