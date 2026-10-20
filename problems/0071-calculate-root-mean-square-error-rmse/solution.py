import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    if y_true.size == 0:
        raise ValueError("Arrays must not be empty.")
        
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_res = np.sqrt(mse)
    return round(float(rmse_res), 3)