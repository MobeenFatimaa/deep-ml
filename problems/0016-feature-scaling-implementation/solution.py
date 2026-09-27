import numpy as np

def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Column-wise mean and population standard deviation
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Column-wise min and max
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    
    # Apply formulas and round to 4 decimal places
    standardized_data = np.round((data - mean) / std, 4)
    normalized_data = np.round((data - min_val) / (max_val - min_val), 4)
    
    return standardized_data, normalized_data