import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # Create a copy to avoid mutating the input array in-place
    result = np.array(data, dtype=float, copy=True)
    
    # Process each column independently
    for col_idx in range(result.shape[1]):
        col = result[:, col_idx]
        
        # Extract valid (non-NaN) values
        valid_mask = ~np.isnan(col)
        valid_vals = col[valid_mask]
        
        # Skip column if all values are missing
        if valid_vals.size == 0:
            continue
            
        # Compute replacement statistic
        if strategy == 'mean':
            fill_value = np.mean(valid_vals)
            
        elif strategy == 'median':
            fill_value = np.median(valid_vals)
            
        elif strategy == 'mode':
            # Count frequency for each unique non-NaN value
            unique_vals, counts = np.unique(valid_vals, return_counts=True)
            max_count = np.max(counts)
            
            # Find all candidate values with the maximum frequency
            candidates = unique_vals[counts == max_count]
            
            # Tie-breaking: select the smallest value
            fill_value = np.min(candidates)
            
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. Choose 'mean', 'median', or 'mode'.")
            
        # Impute NaN entries in the column
        result[~valid_mask, col_idx] = fill_value
        
    return result