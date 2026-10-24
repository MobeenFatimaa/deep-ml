import numpy as np

def calculate_correlation_matrix(X, Y=None):
    """
    Calculate the correlation matrix for a given dataset X and optional dataset Y.
    
    Args:
        X: 2D numpy array of shape (n_samples, n_features_X)
        Y: Optional 2D numpy array of shape (n_samples, n_features_Y)
        
    Returns:
        A 2D numpy array representing the correlation matrix.
    """
    if Y is None:
        # np.corrcoef expects variables in rows, so we use rowvar=False 
        # since features are in columns.
        return np.corrcoef(X, rowvar=False)
    else:
        # If Y is provided, combine X and Y horizontally or compute corrcoef directly.
        # np.corrcoef with rowvar=False on column-stacked array will give the cross-correlation matrix.
        combined = np.hstack((X, Y))
        corr_matrix = np.corrcoef(combined, rowvar=False)
        
        n_x = X.shape[1]
        # Extract the cross-correlation block between X and Y
        return corr_matrix[:n_x, n_x:]