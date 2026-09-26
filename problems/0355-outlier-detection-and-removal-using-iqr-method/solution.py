import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
    """
    Detect and remove outliers using the IQR method.
    
    Args:
        data: List of numerical values
        k: IQR multiplier for determining outlier bounds (default 1.5)
    
    Returns:
        Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
    """
    if not data:
        return {
            'cleaned_data': [],
            'outlier_indices': [],
            'lower_bound': 0.0,
            'upper_bound': 0.0
        }

    # Convert to numpy array for percentile computation
    arr = np.array(data, dtype=float)

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    # Calculate lower and upper bounds
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    cleaned_data = []
    outlier_indices = []

    # Identify outliers and construct cleaned dataset
    for idx, val in enumerate(data):
        if val < lower_bound or val > upper_bound:
            outlier_indices.append(idx)
        else:
            cleaned_data.append(round(float(val), 4))

    return {
        'cleaned_data': cleaned_data,
        'outlier_indices': outlier_indices,
        'lower_bound': round(float(lower_bound), 4),
        'upper_bound': round(float(upper_bound), 4)
    }