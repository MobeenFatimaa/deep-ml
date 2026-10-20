import numpy as np
from collections import Counter

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    arr = np.asarray(data)
    
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    
    # Mode calculation using Counter
    flat_data = arr.tolist() if isinstance(arr, np.ndarray) else list(data)
    counter = Counter(flat_data)
    mode_val = counter.most_common(1)[0][0]
    
    variance_val = float(np.var(arr, ddof=0))
    std_val = float(np.std(arr, ddof=0))
    
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    iqr_val = float(p75 - p25)
    
    return {
        'mean': mean_val,
        'median': median_val,
        'mode': mode_val,
        'variance': variance_val,
        'standard_deviation': std_val,
        '25th_percentile': p25,
        '50th_percentile': p50,
        '75th_percentile': p75,
        'interquartile_range': iqr_val
    }