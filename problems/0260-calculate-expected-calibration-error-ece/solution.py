import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or array of true binary labels (0 or 1)
        y_prob: List or array of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    N = len(y_true)
    if N == 0:
        return 0.0
        
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        # First bin includes both endpoints [0, 1/n_bins], subsequent bins use (lower, upper]
        if i == 0:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob > lower) & (y_prob <= upper)
            
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            # Proportion of samples in the current bin (weight)
            weight = bin_count / N
            
            # Average predicted probability (confidence) in the bin
            avg_confidence = np.mean(y_prob[in_bin])
            
            # Actual accuracy (fraction of positive labels) in the bin
            accuracy = np.mean(y_true[in_bin])
            
            # Accumulate weighted absolute difference
            ece += weight * abs(accuracy - avg_confidence)
            
    return round(float(ece), 3)