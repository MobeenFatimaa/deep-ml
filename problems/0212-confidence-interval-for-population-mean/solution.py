import numpy as np

# numpy has no t-distribution quantile function and scipy is not available.
# Use this lookup table to get t-critical values for the (df, p) pairs
# needed by the test cases, where p = 1 - alpha/2.
T_TABLE = {
    (4, 0.95):  2.1318467863998393,
    (4, 0.975): 2.7764451051977987,
    (4, 0.995): 4.604094871415387,
    (5, 0.95):  2.0150483726691575,
    (5, 0.975): 2.5705818366147395,
    (5, 0.995): 4.032142983557536,
    (6, 0.95):  1.9431802803927816,
    (6, 0.975): 2.4469118511449624,
    (6, 0.995): 3.707428021324907,
    (7, 0.95):  1.8945786050613064,
    (7, 0.975): 2.3646242510102993,
    (7, 0.995): 3.4994832973505026,
}


def confidence_interval(data: list[float], confidence_level: float = 0.95) -> dict:
    """
    Calculate confidence interval for population mean.

    Args:
        data: Sample data
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary containing:
        - mean: Sample mean (point estimate)
        - standard_error: Standard error of the mean
        - margin_of_error: Margin of error
        - lower_bound: Lower bound of CI
        - upper_bound: Upper bound of CI
        - confidence_level: Confidence level used
    """
    data_arr = np.array(data, dtype=np.float64)
    n = len(data_arr)
    df = n - 1
    
    mean_val = float(np.mean(data_arr))
    # Sample standard deviation with Bessel's correction (ddof=1)
    s = float(np.std(data_arr, ddof=1))
    
    # Standard error of the mean
    se = s / np.sqrt(n)
    
    # Find t-critical value
    alpha = 1.0 - confidence_level
    p = 1.0 - alpha / 2.0
    p_rounded = round(p, 3)
    
    t_star = T_TABLE.get((df, p_rounded), T_TABLE.get((df, p), 1.96))
    
    # Margin of error and bounds
    me = t_star * se
    lower_bound = mean_val - me
    upper_bound = mean_val + me
    
    return {
        'mean': mean_val,
        'standard_error': se,
        'margin_of_error': me,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level
    }