import numpy as np


def detect_feature_drift(
    reference_data: list, production_data: list, num_bins: int = 10
) -> dict:
    """Detect feature drift using Population Stability Index (PSI).

    Args:
        reference_data: List of feature values from reference distribution
          (e.g., training)
        production_data: List of feature values from production distribution
        num_bins: Number of bins for histogram comparison

    Returns:
        dict with 'psi', 'drift_detected', and 'drift_level'
    """
    if not reference_data or not production_data:
        return {}

    ref_arr = np.array(reference_data)
    prod_arr = np.array(production_data)

    # Define bin edges using combined minimum and maximum values
    min_val = min(np.min(ref_arr), np.min(prod_arr))
    max_val = max(np.max(ref_arr), np.max(prod_arr))

    # Handle edge case where all values are identical
    if min_val == max_val:
        bin_edges = np.linspace(min_val - 1.0, max_val + 1.0, num_bins + 1)
    else:
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Compute bin counts for reference and production datasets
    ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
    prod_counts, _ = np.histogram(prod_arr, bins=bin_edges)

    # Convert counts to proportions
    ref_pct = ref_counts / len(ref_arr)
    prod_pct = prod_counts / len(prod_arr)

    # Replace zero proportions with epsilon to prevent log(0) and division by zero
    epsilon = 0.0001
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    prod_pct = np.where(prod_pct == 0, epsilon, prod_pct)

    # Calculate Population Stability Index (PSI)
    # PSI = sum( (Actual % - Expected %) * ln(Actual % / Expected %) )
    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    psi = float(round(psi, 4))

    # Categorize drift severity based on standard PSI benchmarks
    if psi < 0.1:
        drift_level = "none"
        drift_detected = False
    elif 0.1 <= psi < 0.25:
        drift_level = "moderate"
        drift_detected = True
    else:
        drift_level = "significant"
        drift_detected = True

    return {
        "psi": psi,
        "drift_detected": drift_detected,
        "drift_level": drift_level,
    }