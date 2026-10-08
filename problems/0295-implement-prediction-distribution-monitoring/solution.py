import numpy as np


def monitor_prediction_distribution(
    reference_preds: list, current_preds: list, n_bins: int = 10
) -> dict:
    """Monitor prediction distribution changes between reference and current predictions.

    Args:
        reference_preds: List of reference prediction scores (floats between 0
          and 1)
        current_preds: List of current prediction scores (floats between 0 and 1)
        n_bins: Number of bins for histogram comparison

    Returns:
        Dictionary with keys: 'mean_shift', 'std_ratio', 'js_divergence',
        'drift_detected'
    """
    ref = np.array(reference_preds)
    cur = np.array(current_preds)

    # 1. Mean Shift
    mean_shift = float(np.mean(cur) - np.mean(ref))

    # 2. Standard Deviation Ratio
    ref_std = float(np.std(ref))
    cur_std = float(np.std(cur))
    std_ratio = cur_std / ref_std if ref_std > 0 else 0.0

    # 3. Jensen-Shannon Divergence calculation
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Bin counts
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    # Apply Laplace smoothing: P(bin) = (count + 1) / (total + n_bins)
    p = (ref_counts + 1.0) / (len(ref) + n_bins)
    q = (cur_counts + 1.0) / (len(cur) + n_bins)

    # Mixture distribution M = 0.5 * (P + Q)
    m = 0.5 * (p + q)

    # KL Divergence: KL(P || M) = sum(P * log(P / M))
    kl_p_m = np.sum(p * np.log(p / m))
    kl_q_m = np.sum(q * np.log(q / m))

    # JS Divergence: JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    js_div = float(0.5 * kl_p_m + 0.5 * kl_q_m)

    # 4. Drift detection threshold check
    drift_detected = bool(js_div > 0.1)

    return {
        "mean_shift": round(mean_shift, 4),
        "std_ratio": round(std_ratio, 4),
        "js_divergence": round(js_div, 4),
        "drift_detected": drift_detected,
    }