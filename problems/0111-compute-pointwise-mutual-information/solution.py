import numpy as np


def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
    """Calculate Pointwise Mutual Information (PMI) in bits (base-2 logarithm).

    Args:
        joint_counts: Number of co-occurrences of events X and Y (or array of
          counts)
        total_counts_x: Total occurrences of event X (or array of counts)
        total_counts_y: Total occurrences of event Y (or array of counts)
        total_samples: Total number of samples/corpus size N

    Returns:
        PMI value in bits, rounded to 3 decimal places
    """
    joint_counts = np.asarray(joint_counts, dtype=float)
    total_counts_x = np.asarray(total_counts_x, dtype=float)
    total_counts_y = np.asarray(total_counts_y, dtype=float)

    # Calculate probabilities
    p_xy = joint_counts / total_samples
    p_x = total_counts_x / total_samples
    p_y = total_counts_y / total_samples

    # Compute expected joint probability under independence
    p_expected = p_x * p_y

    # Calculate PMI using base-2 logarithm (bits)
    # Handle zero probabilities safely (log(0) -> -inf or 0 depending on context)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(p_xy / p_expected)
        pmi = np.nan_to_num(pmi, nan=0.0, neginf=0.0)

    # Return float if input was scalar, otherwise return array
    result = np.round(pmi, 3)
    return float(result) if result.ndim == 0 else result