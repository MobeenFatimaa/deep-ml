import numpy as np


def ipo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta,
):
    """
    Identity Preference Optimization (IPO) loss.

    Args:
        policy_chosen_logps: log πθ(y_w) per example
        policy_rejected_logps: log πθ(y_l) per example
        ref_chosen_logps: log πref(y_w) per example
        ref_rejected_logps: log πref(y_l) per example
        beta: KL temperature (target gap is 1/(2β))

    Returns:
        Mean IPO loss, rounded to 6 decimals.
    """
    if beta <= 0:
        raise ValueError("beta must be greater than 0.")

    pi_w = np.asarray(policy_chosen_logps, dtype=np.float64)
    pi_l = np.asarray(policy_rejected_logps, dtype=np.float64)
    ref_w = np.asarray(ref_chosen_logps, dtype=np.float64)
    ref_l = np.asarray(ref_rejected_logps, dtype=np.float64)

    margin = (pi_w - ref_w) - (pi_l - ref_l)
    target_margin = 1.0 / (2.0 * float(beta))
    loss = np.mean((margin - target_margin) ** 2)

    return float(np.round(loss, 6))