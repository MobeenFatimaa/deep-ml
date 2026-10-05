import numpy as np


def grpo_objective(
    rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01
) -> float:
    """Compute the GRPO objective function as defined in DeepSeekMath.

    Args:
        rhos: List or array of likelihood ratios (pi_theta / pi_theta_old).
        A: List or array of advantage estimates.
        pi_theta_old: List or array of old policy probabilities.
        pi_theta_ref: List or array of reference policy probabilities.
        epsilon: Clipping parameter for PPO surrogate loss.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO scalar objective value.
    """
    rhos = np.array(rhos, dtype=float)
    A = np.array(A, dtype=float)
    pi_theta_old = np.array(pi_theta_old, dtype=float)
    pi_theta_ref = np.array(pi_theta_ref, dtype=float)

    # Step 1: PPO-style clipped surrogate objective
    unclipped_obj = rhos * A
    clipped_obj = np.clip(rhos, 1.0 - epsilon, 1.0 + epsilon) * A
    surrogate_obj = np.minimum(unclipped_obj, clipped_obj)

    # Step 2: Compute current policy probability pi_theta = rho * pi_theta_old
    pi_theta = rhos * pi_theta_old

    # Step 3: Compute reference-to-policy ratio r = pi_ref / pi_theta
    r = pi_theta_ref / pi_theta

    # Step 4: Compute unbiased importance-weighted KL penalty:
    # D_KL(i) = rho_i * ( (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1 )
    kl_div = rhos * (r - np.log(r) - 1.0)

    # Step 5: Combine surrogate objective and KL penalty, averaged over sample group
    final_objective = np.mean(surrogate_obj - beta * kl_div)

    return float(final_objective)