import numpy as np


def rvm_fit(Phi, y, n_iter, alpha_init, beta_init, prune=1e6):
    """Fits a Relevance Vector Machine (RVM) regression model using Type-II Maximum Likelihood."""
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = Phi.shape

    active = list(range(D))
    alpha = np.full(D, alpha_init, dtype=float)
    beta = float(beta_init)

    for _ in range(n_iter):
        if len(active) == 0:
            break
            
        Phi_a = Phi[:, active]
        alpha_a = alpha[active]

        # 1. Compute posterior covariance Sigma and mean mu
        try:
            gram = Phi_a.T @ Phi_a
            inv_term = beta * gram + np.diag(alpha_a)
            Sigma = np.linalg.inv(inv_term)
        except np.linalg.LinAlgError:
            Sigma = np.linalg.pinv(inv_term)

        mu = beta * (Sigma @ (Phi_a.T @ y))

        # 2. Update alpha, beta, and gamma
        gamma = 1.0 - alpha_a * np.diag(Sigma)
        
        # Avoid division by zero or negative values
        mu_sq = mu ** 2
        new_alpha = np.empty_like(alpha_a)
        for i in range(len(active)):
            if mu_sq[i] > 1e-12:
                new_alpha[i] = gamma[i] / mu_sq[i]
            else:
                new_alpha[i] = prune * 10.0  # Force pruning if weight is zero
                
        # Update beta
        residual_norm_sq = np.sum((y - Phi_a @ mu) ** 2)
        sum_gamma = np.sum(gamma)
        new_beta = (N - sum_gamma) / residual_norm_sq if residual_norm_sq > 0 else beta
        new_beta = max(1e-6, new_beta)

        alpha[active] = new_alpha
        beta = new_beta

        # 3. Prune columns whose alpha exceeds the threshold
        new_active = []
        new_alpha_list = []
        for idx, a_val in zip(active, alpha[active]):
            if a_val <= prune:
                new_active.append(idx)
                new_alpha_list.append(a_val)
            else:
                # Reset alpha for pruned features just in case they reactivate or are needed
                alpha[idx] = alpha_init
                
        active = new_active
        if len(active) == 0:
            # If everything was pruned, keep at least the best column or break
            break
            
    # Final recomputation for the active set
    Phi_a = Phi[:, active]
    alpha_a = alpha[active]
    try:
        gram = Phi_a.T @ Phi_a
        inv_term = beta * gram + np.diag(alpha_a)
        Sigma = np.linalg.inv(inv_term)
    except np.linalg.LinAlgError:
        Sigma = np.linalg.pinv(inv_term)

    mu = beta * (Sigma @ (Phi_a.T @ y))

    return {
        "active": active,
        "mu": mu,
        "alpha": alpha_a,
        "beta": beta,
        "Sigma": Sigma
    }


def rvm_predict(Phi_star, model):
    """Computes predictive mean and variance using the RVM model."""
    Phi_star = np.asarray(Phi_star, dtype=float)
    active = model["active"]
    mu = model["mu"]
    beta = model["beta"]
    Sigma = model["Sigma"]

    Phi_star_a = Phi_star[:, active]

    mean = Phi_star_a @ mu
    # Predictive variance: beta^-1 + phi_*^T Sigma phi_*
    var = (1.0 / beta) + np.sum((Phi_star_a @ Sigma) * Phi_star_a, axis=1)

    return mean, var