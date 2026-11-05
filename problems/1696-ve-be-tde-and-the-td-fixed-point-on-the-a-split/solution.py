import numpy as np


def objective_minimizers(X, transitions, start_probs, gamma):
    """Computes the value function each objective (VE, BE, TDE, TD) selects for a linear value function."""
    n_states, d = X.shape
    start_probs = np.array(start_probs, dtype=float)
    
    P = np.zeros((n_states, n_states))
    r_bar = np.zeros(n_states)
    
    for s, p, r, s_next in transitions:
        r_bar[s] += p * r
        if s_next != -1:
            P[s, s_next] += p
            
    # Compute on-policy distribution mu: eta = h + P^T eta -> (I - P^T) eta = h
    I = np.eye(n_states)
    eta = np.linalg.solve(I - P.T, start_probs)
    mu = eta / np.sum(eta)
    D = np.diag(mu)
    sqrt_D = np.diag(np.sqrt(mu))
    
    # True value function v_pi = (I - gamma P)^-1 r_bar
    v_pi = np.linalg.solve(I - gamma * P, r_bar)
    
    # 1. VE: minimize sum mu(s) (v_pi(s) - Xw(s))^2
    A_ve = sqrt_D @ X
    b_ve = sqrt_D @ v_pi
    w_ve, _, _, _ = np.linalg.lstsq(A_ve, b_ve, rcond=None)
    val_ve = X @ w_ve
    
    # 2. BE: minimize sum mu(s) (r_bar + gamma P Xw - Xw)^2
    A_be = sqrt_D @ (I - gamma * P) @ X
    b_be = sqrt_D @ r_bar
    w_be, _, _, _ = np.linalg.lstsq(A_be, b_be, rcond=None)
    val_be = X @ w_be
    
    # 3. TDE: minimize sum mu(s) sum_{(p,r,s')} p (r + gamma v_hat(s') - v_hat(s))^2
    rows_A = []
    rows_b = []
    for s, p, r, s_next in transitions:
        weight = np.sqrt(mu[s] * p)
        feat_next = X[s_next] if s_next != -1 else np.zeros(d)
        feat_diff = X[s] - gamma * feat_next
        rows_A.append(weight * feat_diff)
        rows_b.append(weight * r)
    A_tde = np.array(rows_A)
    b_tde = np.array(rows_b)
    w_tde, _, _, _ = np.linalg.lstsq(A_tde, b_tde, rcond=None)
    val_tde = X @ w_tde
    
    # 4. TD: X^T D (X - gamma P X) w = X^T D r_bar
    A_td = X.T @ D @ (X - gamma * P @ X)
    b_td = X.T @ D @ r_bar
    w_td, _, _, _ = np.linalg.lstsq(A_td, b_td, rcond=None)
    val_td = X @ w_td
    
    return {
        "VE": np.round(val_ve + 0.0, 4),
        "BE": np.round(val_be + 0.0, 4),
        "TDE": np.round(val_tde + 0.0, 4),
        "TD": np.round(val_td + 0.0, 4),
    }