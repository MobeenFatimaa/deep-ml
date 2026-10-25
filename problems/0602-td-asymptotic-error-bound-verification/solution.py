import numpy as np

def td_lambda_error_bound(P, r, gamma, Phi, d, lambdas):
    """
    Compute and verify TD(lambda) asymptotic error bounds.
    
    Args:
        P: Transition probability matrix, shape (n, n)
        r: Expected reward vector, shape (n,)
        gamma: Discount factor (0 <= gamma < 1)
        Phi: Feature matrix, shape (n, k)
        d: Stationary distribution, shape (n,)
        lambdas: List of lambda values to evaluate
    
    Returns:
        Dictionary with keys:
            'true_values': list of floats
            'mc_error': float
            'td_errors': list of floats
            'ratios': list of floats
            'bound': float
            'bound_holds': list of bools
    """
    P = np.array(P, dtype=float)
    r = np.array(r, dtype=float)
    Phi = np.array(Phi, dtype=float)
    d = np.array(d, dtype=float)
    
    n = P.shape[0]
    I = np.eye(n)
    D = np.diag(d)
    
    # 1. True values
    V = np.linalg.solve(I - gamma * P, r)
    
    # 2. Monte Carlo projection
    Phi_D_Phi = Phi.T @ D @ Phi
    Phi_D_V = Phi.T @ D @ V
    w_mc = np.linalg.solve(Phi_D_Phi, Phi_D_V)
    mc_diff = Phi @ w_mc - V
    mc_error = float(np.sqrt(mc_diff.T @ D @ mc_diff))
    
    # 3. TD(lambda) fixed points and errors
    td_errors = []
    ratios = []
    bound_holds = []
    
    bound = float(1.0 / np.sqrt(1.0 - gamma**2))
    
    for lam in lambdas:
        inv_term = np.linalg.inv(I - lam * gamma * P)
        A_lam = Phi.T @ D @ inv_term @ (I - gamma * P) @ Phi
        b_lam = Phi.T @ D @ inv_term @ r
        w_lam = np.linalg.solve(A_lam, b_lam)
        
        td_diff = Phi @ w_lam - V
        td_err = float(np.sqrt(td_diff.T @ D @ td_diff))
        td_errors.append(td_err)
        
        if mc_error <= 1e-12:
            ratio = 0.0
        else:
            ratio = td_err / mc_error
        ratios.append(ratio)
        
        bound_holds.append(bool(ratio <= bound + 1e-9))
        
    return {
        'true_values': np.round(V, 4).tolist(),
        'mc_error': round(mc_error, 4),
        'td_errors': [round(e, 4) for e in td_errors],
        'ratios': [round(r_val, 4) for r_val in ratios],
        'bound': round(bound, 4),
        'bound_holds': bound_holds
    }