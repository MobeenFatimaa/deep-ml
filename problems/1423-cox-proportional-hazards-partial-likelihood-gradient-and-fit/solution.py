import numpy as np


def neg_log_partial_likelihood(beta, X, times, events) -> float:
    """Compute the negative log partial likelihood for the Cox Proportional Hazards model."""
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    
    # Sort by event/censoring times in ascending order
    sort_idx = np.argsort(times)
    X_sorted = X[sort_idx]
    events_sorted = events[sort_idx]
    
    n = len(times)
    total_neg_ll = 0.0
    
    for i in range(n):
        if events_sorted[i] == 1:
            x_i = X_sorted[i]
            x_i_beta = np.dot(x_i, beta)
            
            # Risk set includes all subjects with time >= current time (index i onwards in sorted array)
            risk_betas = X_sorted[i:] @ beta
            
            # Log-sum-exp trick for numerical stability
            max_val = np.max(risk_betas)
            log_sum_exp = max_val + np.log(np.sum(np.exp(risk_betas - max_val)))
            
            total_neg_ll -= (x_i_beta - log_sum_exp)
            
    return float(total_neg_ll)


def gradient(beta, X, times, events) -> np.ndarray:
    """Compute the gradient of the negative log partial likelihood."""
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    
    p = X.shape[1]
    sort_idx = np.argsort(times)
    X_sorted = X[sort_idx]
    events_sorted = events[sort_idx]
    
    n = len(times)
    grad = np.zeros(p)
    
    for i in range(n):
        if events_sorted[i] == 1:
            x_i = X_sorted[i]
            risk_X = X_sorted[i:]
            risk_betas = risk_X @ beta
            
            # Numerical stability using max subtraction
            max_val = np.max(risk_betas)
            exp_betas = np.exp(risk_betas - max_val)
            sum_exp = np.sum(exp_betas)
            
            # Hazard-weighted mean covariate in the risk set
            weighted_x = np.sum(risk_X * exp_betas[:, None], axis=0) / sum_exp
            grad += (x_i - weighted_x)
            
    return -grad


def fit_cox(X, times, events, lr=0.1, n_iter=500) -> np.ndarray:
    """Fit Cox model coefficients using gradient descent on the negative log partial likelihood."""
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    
    p = X.shape[1]
    beta = np.zeros(p)
    
    for _ in range(n_iter):
        grad = gradient(beta, X, times, events)
        beta -= lr * grad
        
    return beta


def hazard_ratio(beta) -> np.ndarray:
    """Compute hazard ratios from coefficients."""
    return np.exp(beta)