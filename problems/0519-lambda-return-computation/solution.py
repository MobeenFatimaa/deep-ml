import numpy as np

def compute_lambda_returns(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float, dones: np.ndarray = None) -> np.ndarray:
    """
    Compute the lambda-return for each timestep in a trajectory.
    
    Args:
        rewards: Rewards at each step, shape (T,)
        values: Value estimates at each state, shape (T+1,)
        gamma: Discount factor in [0, 1]
        lam: Lambda (trace-decay) parameter in [0, 1]
        dones: Optional binary episode termination flags, shape (T,)
    
    Returns:
        Lambda-returns for each timestep, shape (T,)
    """
    T = len(rewards)
    if dones is None:
        dones = np.zeros(T, dtype=float)
        
    lambda_returns = np.zeros(T, dtype=float)
    
    # Iterate backwards from T-1 down to 0
    # G_t^\lambda = (1 - \lambda) \sum_{n=1}^{T-t} \lambda^{n-1} G_t^{(n)} + \lambda^{T-t} G_t^{(T-t)}
    # Equivalently, using the recursive formulation for the truncated lambda-return:
    # L_t = r_t + gamma * ( (1 - lambda) * V(s_{t+1}) + lambda * L_{t+1} )  (with handling for dones)
    
    # Let's compute iteratively backwards:
    # For a single step, the n-step return target or standard backward view definition:
    # We can compute n-step returns or use the standard backwards linear-time algorithm.
    
    # Standard backwards accumulation of lambda-returns:
    # G_t^L = r_t + gamma * (1 - lambda) * V(s_{t+1}) + gamma * lambda * ( G_{t+1}^L if not done else 0 ) ... wait, let's look at the standard definition.
    # Actually, G_t^{n} = r_{t+1} + gamma * r_{t+2} + ... + gamma^{n-1} V(s_{t+n}) (if not terminated).
    # Let's compute via the explicit definition or backwards recursion:
    
    # For robustness and clarity, let's compute n-step returns up to T-t for each t, or use the standard recursive accumulation.
    # Recursive formula for G_t^\lambda:
    # G_t^\lambda = r_{t} + gamma * (1 - lam) * values[t+1] + gamma * lam * ( (0 if dones[t] else G_{t+1}^\lambda) + (1 if dones[t] else 0) * values[t+1] ) -> wait, let's be precise.
    
    # Let's use the explicit summation definition for each t:
    # For each t, we compute n-step returns for n = 1, ..., T-t:
    for t in range(T):
        g_lambda = 0.0
        sum_weights = 0.0
        
        for n in range(1, T - t + 1):
            # Compute n-step return starting from t (note: rewards are 0-indexed: rewards[t] corresponds to r_{t+1})
            G_n = 0.0
            terminated = False
            for i in range(n):
                idx = t + i
                if idx < T:
                    G_n += (gamma ** i) * rewards[idx]
                    if dones[idx] == 1:
                        terminated = True
                        break
            
            # Bootstrap value
            bootstrap_idx = t + n
            if not terminated and bootstrap_idx < len(values):
                G_n += (gamma ** n) * values[bootstrap_idx]
            
            # Weight for n < T-t is (1 - lam) * lam^(n-1)
            # For n = T-t (the maximum horizon), the weight is lam^(T-t-1)
            if n < T - t:
                weight = (1.0 - lam) * (lam ** (n - 1))
            else:
                weight = lam ** (n - 1)
                
            g_lambda += weight * G_n
            sum_weights += weight
            
        # Normalization in case weights don't sum to 1 due to floating point or truncation
        if sum_weights > 0:
            lambda_returns[t] = g_lambda / sum_weights
        else:
            lambda_returns[t] = values[t]
            
    return lambda_returns