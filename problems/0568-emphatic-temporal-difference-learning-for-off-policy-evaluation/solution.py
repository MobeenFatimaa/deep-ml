def emphatic_td(trajectory: list, gamma: float, lam: float, alpha: float, target_policy: dict, behavior_policy: dict, interest: dict, initial_values: dict) -> dict:
    """
    Emphatic TD(lambda) for off-policy tabular policy evaluation.
    
    Args:
        trajectory: list of (state, action, reward, next_state) tuples
        gamma: discount factor
        lam: lambda for eligibility traces
        alpha: step size
        target_policy: dict (state, action) -> probability under target policy
        behavior_policy: dict (state, action) -> probability under behavior policy
        interest: dict state -> interest value (default 0 for missing states)
        initial_values: dict state -> initial value estimate
    
    Returns:
        Dict mapping state -> rounded value estimate
    """
    V = dict(initial_values)
    e = {s: 0.0 for s in V}
    F = 0.0
    prev_rho = 0.0
    
    for state, action, reward, next_state in trajectory:
        if state not in V:
            V[state] = 0.0
            e[state] = 0.0
        if next_state is not None and next_state not in V:
            V[next_state] = 0.0
            e[next_state] = 0.0
            
        pi_prob = target_policy.get((state, action), 0.0)
        b_prob = behavior_policy.get((state, action), 1.0)
        rho = (pi_prob / b_prob) if b_prob > 0 else 0.0
        
        # Follow-on trace update
        F = gamma * prev_rho * F + interest.get(state, 0.0)
        
        # Emphasis M_t
        M = lam * interest.get(state, 0.0) + (1.0 - lam) * F
        
        # TD error
        v_s = V.get(state, 0.0)
        v_next = V.get(next_state, 0.0) if next_state is not None else 0.0
        delta = reward + gamma * v_next - v_s
        
        # Eligibility traces update using current rho
        for s in V:
            e[s] = gamma * lam * e[s]
        e[state] += M
        for s in V:
            e[s] *= rho
        
        # Value update
        for s in V:
            V[s] += alpha * delta * e[s]
            
        prev_rho = rho
        
    result = {s: round(float(V[s]), 4) for s in initial_values}
    return dict(sorted(result.items()))