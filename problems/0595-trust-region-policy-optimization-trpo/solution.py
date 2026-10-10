import numpy as np

def trpo_step(
    theta: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    num_states: int,
    num_actions: int,
    delta: float,
    cg_iters: int = 10,
    line_search_steps: int = 10,
    line_search_decay: float = 0.5
) -> np.ndarray:
    """
    Perform a single TRPO policy update for a tabular softmax policy.
    """
    theta = theta.copy()
    theta_2d = theta.reshape(num_states, num_actions)
    
    # 1. Compute softmax policy probabilities from current logits
    def compute_probs(logits):
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    pi_old = compute_probs(theta_2d)
    
    # Compute state visitation counts and frequencies d(s)
    N = len(states)
    state_counts = np.bincount(states, minlength=num_states)
    d_s = state_counts / max(N, 1)
    
    # 2. Compute the gradient of the surrogate objective g
    # L(theta) = (1/N) * sum_i (pi_new(a_i|s_i) / pi_old(a_i|s_i)) * A_i
    # At theta = theta_old, ratio is 1, gradient w.r.t theta_{s,a} is:
    # g_{s,a} = (1/N) * sum_{i: s_i=s} A_i * (I(a_i == a) - pi(a|s))
    g_2d = np.zeros((num_states, num_actions))
    for i in range(N):
        s = states[i]
        a = actions[i]
        adv = advantages[i]
        g_2d[s] += adv * (np.arange(num_actions) == a)
        g_2d[s] -= adv * pi_old[s]
    g_2d /= max(N, 1)
    g = g_2d.ravel()
    
    # 3 & 4. Fisher Information Matrix-vector product and Conjugate Gradient
    damping = 1e-8
    
    def fisher_vector_product(v):
        v_2d = v.reshape(num_states, num_actions)
        Fv_2d = np.zeros_like(v_2d)
        for s in range(num_states):
            if d_s[s] > 0:
                pi_s = pi_old[s]
                # F_s v_s = sum_a pi(a|s) (e_a - pi_s)(e_a - pi_s)^T v_s + damping * v_s
                # = sum_a pi(a|s) (e_a - pi_s) (e_a - pi_s)^T v_s
                # Notice (e_a - pi_s)^T v_s = v_{s,a} - sum_a' pi(a'|s) v_{s,a'}
                dot_prod = np.sum(pi_s * v_2d[s])
                term = v_2d[s] - dot_prod
                Fv_s = np.sum(pi_s[:, None] * term * (np.eye(num_actions) - pi_s[None, :]), axis=0) # simpler below
                
                # Direct formulation: F_s v_s = pi_s * v_s - pi_s * (pi_s . v_s) + damping * v_s
                Fv_s = pi_s * v_2d[s] - pi_s * dot_prod + damping * v_2d[s]
                Fv_2d[s] = d_s[s] * Fv_s
        return Fv_2d.ravel()

    # Conjugate Gradient solver for F x = g
    x = np.zeros_like(g)
    r = g.copy()
    p = r.copy()
    rs_old = np.dot(r, r)
    
    for _ in range(cg_iters):
        Ap = fisher_vector_product(p)
        pAp = np.dot(p, Ap)
        if pAp <= 1e-12:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if rs_new < 1e-10:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
        
    # 5. Compute maximum step size satisfying KL constraint
    # x^T F x
    Fx = fisher_vector_product(x)
    xFx = np.dot(x, Fx)
    if xFx <= 1e-12:
        return theta
        
    max_step_len = np.sqrt(2.0 * delta / xFx)
    
    # 6. Backtracking line search
    # Baseline surrogate objective value at theta_old is 1.0 (since ratio is 1)
    # But let's compute actual surrogate objective
    def compute_surrogate(pi_new):
        ratios = np.array([pi_new[states[i], actions[i]] / pi_old[states[i], actions[i]] for i in range(N)])
        return np.mean(ratios * advantages) if N > 0 else 0.0

    def compute_kl(pi_new):
        kl = 0.0
        for s in range(num_states):
            if d_s[s] > 0:
                # D_KL(pi_old || pi_new) = sum_a pi_old(a|s) * log(pi_old(a|s) / pi_new(a|s))
                # Add small epsilon for numerical stability
                p_old = pi_old[s]
                p_new = np.clip(pi_new[s], 1e-12, 1.0)
                kl_s = np.sum(p_old * np.log(p_old / p_new))
                kl += d_s[s] * kl_s
        return kl

    old_surrogate = compute_surrogate(pi_old)
    
    # Try step sizes
    success = False
    best_theta = theta
    
    for step in range(line_search_steps):
        step_frac = max_step_len * (line_search_decay ** step)
        candidate_theta = theta + step_frac * x
        pi_new = compute_probs(candidate_theta.reshape(num_states, num_actions))
        
        kl_val = compute_kl(pi_new)
        new_surrogate = compute_surrogate(pi_new)
        
        if kl_val <= delta and new_surrogate > old_surrogate:
            best_theta = candidate_theta
            success = True
            break
            
    return best_theta