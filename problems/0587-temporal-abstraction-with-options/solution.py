import numpy as np

def options_value_iteration(n_states, terminal_states, transitions, options, gamma, theta=1e-10):
    """
    Compute SMDP value iteration with options.

    Args:
        n_states: int, total number of states
        terminal_states: set of int, terminal state indices
        transitions: dict mapping state -> action -> list of (next_state, prob, reward)
        options: list of dicts, each defining 'initiation', 'policy', and 'termination'
        gamma: float, discount factor
        theta: float, convergence threshold

    Returns:
        (V, policy): V is a list of floats rounded to 4 decimals; policy is a list of ints.
    """
    option_R = []
    option_F = []
    
    for opt in options:
        policy_dict = opt['policy']
        term_dict = opt.get('termination', {})
        
        S_o = list(policy_dict.keys())
        N_o = len(S_o)
        
        R_o_full = np.zeros(n_states, dtype=float)
        F_o_full = np.zeros((n_states, n_states), dtype=float)
        
        if N_o == 0:
            option_R.append(R_o_full)
            option_F.append(F_o_full)
            continue
            
        state_to_idx = {s: i for i, s in enumerate(S_o)}
        
        A = np.eye(N_o, dtype=float)
        b_R = np.zeros(N_o, dtype=float)
        B_F = np.zeros((N_o, n_states), dtype=float)
        
        for i, s in enumerate(S_o):
            a = policy_dict[s]
            trans_list = transitions.get(s, {}).get(a, [])
            
            next_state_probs = {}
            next_state_rewards = {}
            for next_s, prob, reward in trans_list:
                next_state_probs[next_s] = next_state_probs.get(next_s, 0.0) + prob
                next_state_rewards[next_s] = next_state_rewards.get(next_s, 0.0) + prob * reward
                
            b_R[i] = sum(next_state_rewards.values())
            
            for next_s, prob_sum in next_state_probs.items():
                beta_val = term_dict.get(next_s, 1.0)
                if next_s in terminal_states:
                    beta_val = 1.0
                
                if next_s in state_to_idx:
                    j = state_to_idx[next_s]
                    A[i, j] -= gamma * prob_sum * (1.0 - beta_val)
                
                if 0 <= next_s < n_states:
                    B_F[i, next_s] += gamma * prob_sum * beta_val
                    
        try:
            R_sub = np.linalg.solve(A, b_R)
            F_sub = np.linalg.solve(A, B_F)
        except np.linalg.LinAlgError:
            R_sub = np.zeros(N_o)
            F_sub = np.zeros((N_o, n_states))
            
        for i, s in enumerate(S_o):
            R_o_full[s] = R_sub[i]
            for s_prime in range(n_states):
                F_o_full[s, s_prime] = F_sub[i, s_prime]
                
        option_R.append(R_o_full)
        option_F.append(F_o_full)
        
    V = np.zeros(n_states, dtype=float)
    
    while True:
        delta = 0.0
        V_new = V.copy()
        for s in range(n_states):
            if s in terminal_states:
                V_new[s] = 0.0
                continue
                
            available_options = [o_idx for o_idx, opt in enumerate(options) if s in opt.get('initiation', set())]
            if not available_options:
                V_new[s] = 0.0
                continue
                
            best_val = -float('inf')
            for o_idx in available_options:
                r_val = option_R[o_idx][s]
                f_val = np.dot(option_F[o_idx][s], V)
                val = r_val + f_val
                if val > best_val:
                    best_val = val
                    
            V_new[s] = best_val
            delta = max(delta, abs(V_new[s] - V[s]))
            
        V = V_new
        if delta < theta:
            break
            
    policy = [0] * n_states
    for s in range(n_states):
        if s in terminal_states:
            policy[s] = 0
            continue
            
        available_options = [o_idx for o_idx, opt in enumerate(options) if s in opt.get('initiation', set())]
        if not available_options:
            policy[s] = 0
            continue
            
        best_val = -float('inf')
        best_o = available_options[0]
        for o_idx in available_options:
            r_val = option_R[o_idx][s]
            f_val = np.dot(option_F[o_idx][s], V)
            val = r_val + f_val
            if val > best_val + 1e-7:
                best_val = val
                best_o = o_idx
                
        policy[s] = best_o
        
    V_rounded = [round(float(val), 4) for val in V]
    return V_rounded, policy