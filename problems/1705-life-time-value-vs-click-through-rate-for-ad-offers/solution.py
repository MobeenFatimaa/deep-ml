import numpy as np

def ctr_and_ltv(logs):
    """
    Computes Click-Through Rate (CTR) and Life-Time Value (LTV).
    - CTR = total clicks / total visits
    - LTV = total clicks / number of users
    """
    total_clicks = 0
    total_visits = 0
    num_users = len(logs)
    
    for user_visits in logs:
        total_clicks += sum(user_visits)
        total_visits += len(user_visits)
        
    ctr = total_clicks / total_visits if total_visits > 0 else 0.0
    ltv = total_clicks / num_users if num_users > 0 else 0.0
    
    return float(ctr), float(ltv)


def greedy_vs_ltv(click, trans, ret):
    """
    Compares greedy ad policy (maximizing immediate click probability) with 
    LTV policy (maximizing expected total clicks over repeated visits).
    
    Args:
        click: array of shape (K, A) - click probabilities click[k, a]
        trans: array of shape (A, K, K) - transition probabilities trans[a, k, k2]
        ret: array of shape (K,) - return probability ret[k2]
        
    Returns:
        (greedy_policy, ltv_policy, (greedy_ctr, greedy_ltv), (ltv_ctr, ltv_ltv))
    """
    K, A = click.shape
    
    # 1. Greedy Policy: in each stage k, pick offer a with highest click[k, a]
    # Tie-breaking: lowest offer index, treating values within 10^-9 as ties.
    greedy_policy = []
    for k in range(K):
        best_a = 0
        best_click = click[k, 0]
        for a in range(1, A):
            # If click[k, a] > best_click (with tolerance)
            if click[k, a] > best_click + 1e-9:
                best_click = click[k, a]
                best_a = a
            elif abs(click[k, a] - best_click) <= 1e-9:
                # Tie: pick lowest index (already preferred by strictly greater check, but keep min index)
                if a < best_a:
                    best_a = a
                    best_click = click[k, a]
        greedy_policy.append(best_a)
        
    # 2. LTV Policy: maximizes expected total clicks from every stage.
    # We can solve this via Value Iteration / Linear Programming / Policy Iteration since return probabilities < 1.
    # Let V(k) be the expected total future clicks starting from stage k.
    # V(k) = max_a [ click[k, a] + sum_{k2} trans[a, k, k2] * ret[k2] * V(k2) ]
    # Let's use value iteration to find optimal V and the LTV policy.
    
    V = np.zeros(K)
    ltv_policy = [0] * K
    
    for _ in range(1000): # Value iteration
        new_V = np.zeros(K)
        new_policy = [0] * K
        for k in range(K):
            best_val = -float('inf')
            best_a = 0
            for a in range(A):
                # Expected clicks for choosing offer a in stage k:
                # immediate click + expected future clicks if user returns
                future_clicks = 0.0
                for k2 in range(K):
                    future_clicks += trans[a, k, k2] * ret[k2] * V[k2]
                val = click[k, a] + future_clicks
                
                if val > best_val + 1e-9:
                    best_val = val
                    best_a = a
                elif abs(val - best_val) <= 1e-9:
                    if a < best_a:
                        best_a = a
                        best_val = val
            new_V[k] = best_val
            new_policy[k] = best_a
        if np.max(np.abs(V - new_V)) < 1e-12:
            V = new_V
            ltv_policy = new_policy
            break
        V = new_V
        ltv_policy = new_policy

    # Helper function to evaluate a given policy (expected total clicks, expected visits, CTR)
    def evaluate_policy(policy):
        # We can set up a system of linear equations or solve via simulation / matrix inversion.
        # Let P_visit[k] be the expected total visits starting from stage k.
        # P_visit[k] = 1 + sum_{k2} trans[policy[k], k, k2] * ret[k2] * P_visit[k2]
        # Let C_total[k] be the expected total clicks starting from stage k.
        # C_total[k] = click[k, policy[k]] + sum_{k2} trans[policy[k], k, k2] * ret[k2] * C_total[k2]
        
        T_mat = np.zeros((K, K))
        b_visits = np.ones(K)
        b_clicks = np.zeros(K)
        
        for k in range(K):
            a = policy[k]
            b_clicks[k] = click[k, a]
            for k2 in range(K):
                T_mat[k, k2] = trans[a, k, k2] * ret[k2]
                
        # (I - T_mat) * X = b => X = inv(I - T_mat) * b
        I = np.eye(K)
        inv_matrix = np.linalg.inv(I - T_mat)
        
        visits_vector = inv_matrix @ b_visits
        clicks_vector = inv_matrix @ b_clicks
        
        expected_visits = visits_vector[0] # starting in stage 0
        expected_clicks = clicks_vector[0] # starting in stage 0
        
        ctr = expected_clicks / expected_visits if expected_visits > 0 else 0.0
        ltv = expected_clicks
        
        return float(ctr), float(ltv)

    greedy_ctr, greedy_ltv = evaluate_policy(greedy_policy)
    ltv_ctr, ltv_ltv = evaluate_policy(ltv_policy)
    
    return greedy_policy, ltv_policy, (greedy_ctr, greedy_ltv), (ltv_ctr, ltv_ltv)