import numpy as np

def agentopsd_turn_advantages(
    turn_evidence,
    A_seq,
    B0,
    gamma: float = 0.95,
    b: float = 0.2,
    lam: float = 0.5,
    eps: float = 1e-4,
) -> np.ndarray:
    """Full AgentOPSD credit assignment → (K,) reshaped advantages."""
    turn_evidence = np.array(turn_evidence, dtype=np.float64)
    K = len(turn_evidence)
    
    # Clip B0 to avoid logit boundaries (e.g., [1e-6, 1 - 1e-6])
    B0_clipped = np.clip(B0, 1e-7, 1.0 - 1e-7)
    
    # 1. l_0 = logit(clip(B0))
    # logit(p) = log(p / (1 - p))
    def logit(p):
        return np.log(p / (1.0 - p))
        
    l_0 = logit(B0_clipped)
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
        
    # 2-4. Recursion for K turns
    c_prev = 0.0
    b_prev = B0_clipped
    
    B_list = []
    for k in range(K):
        ek = turn_evidence[k]
        ck = gamma * c_prev + ek
        lk = l_0 + ck
        bk = sigmoid(lk)
        B_list.append(bk)
        c_prev = ck
        
    B_arr = np.array(B_list, dtype=np.float64)
    
    # 5. Delta B_k = B_k - B_{k-1} (with B_0 as prior)
    B_full = np.concatenate([[B0_clipped], B_arr])
    delta_B = B_full[1:] - B_full[:-1]
    
    # 6. q_k = sign(A_seq) * delta_B_k
    sign_A = np.sign(A_seq)
    q = sign_A * delta_B
    
    # 7. z_k = within-trajectory z-score of q (population std, + eps in denominator)
    mean_q = np.mean(q)
    std_q = np.std(q)  # population std (ddof=0)
    z = (q - mean_q) / (std_q + eps)
    
    # 8. w_k = clip(1 + b * z_k, 1 - b, 1 + b)
    w = np.clip(1.0 + b * z, 1.0 - b, 1.0 + b)
    
    # 9. \tilde{A}_k = A_seq * [(1 - \lambda) + \lambda * w_k]
    A_tilde = A_seq * ((1.0 - lam) + lam * w)
    
    return A_tilde.astype(np.float64)