import numpy as np

def _log_sum_exp(a, b):
    """Numerically stable log-sum-exp for two log-probabilities."""
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + np.log(1.0 + np.exp(b - a))
    else:
        return b + np.log(1.0 + np.exp(a - b))

def ctc_loss(log_probs, target, blank=0):
    """Negative log-likelihood of target under CTC (float rounded to 4 decimals, inf if impossible)."""
    T, C = log_probs.shape
    
    # Construct extended target sequence l' and its length S
    if len(target) == 0:
        l_prime = [blank]
    else:
        l_prime = [blank]
        for t in target:
            l_prime.append(t)
            l_prime.append(blank)
            
    S = len(l_prime)
    
    # Check if sequence length T is shorter than the target length
    if T < len(target):
        return float("inf")
        
    # Initialize log alpha table with -inf
    log_alpha = np.full((T, S), -np.inf)
    
    # Initialization at t = 0
    log_alpha[0, 0] = log_probs[0, l_prime[0]]
    if S > 1:
        log_alpha[0, 1] = log_probs[0, l_prime[1]]
        
    # Forward pass over time steps t = 1 ... T-1
    for t in range(1, T):
        for s in range(S):
            # Accumulate predecessor log probabilities
            # 1. Stay in state s
            acc = log_alpha[t-1, s]
            
            # 2. Come from state s-1
            if s > 0:
                acc = _log_sum_exp(acc, log_alpha[t-1, s-1])
                
            # 3. Come from state s-2 (skip blank, only if l'_s != blank and l'_s != l'_{s-2})
            if s > 1 and l_prime[s] != blank and l_prime[s] != l_prime[s-2]:
                acc = _log_sum_exp(acc, log_alpha[t-1, s-2])
                
            # Multiply by emission probability (add in log space)
            log_alpha[t, s] = acc + log_probs[t, l_prime[s]]
            
    # Total probability is sum of final states S-1 and S-2 (if S-2 >= 0)
    final_log_prob = log_alpha[T-1, S-1]
    if S > 1:
        final_log_prob = _log_sum_exp(final_log_prob, log_alpha[T-1, S-2])
        
    if final_log_prob == -np.inf:
        return float("inf")
        
    # Negative log-likelihood rounded to 4 decimals
    return round(float(-final_log_prob), 4)

def ctc_greedy_decode(log_probs, blank=0):
    """Best-path decoding: per-frame arg-max, merge repeats, drop blanks."""
    best_path = np.argmax(log_probs, axis=-1)
    
    decoded = []
    prev = None
    for token in best_path:
        token = int(token)
        if token != blank and token != prev:
            decoded.append(token)
        if token != blank:
            prev = token
        else:
            prev = blank
            
    return decoded