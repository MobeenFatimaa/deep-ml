import numpy as np

def speculative_decode(draft_probs: list, target_probs: list, draft_tokens: list, accept_rand: list, sample_rand: list) -> list:
    """
    Simulate speculative decoding end-to-end.
    
    Args:
        draft_probs: K arrays of shape (vocab_size,) - draft model probability distributions
        target_probs: K+1 arrays of shape (vocab_size,) - target model probability distributions
        draft_tokens: K integers - token indices proposed by draft model
        accept_rand: K floats in [0,1) - random values for acceptance decisions
        sample_rand: K+1 floats in [0,1) - random values for token sampling
    
    Returns:
        List of token indices produced by speculative decoding
    """
    K = len(draft_tokens)
    accepted_tokens = []
    
    for i in range(K):
        token = draft_tokens[i]
        p_target = target_probs[i][token]
        p_draft = draft_probs[i][token]
        
        # Acceptance condition: accept with probability min(1, p_target / p_draft)
        # Equivalently, accept if accept_rand[i] < min(1, p_target / p_draft)
        # Note: handle division by zero if p_draft is 0 (though normally positive)
        ratio = p_target / p_draft if p_draft > 0 else float('inf')
        
        if accept_rand[i] < min(1.0, ratio):
            accepted_tokens.append(token)
        else:
            # Rejection at position i: sample from adjusted distribution
            target_p = np.array(target_probs[i])
            draft_p = np.array(draft_probs[i])
            
            # Adjusted distribution: max(0, target - draft)
            adjusted = np.maximum(0.0, target_p - draft_p)
            sum_adj = np.sum(adjusted)
            
            if sum_adj > 0:
                adjusted = adjusted / sum_adj
            else:
                # Fallback to target distribution if difference sums to 0
                adjusted = target_p / np.sum(target_p)
                
            # Inverse CDF sampling using sample_rand[i]
            u = sample_rand[i]
            cum_prob = 0.0
            sampled_token = len(adjusted) - 1
            for idx, prob in enumerate(adjusted):
                cum_prob += prob
                if u < cum_prob:
                    sampled_token = idx
                    break
                    
            accepted_tokens.append(sampled_token)
            return accepted_tokens

    # If all K tokens are accepted, sample a bonus token from target_probs[K]
    target_p = np.array(target_probs[K])
    sum_p = np.sum(target_p)
    if sum_p > 0:
        target_p = target_p / sum_p
        
    u = sample_rand[K]
    cum_prob = 0.0
    bonus_token = len(target_p) - 1
    for idx, prob in enumerate(target_p):
        cum_prob += prob
        if u < cum_prob:
            bonus_token = idx
            break
            
    accepted_tokens.append(bonus_token)
    return accepted_tokens