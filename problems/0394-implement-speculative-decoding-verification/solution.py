import numpy as np

def speculative_decode_verify(draft_tokens: list, draft_probs: list, target_probs: list, coin_flips: list, resample_coin: float) -> list:
    """
    Verify draft tokens using speculative decoding.
    
    Args:
        draft_tokens: List of K drafted token indices
        draft_probs: K x V array, draft model distributions at each position
        target_probs: K x V array, target model distributions at each position
        coin_flips: K random values in [0,1) for acceptance decisions
        resample_coin: Random value in [0,1) for resampling on rejection
    
    Returns:
        List of accepted/resampled token indices
    """
    draft_probs = np.array(draft_probs)
    target_probs = np.array(target_probs)
    K = len(draft_tokens)
    
    accepted_tokens = []
    
    for i in range(K):
        token = draft_tokens[i]
        q = draft_probs[i, token]
        p = target_probs[i, token]
        
        # Acceptance probability ratio min(1, p(x) / q(x))
        if q > 0:
            accept_prob = min(1.0, p / q)
        else:
            accept_prob = 1.0 if p > 0 else 0.0
            
        if coin_flips[i] < accept_prob:
            accepted_tokens.append(token)
        else:
            # Token rejected: compute adjusted distribution and resample
            p_dist = target_probs[i]
            q_dist = draft_probs[i]
            
            # Adjusted distribution: max(0, p - q)
            adjusted = np.maximum(0.0, p_dist - q_dist)
            sum_adj = np.sum(adjusted)
            
            if sum_adj > 0:
                adjusted /= sum_adj
            else:
                adjusted = p_dist / np.sum(p_dist)
                
            # Sample from adjusted distribution using resample_coin with <= condition
            cum_prob = 0.0
            sampled_token = len(adjusted) - 1
            for idx, prob in enumerate(adjusted):
                cum_prob += prob
                if resample_coin <= cum_prob:
                    sampled_token = idx
                    break
                    
            accepted_tokens.append(sampled_token)
            return accepted_tokens
            
    return accepted_tokens