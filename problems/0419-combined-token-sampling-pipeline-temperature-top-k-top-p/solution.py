import numpy as np

def combined_sampling(logits: list[float], temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42) -> dict:
    """
    Apply a combined sampling pipeline to raw logits.
    
    Args:
        logits: Raw unnormalized scores for each token
        temperature: Scaling factor for logit sharpness (0 = greedy)
        top_k: Number of top tokens to keep (0 = disabled)
        top_p: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
        seed: Random seed for reproducible sampling
    
    Returns:
        Dict with 'probabilities' (list of floats rounded to 4 decimals)
        and 'sampled_token' (int index of chosen token)
    """
    logits_arr = np.array(logits, dtype=float)
    vocab_size = len(logits_arr)
    
    # Step 1: Temperature Scaling / Greedy Decoding
    if temperature <= 0:
        sampled_token = int(np.argmax(logits_arr))
        probs = np.zeros(vocab_size)
        probs[sampled_token] = 1.0
        return {
            'probabilities': [round(float(p), 4) for p in probs],
            'sampled_token': sampled_token
        }
    
    scaled_logits = logits_arr / temperature
    
    # Step 2: Top-k Filtering
    if 0 < top_k < vocab_size:
        # Find the threshold value for the top_k elements
        top_k_indices = np.argsort(scaled_logits)[-top_k:]
        mask = np.ones(vocab_size, dtype=bool)
        mask[top_k_indices] = False
        scaled_logits[mask] = -float('inf')
        
    # Step 3: Convert to Probabilities (Softmax with numerical stability)
    max_logit = np.max(scaled_logits[np.isfinite(scaled_logits)])
    exp_logits = np.exp(scaled_logits - max_logit)
    # Handle case where all are -inf
    sum_exp = np.sum(exp_logits)
    if sum_exp == 0 or np.isnan(sum_exp):
        probs = np.ones(vocab_size) / vocab_size
    else:
        probs = exp_logits / sum_exp
        
    # Step 4: Top-p (Nucleus) Filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        cumulative_probs = np.cumsum(probs[sorted_indices])
        
        # Find indices to remove (cumulative prob strictly greater than top_p after the first element)
        # Standard nucleus sampling keeps the smallest set whose cumulative prob >= top_p
        exceed_indices = np.where(cumulative_probs > top_p)[0]
        if len(exceed_indices) > 0:
            # Keep the first element that crosses the threshold, mask subsequent ones
            cutoff_idx = exceed_indices[0]
            indices_to_remove = sorted_indices[cutoff_idx + 1:]
            probs[indices_to_remove] = 0.0
            
        # Re-normalize the distribution
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs /= sum_probs
        else:
            probs = np.ones(vocab_size) / vocab_size

    # Step 5: Sampling
    rng = np.random.default_rng(seed)
    sampled_token = int(rng.choice(vocab_size, p=probs))
    
    return {
        'probabilities': [round(float(p), 4) for p in probs],
        'sampled_token': sampled_token
    }