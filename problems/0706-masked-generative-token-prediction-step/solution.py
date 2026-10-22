import numpy as np
import math

def maskgit_decode_step(tokens, logits, step: int, total_steps: int, mask_id: int) -> np.ndarray:
    """
    Perform a single step of parallel masked token decoding.

    Args:
        tokens: 1D array-like of length N containing current token ids (mask_id for masked positions)
        logits: 2D array-like of shape (N, vocab_size) of model logits
        step: current step index (0-indexed)
        total_steps: total number of refinement steps
        mask_id: integer id used to indicate a masked position

    Returns:
        numpy array of length N with updated token ids.
    """
    tokens = np.array(tokens)
    logits = np.array(logits)
    N = len(tokens)
    
    # Compute probabilities via softmax along the vocabulary dimension
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    preds = np.argmax(logits, axis=-1)
    confs = np.max(probs, axis=-1)
    
    # Replace currently masked positions with argmax predictions
    updated_tokens = tokens.copy()
    mask_indices = (tokens == mask_id)
    updated_tokens[mask_indices] = preds[mask_indices]
    
    # Determine number of positions that should remain masked via cosine schedule
    num_masked = math.floor(N * math.cos((math.pi / 2.0) * (step + 1) / total_steps))
    
    if num_masked > 0:
        # Treat confidence of already unmasked positions as effectively infinite
        sort_confs = confs.copy()
        sort_confs[~mask_indices] = np.inf
        
        # Select lowest-confidence positions among eligible ones
        sorted_indices = np.argsort(sort_confs)
        remask_indices = sorted_indices[:num_masked]
        
        updated_tokens[remask_indices] = mask_id
        
    return updated_tokens