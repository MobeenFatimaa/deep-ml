import numpy as np


def acquisition_scores(probs, method):
    """Computes active learning acquisition scores for each unlabelled point.
    
    probs: shape (S, N, C)
    """
    S, N, C = probs.shape
    # Predictive distribution p_bar_n = 1/S sum_s p(y | x_n, theta_s)
    p_bar = np.mean(probs, axis=0)  # shape (N, C)
    
    if method == "entropy":
        # H(p_bar_n) = - sum_c p_bar_nc log(p_bar_nc)
        # Handle log(0) safely
        p_bar_safe = np.clip(p_bar, 1e-12, 1.0)
        scores = -np.sum(p_bar_safe * np.log(p_bar_safe), axis=1)
        return scores
        
    elif method == "least_confident":
        # 1 - max_c p_bar_n,c
        max_probs = np.max(p_bar, axis=1)
        scores = 1.0 - max_probs
        return scores
        
    elif method == "bald":
        # H(p_bar_n) - 1/S sum_s H(p(y | x_n, theta_s))
        p_bar_safe = np.clip(p_bar, 1e-12, 1.0)
        entropy_p_bar = -np.sum(p_bar_safe * np.log(p_bar_safe), axis=1)
        
        probs_safe = np.clip(probs, 1e-12, 1.0)
        entropy_samples = -np.sum(probs_safe * np.log(probs_safe), axis=2)  # shape (S, N)
        expected_entropy = np.mean(entropy_samples, axis=0)  # shape (N,)
        
        scores = entropy_p_bar - expected_entropy
        return scores
        
    else:
        raise ValueError(f"Unknown acquisition method: {method}")


def top_k(scores, k):
    """Returns the indices of the k largest scores, best first, ties broken by smaller index."""
    scores = np.asarray(scores)
    # Sort by score descending, then by index ascending
    # We can use lexsort or negative score with index
    indices = np.arange(len(scores))
    # Lexsort sorts by last key first, so sort by index ascending, then by -score descending
    sorted_idx = np.lexsort((indices, -scores))
    return sorted_idx[:k].tolist()


def batch_bald_greedy(probs, k):
    """Greedy BatchBALD selection of k points.
    
    a(B U {n}) = H(y_{B U {n}}) - 1/S sum_s sum_{m in B U {n}} H(p(y_m | x_m, theta_s))
    where the joint predictive p(y_B) = 1/S sum_s prod_{m in B} p(y_m | x_m, theta_s)
    summed exactly over all C^{|B|} label configurations.
    """
    S, N, C = probs.shape
    selected = []
    
    for _ in range(k):
        best_score = -np.inf
        best_n = -1
        
        for n in range(N):
            if n in selected:
                continue
                
            current_batch = selected + [n]
            b_size = len(current_batch)
            
            # Compute joint predictive distribution over all label configurations for current_batch
            # We need combinations of labels for all items in current_batch.
            # Total configurations: C^{b_size}
            # Instead of generating explicit cartesian product for large C or b_size, 
            # let's note that C and b_size are typically small enough for exact enumeration,
            # but we can use np.ndindex or recursive products.
            
            # probs for nodes in current_batch: shape (S, b_size, C)
            batch_probs = probs[:, current_batch, :]  # (S, b_size, C)
            
            # To compute joint predictive p(y_B) for each configuration:
            # A configuration is a tuple of labels (y_1, y_2, ..., y_{b_size}) where each y_i in [0, C-1].
            # We can use np.indices or grid.
            if b_size == 1:
                # Joint predictive is just p_bar for that single point
                p_y_B = np.mean(batch_probs[:, 0, :], axis=0)  # shape (C,)
                # H(y_B)
                p_y_B_safe = np.clip(p_y_B, 1e-12, 1.0)
                h_y_B = -np.sum(p_y_B_safe * np.log(p_y_B_safe))
            else:
                # Generate all label combinations of shape (C^b_size, b_size)
                # For efficiency with moderate C and b_size:
                grids = np.indices((C,) * b_size)  # tuple of b_size arrays of shape (C, C, ...)
                # Reshape to (b_size, C^b_size)
                configs = np.reshape(grids, (b_size, -1)).T  # (num_configs, b_size)
                
                # For each sample s, compute probability of each config:
                # batch_probs[s, m, config[m]] for m in range(b_size)
                # Shape of batch_probs: (S, b_size, C)
                # We can index batch_probs: batch_probs[:, np.arange(b_size), configs] -> shape (S, num_configs)
                # Actually, batch_probs[:, range(b_size), configs] needs broadcasting:
                # batch_probs[:, None, np.arange(b_size), configs] -> let's do:
                # batch_probs has shape (S, b_size, C). For a config of shape (num_configs, b_size):
                # We want product over b_size of batch_probs[s, m, config[m]]
                # Let's compute log probabilities to avoid underflow, or direct multiplication if stable.
                
                # batch_probs[:, :, :] is in (0, 1]. Let's use log:
                log_batch_probs = np.log(np.clip(batch_probs, 1e-12, 1.0))  # (S, b_size, C)
                
                # For each config (num_configs, b_size), get log prob for each sample s:
                # log_batch_probs[:, np.arange(b_size), configs] -> shape (S, num_configs, b_size)
                # Sum over b_size -> shape (S, num_configs)
                # But to avoid huge memory if num_configs is large, let's index properly:
                # configs is (num_configs, b_size). 
                # log_batch_probs is (S, b_size, C).
                # We can extract log_probs for each config:
                # log_p_s_config = sum_m log_batch_probs[s, m, config[m]]
                # Let's vectorize:
                # log_batch_probs[:, m, configs[:, m]] has shape (S, num_configs)
                log_p_s_configs = np.zeros((S, configs.shape[0]))
                for m_idx in range(b_size):
                    log_p_s_configs += log_batch_probs[:, m_idx, configs[:, m_idx]]  # (S, num_configs)
                    
                # Exponentiate to get probabilities per sample: p_s_configs = exp(log_p_s_configs) -> (S, num_configs)
                p_s_configs = np.exp(log_p_s_configs)
                
                # Joint predictive p(y_B) = 1/S sum_s p_s(y_B)
                p_y_B = np.mean(p_s_configs, axis=0)  # (num_configs,)
                
                # H(y_B) = - sum_{y_B} p(y_B) log p(y_B)
                p_y_B_safe = np.clip(p_y_B, 1e-12, 1.0)
                h_y_B = -np.sum(p_y_B_safe * np.log(p_y_B_safe))
                
            # Expected conditional entropy: 1/S sum_s sum_{m in B U {n}} H(p(y_m | x_m, theta_s))
            # H(p(y_m | x_m, theta_s)) = - sum_c p(y_m = c | theta_s) log p(y_m = c | theta_s)
            # batch_probs: (S, b_size, C)
            bp_safe = np.clip(batch_probs, 1e-12, 1.0)
            sample_entropies = -np.sum(bp_safe * np.log(bp_safe), axis=2)  # shape (S, b_size)
            expected_cond_entropy = np.mean(np.sum(sample_entropies, axis=1))
            
            score = h_y_B - expected_cond_entropy
            
            if score > best_score:
                best_score = score
                best_n = n
                
        selected.append(best_n)
        
    return selected