import numpy as np

def self_speculative_decode(H_true, target_tokens, E, W_out, b_out, Wh, Wx, max_draft):
    """
    Simulate greedy self-speculative decoding with latent-rollout drafting.

    Args:
        H_true:        (T, D) verifier hidden states
        target_tokens: (T,) verifier greedy tokens; target_tokens[p] decodes from H_true[p]
        E:             (V, D) token embeddings
        W_out, b_out:  (D, V), (V,) output head
        Wh, Wx:        (D, D) transition tanh(h @ Wh + e @ Wx)
        max_draft:     maximum drafted tokens per cycle

    Returns:
        (num_cycles, mean_accepted) with mean_accepted a float
    """
    T = len(H_true)
    if T == 0:
        return 0, 0.0

    pos = 0
    accepted_counts = []

    while pos < T:
        # 1. The verifier's own token at pos is always accepted (base token of the cycle)
        curr_token = target_tokens[pos]
        
        # 2. Initialize latent state h and draft up to max_draft tokens
        h = np.tanh(H_true[pos] @ Wh + E[curr_token] @ Wx)
        
        drafted_tokens = []
        for i in range(max_draft):
            if pos + 1 + i >= T:
                break
            
            # Greedy decoding from current latent state h (argmax with default tie-breaking)
            logits = h @ W_out + b_out
            next_token = np.argmax(logits)
            drafted_tokens.append(next_token)
            
            # Update latent state for the next step
            h = np.tanh(h @ Wh + E[next_token] @ Wx)

        # 3. Find the longest matching prefix against target_tokens
        target_slice = target_tokens[pos + 1 : pos + 1 + len(drafted_tokens)]
        n = 0
        for d_tok, t_tok in zip(drafted_tokens, target_slice):
            if d_tok == t_tok:
                n += 1
            else:
                break

        accepted_counts.append(n)
        
        # 4. Advance position by 1 (for the base token) + n (for the accepted drafted tokens)
        pos += 1 + n

    num_cycles = len(accepted_counts)
    mean_accepted = float(np.mean(accepted_counts)) if num_cycles > 0 else 0.0

    return num_cycles, mean_accepted