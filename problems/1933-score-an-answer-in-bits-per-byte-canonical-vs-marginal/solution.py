import math
import numpy as np


def fit_bigram(sequences, vocab, k=0.01):
    """(start, trans) with add-k smoothing."""
    V = len(vocab)
    vocab_to_idx = {tok: i for i, tok in enumerate(vocab)}
    
    start_counts = np.full(V, k, dtype=float)
    trans_counts = np.full((V, V), k, dtype=float)
    
    for seq in sequences:
        if not seq:
            continue
        t0 = seq[0]
        if t0 in vocab_to_idx:
            start_counts[vocab_to_idx[t0]] += 1.0
            
        for i in range(len(seq) - 1):
            u = seq[i]
            v = seq[i+1]
            if u in vocab_to_idx and v in vocab_to_idx:
                trans_counts[vocab_to_idx[u], vocab_to_idx[v]] += 1.0
                
    start = start_counts / start_counts.sum()
    trans = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    return start, trans


def canonical_bpb(prompt_tokens, answer_tokens, vocab, start, trans):
    """Bits per byte of the answer tokens after the prompt tokens."""
    if not answer_tokens:
        raise ValueError("Answer has no tokens.")
    
    total_bytes = sum(len(t) for t in answer_tokens)
    if total_bytes == 0:
        raise ValueError("Answer has no bytes.")
        
    vocab_to_idx = {tok: i for i, tok in enumerate(vocab)}
    
    log_prob = 0.0
    for i, token in enumerate(answer_tokens):
        if token not in vocab_to_idx:
            raise ValueError(f"Token {token} not found in vocabulary.")
        
        idx = vocab_to_idx[token]
        if i == 0:
            if not prompt_tokens:
                p = start[idx]
            else:
                last_prompt = prompt_tokens[-1]
                if last_prompt not in vocab_to_idx:
                    raise ValueError(f"Prompt token {last_prompt} not found in vocabulary.")
                p = trans[vocab_to_idx[last_prompt], idx]
        else:
            prev_token = answer_tokens[i - 1]
            p = trans[vocab_to_idx[prev_token], idx]
            
        if p <= 0.0:
            return float('inf')
        log_prob += math.log2(p)
        
    return float(-log_prob / total_bytes)


def marginal_bpb(prompt, answer, vocab, start, trans):
    """-log2(P(prompt + answer) / P(prompt)) / len(answer), with prefix probabilities."""
    if not answer:
        raise ValueError("Answer is empty bytes.")
    
    vocab_bytes = [vocab[i] for i in range(len(vocab))]
    vocab_lens = [len(vb) for vb in vocab_bytes]
    
    def get_prefix_prob(target_bytes):
        L = len(target_bytes)
        if L == 0:
            return 1.0
            
        memo = {}
        
        def dfs(pos, prev_idx):
            state = (pos, prev_idx)
            if state in memo:
                return memo[state]
                
            total = 0.0
            for i, vb in enumerate(vocab_bytes):
                l_t = vocab_lens[i]
                if prev_idx is None:
                    p = start[i]
                else:
                    p = trans[prev_idx, i]
                    
                if p <= 0.0:
                    continue
                    
                if pos + l_t < L:
                    if vb == target_bytes[pos : pos + l_t]:
                        total += p * dfs(pos + l_t, i)
                else:
                    if vb[:L - pos] == target_bytes[pos : L]:
                        total += p
                        
            memo[state] = total
            return total
            
        return dfs(0, None)

    p_prompt = get_prefix_prob(prompt)
    full_target = prompt + answer
    p_full = get_prefix_prob(full_target)
    
    if p_full == 0.0:
        raise ValueError("P(prompt + answer) = 0")
        
    if p_prompt <= 0.0:
        raise ValueError("P(prompt) = 0")
        
    ratio = p_full / p_prompt
    if ratio <= 0.0:
        raise ValueError("Invalid probability ratio <= 0")
        
    return float(-math.log2(ratio) / len(answer))