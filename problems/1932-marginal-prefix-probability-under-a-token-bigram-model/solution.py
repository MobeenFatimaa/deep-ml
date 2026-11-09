import numpy as np

def prefix_probability(s, vocab, start, trans):
    """P(the model's byte stream starts with s), summed over every tokenization."""
    if len(s) == 0:
        return 1.0
    
    L = len(s)
    V = len(vocab)
    start = np.asarray(start, dtype=float)
    trans = np.asarray(trans, dtype=float)
    
    dp = np.zeros((L + 1, V), dtype=float)
    total_prob = 0.0
    
    for u in range(V):
        tok = vocab[u]
        l_tok = len(tok)
        
        if l_tok >= L:
            if tok[:L] == s:
                total_prob += start[u]
        else:
            if tok == s[:l_tok]:
                dp[l_tok, u] += start[u]
                
    for p in range(1, L):
        for u in range(V):
            prob_u = dp[p, u]
            if prob_u == 0.0:
                continue
            
            for v in range(V):
                tok_v = vocab[v]
                l_v = len(tok_v)
                p_next = p + l_v
                
                if p_next >= L:
                    if tok_v[:L - p] == s[p:L]:
                        total_prob += prob_u * trans[u, v]
                else:
                    if tok_v == s[p:p_next]:
                        dp[p_next, v] += prob_u * trans[u, v]
                        
    return float(total_prob)


def next_byte_distribution(s, vocab, start, trans):
    """{byte: P(next byte = byte | stream starts with s)} for bytes with positive probability."""
    p_s = prefix_probability(s, vocab, start, trans)
    if p_s == 0.0:
        raise ValueError("Prefix has probability 0")
        
    L = len(s)
    V = len(vocab)
    start = np.asarray(start, dtype=float)
    trans = np.asarray(trans, dtype=float)
    
    dp = np.zeros((L + 1, V), dtype=float)
    
    # Also track DP states that terminate exactly at length L (p = L)
    for u in range(V):
        tok = vocab[u]
        l_tok = len(tok)
        if l_tok <= L:
            if tok == s[:l_tok]:
                if l_tok == L:
                    dp[L, u] += start[u]
                else:
                    dp[l_tok, u] += start[u]
                
    for p in range(1, L):
        for u in range(V):
            prob_u = dp[p, u]
            if prob_u == 0.0:
                continue
            for v in range(V):
                tok_v = vocab[v]
                l_v = len(tok_v)
                p_next = p + l_v
                if p_next <= L:
                    if tok_v == s[p:p_next]:
                        dp[p_next, v] += prob_u * trans[u, v]
                        
    b_probs = np.zeros(256, dtype=float)
    
    # Case A: single tokens of length >= L+1 matching s + b
    for u in range(V):
        tok = vocab[u]
        if len(tok) >= L + 1:
            if tok[:L] == s:
                b_probs[tok[L]] += start[u]
                
    # Case B: multi-token sequences via dp states
    for p in range(1, L + 1):
        for u in range(V):
            prob_u = dp[p, u]
            if prob_u == 0.0:
                continue
            for v in range(V):
                tok_v = vocab[v]
                l_v = len(tok_v)
                if p < L:
                    if l_v >= L + 1 - p:
                        if tok_v[:L - p] == s[p:L]:
                            b = tok_v[L - p]
                            b_probs[b] += prob_u * trans[u, v]
                else:  # p == L
                    if l_v >= 1:
                        b = tok_v[0]
                        b_probs[b] += prob_u * trans[u, v]
                        
    dist = {}
    for b_val in range(256):
        prob = b_probs[b_val]
        if prob > 0.0:
            dist[b_val] = float(prob / p_s)
            
    return dist