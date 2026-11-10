import numpy as np

def dense_ffn(x, W1, W2, activation):
    """Computes the forward pass of the dense FFN."""
    h = x @ W1
    if activation == 'relu':
        phi = np.maximum(0, h)
    elif activation == 'relu2':
        phi = np.maximum(0, h) ** 2
    else:
        raise ValueError(f"Unknown activation {activation}")
    return phi @ W2

def upcycle(W1, W2, granularity, expansion_rate, top_k, activation, pre_softmax=False):
    """Upcycles a dense FFN into fine-grained experts with appropriate scaling."""
    d, H = W1.shape
    G = granularity
    H_slice = H // G
    E = G * expansion_rate
    
    if pre_softmax:
        a = (E * G) / top_k
    else:
        a = float(G)
        
    if activation == 'relu':
        c = a ** 0.5
    elif activation == 'relu2':
        c = a ** (1.0 / 3.0)
    else:
        raise ValueError(f"Unknown activation {activation}")
        
    W1e = np.zeros((E, d, H_slice))
    W2e = np.zeros((E, H_slice, d))
    
    for e in range(E):
        s = e % G
        W1e[e] = W1[:, s * H_slice : (s + 1) * H_slice] * c
        W2e[e] = W2[s * H_slice : (s + 1) * H_slice, :] * c
        
    return {
        "W1": W1e,
        "W2": W2e,
        "activation_scale": float(a),
        "weight_scale": float(c)
    }

def moe_ffn(x, W1e, W2e, W_router, top_k, activation, pre_softmax=False):
    """Computes the forward pass of the MoE FFN layer."""
    T, d = x.shape
    E = W1e.shape[0]
    if W_router is None:
        W_router = np.zeros((d, E))
        
    logits = x @ W_router  # (T, E)
    
    top_indices = np.empty((T, top_k), dtype=int)
    weights = np.empty((T, top_k), dtype=float)
    
    if pre_softmax:
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
    for t in range(T):
        row = logits[t]
        # Sort descending by logit value, then ascending by expert index for ties
        sorted_idxs = sorted(range(E), key=lambda i: (-row[i], i))
        sel = sorted_idxs[:top_k]
        top_indices[t] = sel
        
        if pre_softmax:
            weights[t] = probs[t, sel]
        else:
            sel_logits = row[sel]
            max_l = np.max(sel_logits)
            exp_sel = np.exp(sel_logits - max_l)
            weights[t] = exp_sel / np.sum(exp_sel)
            
    out = np.zeros((T, d))
    H_slice = W1e.shape[2]
    
    for t in range(T):
        xt = x[t]
        for j in range(top_k):
            e = top_indices[t, j]
            w = weights[t, j]
            
            w1 = W1e[e]
            w2 = W2e[e]
            
            h = xt @ w1
            if activation == 'relu':
                phi = np.maximum(0, h)
            elif activation == 'relu2':
                phi = np.maximum(0, h) ** 2
            else:
                raise ValueError(f"Unknown activation {activation}")
                
            out[t] += w * (phi @ w2)
            
    return out

def parity_gap(x, W1, W2, granularity, expansion_rate, top_k, activation, pre_softmax=False, W_router=None):
    """Upcycles the dense FFN, runs the MoE, and computes the maximum absolute parity gap."""
    up = upcycle(W1, W2, granularity, expansion_rate, top_k, activation, pre_softmax=pre_softmax)
    moe_out = moe_ffn(x, up['W1'], up['W2'], W_router, top_k, activation, pre_softmax=pre_softmax)
    dense_out = dense_ffn(x, W1, W2, activation)
    gap = np.max(np.abs(moe_out - dense_out))
    return round(float(gap), 6)