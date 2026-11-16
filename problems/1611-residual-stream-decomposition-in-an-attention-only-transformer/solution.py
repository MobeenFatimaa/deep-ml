import numpy as np


def attn_only_forward(tokens, W):
    """
    Returns (final residual (T, d), writes dict with "embed" and ("head", layer, head) keys).
    """
    W_E = W["W_E"]
    W_pos = W["W_pos"]
    layers = W["layers"]
    
    T = len(tokens)
    # Initial residual stream
    x = W_E[tokens] + W_pos[:T]
    writes = {"embed": x.copy()}
    
    current_res = x.copy()
    
    for l_idx, layer in enumerate(layers):
        layer_delta = np.zeros_like(current_res)
        for h_idx, head in enumerate(layer):
            W_Q = head["W_Q"]
            W_K = head["W_K"]
            W_V = head["W_V"]
            W_O = head["W_O"]
            
            d_h = W_Q.shape[1]
            
            # Compute Q, K, V from the current layer-input residual
            Q = np.dot(current_res, W_Q)
            K = np.dot(current_res, W_K)
            V = np.dot(current_res, W_V)
            
            # Attention scores and causal masking
            scores = np.matmul(Q, K.T) / np.sqrt(d_h)
            mask = np.triu(np.ones((T, T)), 1)
            scores = np.where(mask == 1, -1e9, scores)
            
            # Softmax over source axis
            max_scores = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            
            # Head output
            head_out = np.matmul(np.matmul(weights, V), W_O)
            
            writes[("head", l_idx, h_idx)] = head_out
            layer_delta += head_out
            
        # Every component writes, nothing overwrites: add layer delta to residual
        current_res = current_res + layer_delta
        
    return current_res, writes


def direct_logit_attribution(writes, W_U, tok_a, tok_b, pos=-1):
    """
    Per-component contribution to the logit difference u_a - u_b at position pos.
    """
    # Unembedding vector difference for tok_a and tok_b
    u_diff = W_U[:, tok_a] - W_U[:, tok_b]
    
    dla = {}
    for k, w in writes.items():
        # Dot product of unembedding diff with the component's residual write vector at position pos
        contribution = float(np.dot(u_diff, w[pos]))
        dla[k] = round(contribution, 6)
        
    return dla