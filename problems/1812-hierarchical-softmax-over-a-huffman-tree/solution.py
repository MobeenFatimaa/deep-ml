import numpy as np


def build_huffman_tree(freqs):
    """
    Builds a Huffman tree over C classes and returns the paths from root to each leaf.
    
    Args:
        freqs: list or array of frequencies for each class (length C >= 2)
        
    Returns:
        paths: list of length C, where paths[c] is a list of (node_index, bit) tuples 
               from root to leaf c (bit=0 for left, bit=1 for right; node_index = id - C).
    """
    C = len(freqs)
    # Node structure: (freq, id, is_leaf, left_child, right_child)
    # Leaves have ids 0 .. C-1
    # Internal nodes have ids C, C+1, ... in creation order (node_index = id - C)
    nodes = []
    for i, f in enumerate(freqs):
        nodes.append({
            'freq': float(f),
            'id': i,
            'is_leaf': True,
            'left': None,
            'right': None
        })
        
    next_internal_id = C
    
    # Priority queue / active node list sorted by (freq, id)
    # Since C is typically small, we can maintain a sorted list or list of active nodes
    active = list(range(C))
    
    while len(active) > 1:
        # Sort active nodes by (freq, id) to ensure tie-breaking by smaller id
        active.sort(key=lambda idx: (nodes[idx]['freq'], nodes[idx]['id']))
        
        # Remove the two nodes with smallest frequency
        left_idx = active.pop(0)
        right_idx = active.pop(0)
        
        new_node = {
            'freq': nodes[left_idx]['freq'] + nodes[right_idx]['freq'],
            'id': next_internal_id,
            'is_leaf': False,
            'left': left_idx,
            'right': right_idx
        }
        
        nodes.append(new_node)
        active.append(len(nodes) - 1)
        next_internal_id += 1
        
    root_idx = active[0]
    
    # Traverse tree to extract paths for each leaf
    paths = [[] for _ in range(C)]
    
    def dfs(curr_idx, current_path):
        curr = nodes[curr_idx]
        if curr['is_leaf']:
            # Leaf node, store path
            paths[curr['id']] = current_path
        else:
            # Internal node: node_index = id - C
            node_index = curr['id'] - C
            dfs(curr['left'], current_path + [(node_index, 0)])
            dfs(curr['right'], current_path + [(node_index, 1)])
            
    dfs(root_idx, [])
    return paths


def _sigmoid(x):
    # Numerically stable sigmoid
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _log_sigmoid(x):
    # Numerically stable log-sigmoid: log(sigmoid(x)) = -softplus(-x) = -log1p(exp(-x)) for x >= 0
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))


def hsoftmax_proba(h, node_weights, paths):
    """
    Computes the length-C vector of leaf probabilities using the Huffman hierarchy.
    """
    C = len(paths)
    probs = np.zeros(C)
    
    # Precompute sigmoid probabilities for all internal nodes
    # node_weights has shape (C-1, H), row k corresponds to internal node k
    logits = node_weights @ h  # shape (C-1,)
    sigmoids = _sigmoid(logits)
    
    for c, path in enumerate(paths):
        prob = 1.0
        for node_idx, bit in path:
            p_left = sigmoids[node_idx]
            if bit == 0:
                prob *= p_left
            else:
                prob *= (1.0 - p_left)
        probs[c] = prob
        
    return probs


def log_prob(h, y, node_weights, paths):
    """
    Computes log p(y | h) for a single class y, touching only nodes on its path.
    """
    path = paths[y]
    log_p = 0.0
    
    for node_idx, bit in path:
        logit = np.dot(node_weights[node_idx], h)
        # log_sigmoid(logit) for bit=0 (left), log_sigmoid(-logit) for bit=1 (right)
        if bit == 0:
            log_p += _log_sigmoid(logit)
        else:
            log_p += _log_sigmoid(-logit)
            
    return float(log_p)