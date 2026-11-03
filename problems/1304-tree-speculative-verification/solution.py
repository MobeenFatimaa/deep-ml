def tree_speculative_verify(draft_tokens, parents, target_next):
    """
    Walk a draft tree and return the accepted token id path (root excluded).
    """
    n = len(draft_tokens)
    children = [[] for _ in range(n)]
    root = -1
    
    # Build adjacency list for children (automatically in increasing order of node indices)
    for i in range(n):
        p = parents[i]
        if p == -1:
            root = i
        else:
            children[p].append(i)
            
    accepted = []
    current = root
    
    # Traverse the tree greedily matching target_next
    while current < n:
        target = target_next[current]
        next_node = -1
        
        for j in children[current]:
            if draft_tokens[j] == target:
                next_node = j
                break
        
        if next_node != -1:
            accepted.append(draft_tokens[next_node])
            current = next_node
        else:
            break
            
    return accepted