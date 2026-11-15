import numpy as np
import heapq

def hnsw_search(vectors, layers, entry_point, query, k, ef):
    """Return the k nearest nodes found as a list of (node_id, rounded squared distance)."""
    curr = entry_point
    d_query = lambda idx: float(np.sum((vectors[idx] - query) ** 2))
    
    # 1. Upper layers (from layers[-1] down to layers[1]): greedy descent
    for l in range(len(layers) - 1, 0, -1):
        best_node = curr
        best_dist = d_query(curr)
        changed = True
        while changed:
            changed = False
            neighbors = layers[l].get(best_node, [])
            for nb in neighbors:
                dist = d_query(nb)
                if dist < best_dist or (dist == best_dist and nb < best_node):
                    best_dist = dist
                    best_node = nb
                    changed = True
        curr = best_node

    # 2. Layer 0: beam search with width e = max(ef, k)
    e = max(ef, k)
    d_curr = d_query(curr)
    
    visited = {curr}
    
    # C: min-heap storing (dist, node_id)
    C = [(d_curr, curr)]
    heapq.heapify(C)
    
    # W_heap: max-heap storing ((-dist, -node_id), dist, node_id) so the worst element is at the root
    W_heap = [((-d_curr, -curr), d_curr, curr)]
    W_set = {curr}
    
    while C:
        # Remove closest candidate from C
        c_dist, c_node = heapq.heappop(C)
        
        # Get farthest node in W
        _, farthest_dist, farthest_node = W_heap[0]
        
        # If c is farther than the farthest node in W, stop
        if (c_dist > farthest_dist) or (c_dist == farthest_dist and c_node > farthest_node):
            break
            
        neighbors = layers[0].get(c_node, [])
        for u in neighbors:
            if u not in visited:
                visited.add(u)
                u_dist = d_query(u)
                
                _, farthest_dist, farthest_node = W_heap[0]
                
                is_closer = (u_dist < farthest_dist) or (u_dist == farthest_dist and u < farthest_node)
                if len(W_set) < e or is_closer:
                    heapq.heappush(C, (u_dist, u))
                    
                    sort_key = (-u_dist, -u)
                    heapq.heappush(W_heap, (sort_key, u_dist, u))
                    W_set.add(u)
                    
                    if len(W_heap) > e:
                        _, worst_dist, worst_node = heapq.heappop(W_heap)
                        W_set.remove(worst_node)
                        
    # Extract results from W and sort in increasing (δ, i) order
    w_elements = [(dist, node_id) for _, dist, node_id in W_heap]
    w_elements.sort(key=lambda x: (x[0], x[1]))
    
    result = []
    for dist, node_id in w_elements[:k]:
        result.append((node_id, round(dist, 4)))
        
    return result