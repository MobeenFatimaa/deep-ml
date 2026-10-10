import heapq

def alien_order(words: list[str]) -> str:
    """
    Returns a string of distinct characters ordered consistently with the alien language,
    resolving ties by choosing the alphabetically smallest available character.
    Returns an empty string if invalid.
    """
    # Step 1: Collect all unique characters
    chars = set()
    for word in words:
        for c in word:
            chars.add(c)
            
    adj = {c: set() for c in chars}
    in_degree = {c: 0 for c in chars}
    
    # Step 2: Build graph from adjacent word comparisons
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        min_len = min(len(w1), len(w2))
        
        # Check prefix rule: longer word cannot precede its own prefix
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
            
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in adj[w1[j]]:
                    adj[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break

    # Step 3: Kahn's algorithm using a min-heap for lexicographically smallest order
    heap = [c for c in chars if in_degree[c] == 0]
    heapq.heapify(heap)
    
    res = []
    while heap:
        c = heapq.heappop(heap)
        res.append(c)
        for neighbor in adj[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)
                
    # Step 4: Check for cycles (if not all characters were included)
    if len(res) != len(chars):
        return ""
        
    return "".join(res)