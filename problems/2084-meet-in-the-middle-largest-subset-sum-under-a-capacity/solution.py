import bisect

def max_subset_sum(weights, capacity):
    """Return the largest subset sum of weights that does not exceed capacity."""
    if not weights:
        return 0
        
    n = len(weights)
    mid = n // 2
    left_weights = weights[:mid]
    right_weights = weights[mid:]
    
    def generate_sums(arr):
        sums = [0]
        for w in arr:
            sums.extend([s + w for s in sums])
        return sums
        
    left_sums = generate_sums(left_weights)
    right_sums = generate_sums(right_weights)
    
    # Filter right sums to be <= capacity and sort them
    right_sums = sorted([s for s in right_sums if s <= capacity])
    
    max_sum = 0
    
    for s1 in left_sums:
        if s1 > capacity:
            continue
        remaining = capacity - s1
        # Find the largest s2 in right_sums such that s2 <= remaining
        idx = bisect.bisect_right(right_sums, remaining)
        if idx > 0:
            s2 = right_sums[idx - 1]
            if s1 + s2 > max_sum:
                max_sum = s1 + s2
                
    return max_sum