import numpy as np

def expected_recolor_steps(counts, target_colors=1):
    """Return the expected steps until at most target_colors colours remain."""
    # Filter out zero counts and sort in descending order to form the canonical state
    counts = tuple(sorted([c for c in counts if c > 0], reverse=True))
    N = sum(counts)
    
    if N == 0 or len(counts) <= target_colors:
        return 0.0

    # Generate all integer partitions of N
    def gen(n, max_val):
        if n == 0:
            yield ()
            return
        for i in range(min(n, max_val), 0, -1):
            for p in gen(n - i, i):
                yield (i,) + p
                
    partitions = list(gen(N, N))
    part_to_idx = {p: idx for idx, p in enumerate(partitions)}
    
    M = len(partitions)
    A = np.zeros((M, M), dtype=np.float64)
    b = np.zeros(M, dtype=np.float64)
    
    # Build the linear system Ax = b
    for idx, s in enumerate(partitions):
        k = len(s)
        if k <= target_colors:
            # Absorbing states (at or below target colors)
            A[idx, idx] = 1.0
            b[idx] = 0.0
        else:
            # W(s) is the sum of transition weights out of state s
            W = N * N - sum(c * c for c in s)
            A[idx, idx] = W
            b[idx] = N * (N - 1)
            
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    c_i = s[i]
                    c_j = s[j]
                    weight = c_i * c_j
                    
                    # Compute next state after drawing from i and repointing j
                    new_s = list(s)
                    new_s[j] -= 1
                    new_s[i] += 1
                    new_s = tuple(sorted([x for x in new_s if x > 0], reverse=True))
                    
                    next_idx = part_to_idx[new_s]
                    A[idx, next_idx] -= weight
                    
    # Solve the system of linear equations
    x = np.linalg.solve(A, b)
    
    input_idx = part_to_idx.get(counts)
    if input_idx is None:
        return 0.0
        
    return float(x[input_idx])