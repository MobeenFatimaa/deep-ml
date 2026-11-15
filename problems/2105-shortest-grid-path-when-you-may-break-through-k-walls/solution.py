from collections import deque

def shortest_path_with_breaks(grid, k):
    """Return the fewest steps from top-left to bottom-right breaking at most k walls, or -1."""
    R = len(grid)
    C = len(grid[0])
    
    # Base case for 1x1 grid
    if R == 1 and C == 1:
        return 0
        
    # max_breaks[r][c] stores the maximum k_left we've had when visiting (r, c)
    # This pruning prevents exploring redundant or suboptimal paths.
    max_breaks = [[-1] * C for _ in range(R)]
    
    # Queue stores tuples of (row, col, breaks_remaining, steps)
    queue = deque()
    
    initial_k = min(k, R * C)  # Cap k to maximum possible walls
    queue.append((0, 0, initial_k, 0))
    max_breaks[0][0] = initial_k
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c, k_left, steps = queue.popleft()
        
        # If we reached the bottom-right corner, return the step count
        if r == R - 1 and c == C - 1:
            return steps
            
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                cell_val = grid[nr][nc]
                new_k = k_left - cell_val  # Deduct 1 break if it's a wall (1)
                
                # Check if we have enough breaks left and if this path is better
                if new_k >= 0 and new_k > max_breaks[nr][nc]:
                    max_breaks[nr][nc] = new_k
                    queue.append((nr, nc, new_k, steps + 1))
                    
    return -1