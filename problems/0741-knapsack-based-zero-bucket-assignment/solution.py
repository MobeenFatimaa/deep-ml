def zero_bucket_assignment(sizes: list, num_ranks: int) -> list:
    """
    Assign whole parameter matrices to ranks to balance loads.

    Args:
        sizes: list of parameter matrix sizes (number of elements).
        num_ranks: number of ZeRO ranks.

    Returns:
        List of length num_ranks with the total size assigned to each rank.
    """
    if num_ranks <= 0:
        return []
    
    # Initialize per-rank total loads
    rank_loads = [0] * num_ranks

    if not sizes:
        return rank_loads

    # Pair each size with its original index to break ties deterministically
    indexed_sizes = [(size, i) for i, size in enumerate(sizes)]
    
    # Sort descending by size; ties broken by original index ascending
    indexed_sizes.sort(key=lambda x: (-x[0], x[1]))

    # Greedily assign each matrix to the rank with the smallest current load
    for size, _ in indexed_sizes:
        min_rank = 0
        min_load = rank_loads[0]
        
        for r in range(1, num_ranks):
            if rank_loads[r] < min_load:
                min_load = rank_loads[r]
                min_rank = r
                
        rank_loads[min_rank] += size

    return rank_loads