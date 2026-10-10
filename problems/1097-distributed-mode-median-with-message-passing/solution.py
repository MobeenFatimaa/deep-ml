import numpy as np

def distributed_stats(nodes: list[list[int]], lo: int, hi: int) -> tuple[int, float]:
    """
    Compute global mode and median across distributed nodes given an integer range [lo, hi].
    
    Args:
        nodes: List of lists containing integer measurements.
        lo: Minimum possible integer value (inclusive).
        hi: Maximum possible integer value (inclusive).
        
    Returns:
        A tuple of (mode, median), where mode is an int and median is a float.
    """
    range_size = hi - lo + 1
    global_counts = np.zeros(range_size, dtype=np.int64)
    
    # Step 1: Collect compact per-node summaries (histograms) and combine them
    for node_data in nodes:
        if node_data:
            # Use bincount shifted by 'lo' to count frequencies efficiently
            counts = np.bincount(np.array(node_data) - lo, minlength=range_size)
            global_counts += counts

    total_count = np.sum(global_counts)
    
    # Step 2: Compute the global mode
    # Find the value(s) with the maximum frequency. If there's a tie, min() picks the smallest value.
    max_freq = np.max(global_counts)
    mode_candidates = np.where(global_counts == max_freq)[0] + lo
    global_mode = int(np.min(mode_candidates))
    
    # Step 3: Compute the global median using the global frequency array
    # Find the middle rank(s)
    if total_count % 2 == 1:
        # Odd number of elements: single middle element at index total_count // 2 (0-indexed)
        target_rank1 = total_count // 2
        target_rank2 = target_rank1
    else:
        # Even number of elements: two middle elements
        target_rank1 = (total_count // 2) - 1
        target_rank2 = total_count // 2

    current_sum = 0
    val1 = None
    val2 = None
    
    for i, count in enumerate(global_counts):
        if count == 0:
            continue
        next_sum = current_sum + count
        
        # Check if target_rank1 falls in the current value's bin
        if val1 is None and next_sum > target_rank1:
            val1 = lo + i
        # Check if target_rank2 falls in the current value's bin
        if val2 is None and next_sum > target_rank2:
            val2 = lo + i
            break
            
        current_sum = next_sum
        
    global_median = float(val1 + val2) / 2.0
    
    return global_mode, global_median