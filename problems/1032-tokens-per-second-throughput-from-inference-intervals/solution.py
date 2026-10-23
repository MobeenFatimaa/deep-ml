def compute_throughput(intervals):
    """
    Compute step-wise, cumulative, and overall tokens-per-second.

    Args:
        intervals: list of (tokens_processed, elapsed_seconds) tuples.

    Returns:
        dict with keys 'step_tps', 'cumulative_tps', 'overall_tps'.
    """
    if not intervals:
        return {'step_tps': [], 'cumulative_tps': [], 'overall_tps': 0.0}

    step_tps = []
    cumulative_tps = []
    
    total_tokens = 0
    total_seconds = 0.0

    for tokens, seconds in intervals:
        if seconds <= 0:
            raise ValueError("elapsed_seconds must be greater than 0.")
        if tokens < 0:
            raise ValueError("tokens_processed cannot be negative.")
        
        step_val = tokens / seconds
        step_tps.append(round(step_val, 2))
        
        total_tokens += tokens
        total_seconds += seconds
        cum_val = total_tokens / total_seconds
        cumulative_tps.append(round(cum_val, 2))
        
    overall_tps = round(total_tokens / total_seconds, 2) if total_seconds > 0 else 0.0

    return {
        'step_tps': step_tps,
        'cumulative_tps': cumulative_tps,
        'overall_tps': overall_tps
    }