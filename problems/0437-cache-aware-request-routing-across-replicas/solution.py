def get_longest_prefix_match(req_tokens: list, cached_prefixes: list) -> int:
    """Computes the maximum matching prefix length between request tokens

    and any cached prefix in a replica.
    """
    max_match = 0
    for prefix in cached_prefixes:
        match_len = 0
        for r_tok, p_tok in zip(req_tokens, prefix):
            if r_tok == p_tok:
                match_len += 1
            else:
                break
        if match_len > max_match:
            max_match = match_len
    return max_match


def cache_aware_route(
    replicas: list, requests: list, alpha: float = 0.7, beta: float = 0.3
) -> dict:
    """Route inference requests to model replicas based on cache affinity and load.

    Args:
        replicas: list of dicts with keys: 'cached_prefixes': list of lists
          (cached token sequences) 'active_requests': int (current load)
          'max_capacity': int (max concurrent requests)
        requests: list of dicts with key: 'tokens': list of ints (token sequence)
        alpha: weight for cache affinity (default 0.7)
        beta: weight for load penalty (default 0.3)

    Returns:
        dict with 'assignments', 'cache_hit_rates', 'avg_cache_hit_rate',
        'load_distribution'
    """
    if not requests or not replicas:
        return {
            "assignments": [],
            "cache_hit_rates": [],
            "avg_cache_hit_rate": 0.0,
            "load_distribution": [
                r.get("active_requests", 0) for r in replicas
            ],
        }

    # Deep copy active_requests tracking so input dicts aren't mutated permanently
    replica_loads = [r["active_requests"] for r in replicas]
    max_capacities = [r["max_capacity"] for r in replicas]
    cached_prefixes_list = [r["cached_prefixes"] for r in replicas]

    assignments = []
    cache_hit_rates = []

    for req in requests:
        tokens = req["tokens"]
        req_len = len(tokens)

        # Candidate replicas that are not full
        available_indices = [
            i
            for i, (load, cap) in enumerate(zip(replica_loads, max_capacities))
            if load < cap
        ]

        # If all replicas are full, consider all replicas
        if not available_indices:
            candidate_indices = list(range(len(replicas)))
        else:
            candidate_indices = available_indices

        best_replica_idx = -1
        best_score = float("-inf")
        best_hit_rate = 0.0

        for idx in candidate_indices:
            load = replica_loads[idx]
            cap = max_capacities[idx]
            cached_prefixes = cached_prefixes_list[idx]

            match_len = get_longest_prefix_match(tokens, cached_prefixes)
            hit_rate = match_len / req_len if req_len > 0 else 0.0
            load_ratio = load / cap if cap > 0 else 0.0

            score = alpha * hit_rate - beta * load_ratio

            if score > best_score:
                best_score = score
                best_replica_idx = idx
                best_hit_rate = hit_rate

        # Assign request to best replica and increment active load
        assignments.append(best_replica_idx)
        hit_rate_rounded = round(best_hit_rate, 2)
        cache_hit_rates.append(hit_rate_rounded)
        replica_loads[best_replica_idx] += 1

    avg_hit_rate = (
        round(sum(cache_hit_rates) / len(cache_hit_rates), 2)
        if cache_hit_rates
        else 0.0
    )

    return {
        "assignments": assignments,
        "cache_hit_rates": cache_hit_rates,
        "avg_cache_hit_rate": avg_hit_rate,
        "load_distribution": replica_loads,
    }