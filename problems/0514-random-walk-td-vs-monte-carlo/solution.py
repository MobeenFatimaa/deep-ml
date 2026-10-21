def random_walk_td_mc(episodes: list, n_states: int, gamma: float = 1.0, alpha: float = 0.1) -> tuple:
    """
    Run TD(0) and first-visit Monte Carlo prediction on random walk episodes.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of non-terminal states
        gamma: Discount factor
        alpha: Learning rate for TD(0)
    
    Returns:
        Tuple of (td_values, mc_values) as lists of floats
    """
    # Initialize TD(0) values
    td_values = [0.0] * n_states
    
    # Process episodes for TD(0)
    for episode in episodes:
        for state, reward, next_state, done in episode:
            target = reward + (0.0 if done else gamma * td_values[next_state])
            td_values[state] += alpha * (target - td_values[state])
            
    # Process episodes for First-Visit Monte Carlo
    mc_returns = [[] for _ in range(n_states)]
    
    for episode in episodes:
        G = 0.0
        episode_data = []
        for state, reward, next_state, done in reversed(episode):
            G = reward + gamma * G
            episode_data.append((state, G))
            
        first_visit_G = {}
        for state, G_val in reversed(episode_data):
            if state not in first_visit_G:
                first_visit_G[state] = G_val
                
        for state, G_val in first_visit_G.items():
            if 0 <= state < n_states:
                mc_returns[state].append(G_val)
                
    mc_values = [
        sum(rets) / len(rets) if rets else 0.0
        for rets in mc_returns
    ]
    
    return td_values, mc_values