import numpy as np

def shortcut_maze_exploration(grid_rows: int, grid_cols: int, walls_phase1: list, walls_phase2: list,
                              change_step: int, start: list, goal: tuple, total_steps: int,
                              alpha: float, gamma: float, epsilon: float, kappa: float,
                              n_planning: int, seed: int) -> dict:
    """
    Simulate a Dyna-Q+ agent in a shortcut maze where walls change during training.
    
    Args:
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        walls_phase1: List of [row, col] wall positions before change_step
        walls_phase2: List of [row, col] wall positions at and after change_step
        change_step: Timestep at which walls transition from phase1 to phase2
        start: [row, col] start position
        goal: (row, col) goal position
        total_steps: Total number of simulation steps
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration probability
        kappa: Weight for the exploration bonus in Dyna-Q+
        n_planning: Number of planning updates per real step
        seed: Random seed
        
    Returns:
        Dictionary containing 'episodes_completed', 'total_reward', and 'cumulative_reward'.
    """
    rng = np.random.RandomState(seed)
    
    walls1 = set(tuple(w) for w in walls_phase1)
    walls2 = set(tuple(w) for w in walls_phase2)
    
    n_states = grid_rows * grid_cols
    Q = np.zeros((n_states, 4))
    
    model = {}        # (s, a) -> (s_prime, r)
    last_visit = {}   # (s, a) -> t
    
    current_pos = list(start)
    episodes_completed = 0
    total_reward = 0.0
    cumulative_reward = []
    
    def get_next_state_reward(r, c, action, current_walls):
        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r + 1, c
        elif action == 2:
            nr, nc = r, c - 1
        else:
            nr, nc = r, c + 1
            
        if 0 <= nr < grid_rows and 0 <= nc < grid_cols and (nr, nc) not in current_walls:
            actual_r, actual_c = nr, nc
        else:
            actual_r, actual_c = r, c
            
        s_next = actual_r * grid_cols + actual_c
        rew = 1.0 if [actual_r, actual_c] == list(goal) else 0.0
        return s_next, [actual_r, actual_c], rew

    for t in range(total_steps):
        # Determine walls based on current timestep
        current_walls = walls1 if t < change_step else walls2
        
        r_curr, c_curr = current_pos
        s = r_curr * grid_cols + c_curr
        
        # Epsilon-greedy action selection
        if rng.random() < epsilon:
            action = rng.randint(4)
        else:
            max_q = np.max(Q[s])
            best_actions = [a for a in range(4) if np.isclose(Q[s, a], max_q, atol=1e-12)]
            action = best_actions[rng.randint(len(best_actions))]
            
        # Real environment step
        s_next, next_pos, r = get_next_state_reward(r_curr, c_curr, action, current_walls)
        
        # Q-learning update for real experience
        Q[s, action] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, action])
        
        # Update model and last visit time
        model[(s, action)] = (s_next, r)
        last_visit[(s, action)] = t
        
        total_reward += r
        cumulative_reward.append(total_reward)
        
        if next_pos == list(goal):
            episodes_completed += 1
            current_pos = list(start)
        else:
            current_pos = next_pos
            
        # Planning steps (Dyna-Q+)
        if len(model) > 0:
            model_keys = list(model.keys())
            for _ in range(n_planning):
                idx = rng.randint(len(model))
                ps, pa = model_keys[idx]
                ps_next, pr = model[(ps, pa)]
                
                # Exploration bonus for unvisited/stale state-action pairs
                tau = t - last_visit[(ps, pa)]
                bonus_reward = pr + kappa * np.sqrt(tau)
                
                Q[ps, pa] += alpha * (bonus_reward + gamma * np.max(Q[ps_next]) - Q[ps, pa])
                
    return {
        'episodes_completed': episodes_completed,
        'total_reward': float(total_reward),
        'cumulative_reward': cumulative_reward
    }