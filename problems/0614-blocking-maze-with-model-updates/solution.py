import numpy as np

def blocking_maze_dyna_q(rows: int, cols: int, initial_walls: list, added_walls: list,
                         start: list, goal: list, change_step: int,
                         gamma: float, alpha: float, epsilon: float,
                         n_planning: int, n_steps: int, seed: int) -> dict:
    """
    Simulate a Dyna-Q agent in a blocking maze where walls change mid-simulation.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        initial_walls: List of [row, col] wall positions at start
        added_walls: List of [row, col] walls added at change_step
        start: [row, col] start position
        goal: [row, col] goal position
        change_step: Timestep at which added_walls appear
        gamma: Discount factor
        alpha: Learning rate
        epsilon: Exploration probability
        n_planning: Number of planning updates per real step
        n_steps: Total number of simulation steps
        seed: Random seed
    
    Returns:
        Dictionary with episodes_completed, cumulative_reward, 
        model_entries, and stale_entries
    """
    np.random.seed(seed)
    
    walls = set(tuple(w) for w in initial_walls)
    added_walls_set = set(tuple(w) for w in added_walls)
    
    Q = np.zeros((rows * cols, 4))
    model = {}
    
    current_pos = list(start)
    episodes_completed = 0
    cumulative_reward = 0.0
    
    def get_next_state_reward(r, c, action, current_walls):
        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r + 1, c
        elif action == 2:
            nr, nc = r, c - 1
        else:
            nr, nc = r, c + 1
            
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in current_walls:
            actual_r, actual_c = nr, nc
        else:
            actual_r, actual_c = r, c
            
        s_next = actual_r * cols + actual_c
        rew = 1.0 if [actual_r, actual_c] == goal else 0.0
        return s_next, [actual_r, actual_c], rew

    for t in range(n_steps):
        if t == change_step:
            walls.update(added_walls_set)
            
        s = current_pos[0] * cols + current_pos[1]
        
        # Action selection
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = int(np.argmax(Q[s]))
            
        s_next, next_pos, r = get_next_state_reward(current_pos[0], current_pos[1], action, walls)
        
        # Q-learning update
        Q[s, action] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, action])
        
        # Model update
        model[(s, action)] = (s_next, r)
        
        cumulative_reward += r
        
        if next_pos == goal:
            episodes_completed += 1
            current_pos = list(start)
        else:
            current_pos = next_pos
            
        # Planning steps
        if len(model) > 0:
            keys_list = list(model.keys())
            for _ in range(n_planning):
                idx = np.random.randint(len(model))
                ps, pa = keys_list[idx]
                ps_next, pr = model[(ps, pa)]
                Q[ps, pa] += alpha * (pr + gamma * np.max(Q[ps_next]) - Q[ps, pa])
                
    # Calculate stale entries against final walls
    stale_entries = 0
    for (s, a), (s_stored, r_stored) in model.items():
        r_coord = s // cols
        c_coord = s % cols
        s_true, _, r_true = get_next_state_reward(r_coord, c_coord, a, walls)
        if s_stored != s_true or r_stored != r_true:
            stale_entries += 1
            
    return {
        "episodes_completed": episodes_completed,
        "cumulative_reward": round(float(cumulative_reward), 4),
        "model_entries": len(model),
        "stale_entries": stale_entries
    }