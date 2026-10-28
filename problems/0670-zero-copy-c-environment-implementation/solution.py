import numpy as np

def zero_copy_env_simulation(num_envs: int, obs_dim: int, step_data: list, commands: list) -> dict:
    """
    Simulate a vectorized C environment with zero-copy buffer management.

    Args:
        num_envs: Number of parallel environments.
        obs_dim: Dimension of each observation vector.
        step_data: List of (obs_2d, rewards, dones) tuples for each timestep.
        commands: List of command tuples to execute.

    Returns:
        Dictionary with keys: 'reads', 'alias_checks', 'total_copies',
        'total_views', 'total_bytes_saved', 'buffer_reallocs'.
    """
    # Pre-allocate zero-initialized buffers
    observations = np.zeros((num_envs, obs_dim), dtype=np.float64)
    rewards = np.zeros((num_envs,), dtype=np.float64)
    dones = np.zeros((num_envs,), dtype=np.int8)
    
    stored_arrays = {}
    step_idx = 0
    
    reads = []
    alias_checks = []
    total_copies = 0
    total_views = 0
    
    for cmd in commands:
        action = cmd[0]
        
        if action == "step":
            if step_idx < len(step_data):
                obs_in, rew_in, done_in = step_data[step_idx]
                observations[:, :] = obs_in
                rewards[:] = rew_in
                dones[:] = done_in
                step_idx += 1
                
        elif action == "store_view":
            name = cmd[1]
            # Store a view sharing underlying memory
            stored_arrays[name] = observations[:]
            total_views += 1
            
        elif action == "store_copy":
            name = cmd[1]
            # Store an independent copy
            stored_arrays[name] = observations.copy()
            total_copies += 1
            
        elif action == "read":
            name = cmd[1]
            arr = stored_arrays[name]
            # Round values to 4 decimals in nested list representation
            rounded = np.round(arr, 4).tolist()
            reads.append(rounded)
            
        elif action == "read_buffer":
            rounded = np.round(observations, 4).tolist()
            reads.append(rounded)
            
        elif action == "check_alias":
            name = cmd[1]
            arr = stored_arrays[name]
            # Check if arr shares memory with observations buffer
            shares_memory = np.shares_memory(arr, observations)
            alias_checks.append(bool(shares_memory))
            
        elif action == "auto_reset":
            # For every environment whose done flag is 1, zero its observation row and reward in place
            mask = (dones == 1)
            if np.any(mask):
                observations[mask, :] = 0.0
                rewards[mask] = 0.0
                
    total_bytes_saved = total_views * num_envs * obs_dim * 8
    
    return {
        "reads": reads,
        "alias_checks": alias_checks,
        "total_copies": total_copies,
        "total_views": total_views,
        "total_bytes_saved": total_bytes_saved,
        "buffer_reallocs": 0
    }