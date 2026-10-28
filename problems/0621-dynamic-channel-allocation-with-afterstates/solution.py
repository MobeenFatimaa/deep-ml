import numpy as np

def channel_afterstate_step(
    allocation: np.ndarray,
    call_cell: int,
    adjacency: np.ndarray,
    V: dict,
    alpha: float,
    gamma: float,
    prev_afterstate_key
) -> dict:
    """
    Perform one step of dynamic channel allocation using afterstate values.

    Args:
        allocation: binary array (n_cells, n_channels), 1 = channel in use
        call_cell: cell index requesting a channel
        adjacency: binary array (n_cells, n_cells), 1 = cells are adjacent
        V: dict mapping afterstate keys (tuple of tuples) to float values
        alpha: learning rate
        gamma: discount factor
        prev_afterstate_key: afterstate chosen at the previous step (None on the first step)

    Returns:
        dict with 'chosen_channel', 'reward', 'afterstate_key', 'V'
    """
    n_cells, n_channels = allocation.shape
    
    # Create a copy of V to avoid mutating the input dictionary
    new_V = dict(V)
    
    candidates = []  
    
    # 1. Evaluate Rejection Option (chosen_channel = -1)
    rej_afterstate = tuple(tuple(int(val) for val in row) for row in allocation)
    rej_val = new_V.get(rej_afterstate, 0.0)
    rej_return = 0.0 + gamma * rej_val
    candidates.append((rej_return, -1, 0.0, rej_afterstate))
    
    # 2. Evaluate Acceptance Options (for each free channel)
    for ch in range(n_channels):
        if allocation[call_cell, ch] == 0:
            after_alloc = allocation.copy()
            after_alloc[call_cell, ch] = 1
            
            current_users = np.where(allocation[:, ch] == 1)[0]
            
            additional_interference = 0
            for neighbor in current_users:
                if neighbor != call_cell and adjacency[call_cell, neighbor] == 1:
                    additional_interference += 1
            
            reward = 1.0 - float(additional_interference)
            afterstate_key = tuple(tuple(int(val) for val in row) for row in after_alloc)
            
            val = new_V.get(afterstate_key, 0.0)
            expected_return = reward + gamma * val
            
            candidates.append((expected_return, ch, reward, afterstate_key))
            
    def selection_key(item):
        ret, ch, rew, key = item
        is_accept = 1 if ch != -1 else 0
        ch_priority = -ch if ch != -1 else 0
        return (ret, is_accept, ch_priority)
        
    best_candidate = max(candidates, key=selection_key)
    chosen_return, chosen_channel, chosen_reward, chosen_afterstate_key = best_candidate
    
    # 3. Perform TD(0) update on previous afterstate if applicable
    if prev_afterstate_key is not None:
        old_v_prev = new_V.get(prev_afterstate_key, 0.0)
        v_next = new_V.get(chosen_afterstate_key, 0.0)
        td_target = chosen_reward + gamma * v_next
        updated_v_prev = old_v_prev + alpha * (td_target - old_v_prev)
        new_V[prev_afterstate_key] = round(updated_v_prev, 4)
        
    for k in new_V:
        new_V[k] = round(float(new_V[k]), 4)
        
    return {
        'chosen_channel': int(chosen_channel),
        'reward': round(float(chosen_reward), 4),
        'afterstate_key': chosen_afterstate_key,
        'V': new_V
    }