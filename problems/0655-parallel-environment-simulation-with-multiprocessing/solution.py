import numpy as np
from multiprocessing import Pool

def simulate_episode(args: tuple) -> tuple:
    """
    Worker function for simulating a single episode in a tabular MDP.
    Must be defined at module level for pickling.
    
    Args:
        args: tuple of (worker_id, seed, max_steps, trans_probs, rewards,
            terminal_states, policy, gamma, start_state)
    
    Returns:
        Tuple of (worker_id, total_return, episode_length)
    """
    worker_id, seed, max_steps, trans_probs, rewards, terminal_states, policy, gamma, start_state = args
    rng = np.random.RandomState(seed)
    
    state = start_state
    total_return = 0.0
    episode_length = 0
    
    for k in range(max_steps):
        if state in terminal_states:
            break
            
        action_probs = policy[state]
        action = rng.choice(len(action_probs), p=action_probs)
        
        reward = rewards[state, action]
        total_return += (gamma ** k) * reward
        episode_length += 1
        
        next_state_probs = trans_probs[state, action]
        state = rng.choice(len(next_state_probs), p=next_state_probs)
        
    return (worker_id, total_return, episode_length)

def parallel_env_simulate(
    num_workers: int,
    max_steps: int,
    trans_probs: np.ndarray,
    rewards: np.ndarray,
    terminal_states: list,
    policy: np.ndarray,
    gamma: float,
    start_state: int,
    seeds: list
) -> dict:
    """
    Run parallel environment simulations using multiprocessing.
    
    Returns:
        Dictionary with keys: 'mean_return', 'std_return', 'returns', 'lengths', 'total_steps'
    """
    args_list = [
        (i, seeds[i], max_steps, trans_probs, rewards, terminal_states, policy, gamma, start_state)
        for i in range(num_workers)
    ]
    
    with Pool(num_workers) as pool:
        results = pool.map(simulate_episode, args_list)
        
    results.sort(key=lambda x: x[0])
    
    returns = [float(res[1]) for res in results]
    lengths = [int(res[2]) for res in results]
    total_steps = sum(lengths)
    
    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    
    return {
        'mean_return': round(mean_return, 4),
        'std_return': round(std_return, 4),
        'returns': [round(r, 4) for r in returns],
        'lengths': lengths,
        'total_steps': total_steps
    }