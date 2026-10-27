import numpy as np

def async_env_pool(num_envs: int, step_durations: list, rewards_per_step: list, total_ticks: int, batch_size: int) -> tuple:
    """
    Simulate an asynchronous pool of environments over discrete ticks.

    Args:
        num_envs: int, number of environments
        step_durations: list of ints, ticks per step for each environment
        rewards_per_step: list of floats, reward per step for each environment
        total_ticks: int, total simulation ticks
        batch_size: int, maximum ready environments to dispatch per tick

    Returns:
        tuple: (total_steps, total_reward, avg_envs_active, throughput)
    """
    if total_ticks == 0:
        return (0, 0.0, 0.0, 0.0)
    
    remaining_time = [0] * num_envs
    total_steps = 0
    total_reward = 0.0
    busy_counts = []
    
    for _ in range(total_ticks):
        # 1. Advance busy environments
        for i in range(num_envs):
            if remaining_time[i] > 0:
                remaining_time[i] -= 1
                if remaining_time[i] == 0:
                    total_steps += 1
                    total_reward += rewards_per_step[i]
                    
        # 2. Dispatch up to batch_size ready environments
        dispatched_count = 0
        for i in range(num_envs):
            if dispatched_count >= batch_size:
                break
            if remaining_time[i] == 0:
                remaining_time[i] = step_durations[i]
                dispatched_count += 1
                
        # 3. Record number of busy environments
        busy_count = sum(1 for r in remaining_time if r > 0)
        busy_counts.append(busy_count)
        
    avg_envs_active = float(np.mean(busy_counts)) if busy_counts else 0.0
    throughput = total_steps / total_ticks if total_ticks > 0 else 0.0
    
    return (
        int(total_steps),
        round(float(total_reward), 4),
        round(avg_envs_active, 4),
        round(throughput, 4)
    )