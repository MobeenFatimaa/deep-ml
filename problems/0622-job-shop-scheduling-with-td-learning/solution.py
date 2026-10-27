import numpy as np

def jobshop_td_learning(jobs, n_machines, n_episodes, gamma, alpha, epsilon, seed=42):
    """
    Learn a scheduling policy for job-shop scheduling using SARSA (TD(0)).

    Args:
        jobs: list of lists, jobs[i] = [(machine, duration), ...] for job i
        n_machines: int, number of machines
        n_episodes: int, number of training episodes
        gamma: float, discount factor
        alpha: float, learning rate
        epsilon: float, exploration rate
        seed: int, random seed

    Returns:
        dict with 'best_makespan', 'best_schedule', 'q_values'
    """
    rng = np.random.RandomState(seed)
    n_jobs = len(jobs)
    
    q_table = {}
    updated_pairs = set()
    
    best_makespan = float('inf')
    best_schedule = None
    
    def get_q(s, a):
        return q_table.get((s, a), 0.0)

    def select_action(s, available):
        if not available:
            return None
        if rng.random() < epsilon:
            idx = rng.randint(len(available))
            return available[idx]
        else:
            best_a = available[0]
            best_q = get_q(s, best_a)
            for a in available[1:]:
                q = get_q(s, a)
                if q > best_q:
                    best_q = q
                    best_a = a
            return best_a

    for ep in range(n_episodes):
        state = [0] * n_jobs
        machine_free_time = [0.0] * n_machines
        job_free_time = [0.0] * n_jobs
        schedule = []

        available = [i for i in range(n_jobs) if state[i] < len(jobs[i])]
        action = select_action(tuple(state), available)

        while action is not None:
            s_tuple = tuple(state)
            job_idx = action
            schedule.append(job_idx)

            op_idx = state[job_idx]
            machine, duration = jobs[job_idx][op_idx]

            start_time = max(machine_free_time[machine], job_free_time[job_idx])
            end_time = start_time + duration

            machine_free_time[machine] = end_time
            job_free_time[job_idx] = end_time

            state[job_idx] += 1
            s_prime_tuple = tuple(state)

            next_available = [i for i in range(n_jobs) if state[i] < len(jobs[i])]
            is_terminal = (len(next_available) == 0)

            if is_terminal:
                makespan = max(machine_free_time)
                reward = -makespan
                next_action = None
            else:
                reward = 0.0
                next_action = select_action(s_prime_tuple, next_available)

            current_q = get_q(s_tuple, action)
            if is_terminal:
                target = reward
            else:
                target = reward + gamma * get_q(s_prime_tuple, next_action)

            new_q = current_q + alpha * (target - current_q)
            q_table[(s_tuple, action)] = new_q
            updated_pairs.add((s_tuple, action))

            if is_terminal:
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = list(schedule)
                break

            action = next_action

    formatted_q = {k: round(q_table[k], 4) for k in updated_pairs}
    return {
        'best_makespan': round(float(best_makespan), 4),
        'best_schedule': best_schedule,
        'q_values': formatted_q
    }