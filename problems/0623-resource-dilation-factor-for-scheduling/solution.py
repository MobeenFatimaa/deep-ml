import heapq

def compute_dilation_factor(tasks: list, capacity: int) -> dict:
    """
    Compute the resource dilation factor for a constrained scheduling problem.
    
    Args:
        tasks: List of dicts with 'duration', 'resources', 'dependencies' keys.
        capacity: Total available resource units.
    
    Returns:
        Dict with 'dilation_factor', 'actual_makespan', 'lower_bound', 'start_times'.
    """
    n = len(tasks)
    if n == 0:
        return {
            "dilation_factor": 0.0,
            "actual_makespan": 0.0,
            "lower_bound": 0.0,
            "start_times": []
        }

    # 1. Critical Path Length via Topological Order DP
    in_degree = [len(task["dependencies"]) for task in tasks]
    dependents = [[] for _ in range(n)]
    for i, task in enumerate(tasks):
        for dep in task["dependencies"]:
            dependents[dep].append(i)
            
    task_cp = [0.0] * n
    queue = [i for i in range(n) if in_degree[i] == 0]
    
    while queue:
        curr = queue.pop(0)
        max_dep_finish = max((task_cp[dep] for dep in tasks[curr]["dependencies"]), default=0.0)
        task_cp[curr] = max_dep_finish + tasks[curr]["duration"]
        
        for neighbor in dependents[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    critical_path = max(task_cp) if task_cp else 0.0

    # 2. Total Work & Lower Bound
    total_work = sum(task["duration"] * task["resources"] for task in tasks)
    lower_bound = max(critical_path, total_work / capacity)

    # 3. Greedy List Simulation
    start_times = [None] * n
    completed = [False] * n
    scheduled = [False] * n
    
    current_time = 0.0
    used_resources = 0
    running_events = []  # Min-heap: (finish_time, task_index)

    while sum(completed) < n:
        # Process completions
        while running_events and running_events[0][0] <= current_time + 1e-12:
            f_time, task_idx = heapq.heappop(running_events)
            completed[task_idx] = True
            used_resources -= tasks[task_idx]["resources"]

        # Attempt to schedule ready tasks by index
        for i in range(n):
            if not scheduled[i]:
                deps_satisfied = all(completed[dep] for dep in tasks[i]["dependencies"])
                req_resources = tasks[i]["resources"]
                
                if deps_satisfied and (used_resources + req_resources <= capacity):
                    start_times[i] = current_time
                    scheduled[i] = True
                    used_resources += req_resources
                    heapq.heappush(running_events, (current_time + tasks[i]["duration"], i))

        # Advance time to next completion
        if running_events:
            current_time = running_events[0][0]
        else:
            break

    actual_makespan = max((start_times[i] + tasks[i]["duration"] for i in range(n)), default=0.0)
    dilation_factor = actual_makespan / lower_bound if lower_bound > 0 else 0.0

    return {
        "dilation_factor": round(dilation_factor, 4),
        "actual_makespan": round(actual_makespan, 4),
        "lower_bound": round(lower_bound, 4),
        "start_times": [round(st, 4) for st in start_times]
    }