from collections import deque


def critical_path_method(tasks: dict, dependencies: dict) -> dict:
    """Compute the critical path, project duration, and task slack.

    Args:
        tasks: Dict mapping task name to its duration.
        dependencies: Dict mapping task name to list of prerequisite task names.

    Returns:
        Dict with keys 'project_duration', 'critical_path', and 'slack'.
    """
    # Build adjacency list (dependents) and calculate in-degrees
    dependents = {task: [] for task in tasks}
    in_degree = {task: 0 for task in tasks}

    for task, prereqs in dependencies.items():
        in_degree[task] = len(prereqs)
        for p in prereqs:
            dependents[p].append(task)

    # 1. Topological Sort (using Kahn's Algorithm)
    # Using a list sorted alphabetically to tie-break equal precedence tasks
    zero_in_degree = [task for task in tasks if in_degree[task] == 0]
    zero_in_degree.sort()

    topo_order = []
    queue = deque(zero_in_degree)

    while queue:
        u = queue.popleft()
        topo_order.append(u)

        # Collect newly zeroed nodes to sort them deterministically
        next_nodes = []
        for v in dependents[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                next_nodes.append(v)

        for node in sorted(next_nodes):
            queue.append(node)

    # 2. Forward Pass: Calculate Earliest Start (ES) and Earliest Finish (EF)
    ES = {task: 0 for task in tasks}
    EF = {task: 0 for task in tasks}

    for u in topo_order:
        prereqs = dependencies[u]
        if prereqs:
            ES[u] = max(EF[p] for p in prereqs)
        else:
            ES[u] = 0
        EF[u] = ES[u] + tasks[u]

    project_duration = max(EF.values()) if EF else 0

    # 3. Backward Pass: Calculate Latest Finish (LF) and Latest Start (LS)
    LF = {task: project_duration for task in tasks}
    LS = {task: project_duration for task in tasks}

    # Process in reverse topological order
    for u in reversed(topo_order):
        deps = dependents[u]
        if deps:
            LF[u] = min(LS[d] for d in deps)
        else:
            LF[u] = project_duration
        LS[u] = LF[u] - tasks[u]

    # 4. Calculate Slack and Critical Path
    slack = {task: LS[task] - ES[task] for task in tasks}

    # Order critical path by topological/execution sequence
    critical_path = [task for task in topo_order if slack[task] == 0]

    return {
        "project_duration": project_duration,
        "critical_path": critical_path,
        "slack": slack,
    }