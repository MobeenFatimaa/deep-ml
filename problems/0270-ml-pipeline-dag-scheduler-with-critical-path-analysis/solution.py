import heapq

def analyze_ml_pipeline(tasks: list) -> dict:
    """
    Analyze an ML pipeline DAG for scheduling and critical path.
    
    Args:
        tasks: list of task dicts with:
            - 'id': task identifier (str)
            - 'duration': task duration in minutes (int)
            - 'dependencies': list of task IDs this task depends on
    
    Returns:
        dict with:
            - 'execution_order': topologically sorted list of task IDs
            - 'earliest_start': dict mapping task ID to earliest start time
            - 'earliest_finish': dict mapping task ID to earliest finish time
            - 'latest_start': dict mapping task ID to latest start time
            - 'latest_finish': dict mapping task ID to latest finish time
            - 'slack': dict mapping task ID to slack time
            - 'critical_path': list of task IDs on critical path (in execution order)
            - 'makespan': total time to complete pipeline
    """
    if not tasks:
        return {
            'execution_order': [],
            'earliest_start': {},
            'earliest_finish': {},
            'latest_start': {},
            'latest_finish': {},
            'slack': {},
            'critical_path': [],
            'makespan': 0
        }
        
    task_map = {t['id']: t for t in tasks}
    task_ids = list(task_map.keys())
    
    predecessors = {tid: list(task_map[tid]['dependencies']) for tid in task_ids}
    successors = {tid: [] for tid in task_ids}
    in_degree = {tid: len(predecessors[tid]) for tid in task_ids}
    
    for tid in task_ids:
        for dep in predecessors[tid]:
            successors[dep].append(tid)
            
    # 1. Topological Sorting using Kahn's algorithm with min-heap for alphabetical tie-breaking
    queue = [tid for tid in task_ids if in_degree[tid] == 0]
    heapq.heapify(queue)
    
    execution_order = []
    while queue:
        curr = heapq.heappop(queue)
        execution_order.append(curr)
        for succ in successors[curr]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                heapq.heappush(queue, succ)
                
    # 2. Forward Pass: Earliest Start & Earliest Finish
    earliest_start = {}
    earliest_finish = {}
    
    for tid in execution_order:
        duration = task_map[tid]['duration']
        if not predecessors[tid]:
            es = 0
        else:
            es = max(earliest_finish[dep] for dep in predecessors[tid])
        earliest_start[tid] = es
        earliest_finish[tid] = es + duration
        
    makespan = max(earliest_finish.values()) if earliest_finish else 0
    
    # 3. Backward Pass: Latest Start & Latest Finish
    temp_latest_finish = {}
    temp_latest_start = {}
    
    # Process in reverse topological order
    for tid in reversed(execution_order):
        duration = task_map[tid]['duration']
        if not successors[tid]:
            lf = makespan
        else:
            lf = min(temp_latest_start[succ] for succ in successors[tid])
        temp_latest_finish[tid] = lf
        temp_latest_start[tid] = lf - duration
        
    # Populate in execution_order to ensure consistent key insertion order matching earliest_start
    latest_start = {}
    latest_finish = {}
    slack = {}
    
    for tid in execution_order:
        latest_finish[tid] = temp_latest_finish[tid]
        latest_start[tid] = temp_latest_start[tid]
        slack[tid] = latest_start[tid] - earliest_start[tid]
        
    # 5. Critical Path
    zero_slack_tasks = {tid for tid in task_ids if slack[tid] == 0}
    
    critical_path = []
    for tid in execution_order:
        if tid in zero_slack_tasks:
            if not critical_path:
                critical_path.append(tid)
            else:
                if tid in successors[critical_path[-1]]:
                    critical_path.append(tid)
                    
    return {
        'execution_order': execution_order,
        'earliest_start': earliest_start,
        'earliest_finish': earliest_finish,
        'latest_start': latest_start,
        'latest_finish': latest_finish,
        'slack': slack,
        'critical_path': critical_path,
        'makespan': makespan
    }