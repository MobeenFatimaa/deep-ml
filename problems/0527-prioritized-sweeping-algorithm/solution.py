import numpy as np
import heapq

def prioritized_sweeping(
    num_states: int,
    num_actions: int,
    experiences: list,
    alpha: float,
    gamma: float,
    theta: float,
    n_planning: int
) -> np.ndarray:
    """
    Perform prioritized sweeping on a sequence of experiences.
    
    Args:
        num_states: Number of states in the environment
        num_actions: Number of possible actions
        experiences: List of (state, action, reward, next_state, done) tuples
        alpha: Learning rate
        gamma: Discount factor
        theta: Priority threshold
        n_planning: Max planning steps per experience
        
    Returns:
        Q-table as numpy array of shape (num_states, num_actions)
    """
    Q = np.zeros((num_states, num_actions), dtype=float)
    model = {}          # maps (s, a) -> (r, s_next, done)
    predecessors = {}   # maps s_next -> set of (s, a) pairs leading to it
    
    # Priority queue stores tuples of (-priority, s, a) for max-heap behavior
    pq = []
    pq_set = set()      # to keep track of items in the queue and avoid duplicates
    
    def compute_priority(s, a):
        r, s_next, done = model[(s, a)]
        if done:
            target = r
        else:
            target = r + gamma * np.max(Q[s_next])
        return abs(target - Q[s, a])

    for s, a, r, s_next, done in experiences:
        # 1. Update model
        model[(s, a)] = (r, s_next, done)
        
        # 2. Update predecessors
        if s_next not in predecessors:
            predecessors[s_next] = set()
        predecessors[s_next].add((s, a))
        
        # 3. Compute priority and push to queue if above theta
        priority = compute_priority(s, a)
        if priority > theta:
            if (s, a) not in pq_set:
                heapq.heappush(pq, (-priority, s, a))
                pq_set.add((s, a))
            else:
                # Update priority in place if already present (by re-pushing with max priority)
                # To handle efficiently, we can just push and filter, or use a lookup. 
                # Since priorities can increase, we push a new entry and let outdated ones be handled or cleaned.
                # A standard way is pushing (-priority, s, a) and maintaining set/tracking.
                heapq.heappush(pq, (-priority, s, a))
                pq_set.add((s, a))

        # 4. Planning steps
        steps = 0
        while pq and steps < n_planning:
            neg_p, u_s, u_a = heapq.heappop(pq)
            pq_set.discard((u_s, u_a))
            
            # If the popped priority is smaller than the current actual priority (due to multiple pushes), skip
            current_p = compute_priority(u_s, u_a)
            if current_p <= theta:
                continue
                
            # Update Q-value for (u_s, u_a)
            r_m, s_next_m, done_m = model[(u_s, u_a)]
            if done_m:
                target_m = r_m
            else:
                target_m = r_m + gamma * np.max(Q[s_next_m])
            Q[u_s, u_a] += alpha * (target_m - Q[u_s, u_a])
            
            # Propagate priority to predecessors of u_s
            if u_s in predecessors:
                for pred_s, pred_a in predecessors[u_s]:
                    p_pred = compute_priority(pred_s, pred_a)
                    if p_pred > theta:
                        if (pred_s, pred_a) not in pq_set:
                            heapq.heappush(pq, (-p_pred, pred_s, pred_a))
                            pq_set.add((pred_s, pred_a))
                        else:
                            heapq.heappush(pq, (-p_pred, pred_s, pred_a))
                            pq_set.add((pred_s, pred_a))
            
            steps += 1

    return Q