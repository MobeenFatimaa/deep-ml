import math

class Node:
    def __init__(self, path: tuple):
        self.path = path          # tuple of actions from root
        self.visits = 0           # N visits
        self.total = 0.0          # total reward sum
        self.children = {}        # action -> Node mapping

def mcts_reasoning_trace(reward_table: dict, branching: int, depth: int, n_simulations: int, c: float) -> list:
    """
    UCB1-MCTS over a B-ary tree of depth `depth` with a step-wise reward model.

    Args:
        reward_table: dict mapping path tuples to step rewards (missing -> 0.0).
        branching: branching factor B (actions 0..B-1 at every internal node).
        depth: maximum trace depth (leaves live at this depth).
        n_simulations: number of MCTS iterations.
        c: UCB1 exploration constant.

    Returns:
        Best complete trace found (list of length `depth`).
    """
    root = Node(path=())
    
    best_trace = None
    best_v = -float('inf')
    
    for _ in range(n_simulations):
        # 1. Selection
        current = root
        search_path = [current]
        
        while len(current.path) < depth and len(current.children) == branching:
            best_child = None
            best_score = -float('inf')
            
            for a in range(branching):
                child = current.children[a]
                q = child.total / child.visits if child.visits > 0 else 0.0
                score = q + c * (math.log(current.visits) / child.visits if child.visits > 0 else float('inf'))
                
                # Lowest action index on ties
                if score > best_score:
                    best_score = score
                    best_child = child
            
            current = best_child
            search_path.append(current)
            
        # 2. Expansion
        if len(current.path) < depth and len(current.children) < branching:
            untried = [a for a in range(branching) if a not in current.children]
            if untried:
                action = min(untried)
                new_path = current.path + (action,)
                child = Node(path=new_path)
                current.children[action] = child
                current = child
                search_path.append(current)
                
        # 3. Rollout
        rollout_path = list(current.path)
        while len(rollout_path) < depth:
            best_action = 0
            best_step_reward = -float('inf')
            for a in range(branching):
                candidate_path = tuple(rollout_path + [a])
                step_rew = reward_table.get(candidate_path, 0.0)
                if step_rew > best_step_reward:
                    best_step_reward = step_rew
                    best_action = a
            rollout_path.append(best_action)
            
        # 4. Compute total reward V of the completed trace
        completed_trace = tuple(rollout_path)
        V = 0.0
        for i in range(1, depth + 1):
            subpath = completed_trace[:i]
            V += reward_table.get(subpath, 0.0)
            
        # Track best trace by strict > on V (earliest found wins ties)
        if V > best_v:
            best_v = V
            best_trace = list(completed_trace)
            
        # 5. Backpropagate along selection-plus-expansion path
        for node in search_path:
            node.visits += 1
            node.total += V
            
    return best_trace