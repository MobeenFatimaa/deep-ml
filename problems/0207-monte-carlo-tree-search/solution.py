import math
import random
import torch
from typing import Optional

class MCTSNode:
    def __init__(self, state: int, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def ucb1(self, c: float = 1.414) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

def mcts_search(initial_state: int, max_value: int, iterations: int, seed: int = 42) -> int:
    """
    Monte Carlo Tree Search for sequential game.
    
    Args:
        initial_state: Starting value
        max_value: Target value to reach
        iterations: Number of MCTS iterations
        seed: Random seed
    
    Returns:
        Best action (1 or 2)
    """
    random.seed(seed)
    root = MCTSNode(state=initial_state)
    
    for _ in range(iterations):
        node = root
        path = [node]
        
        # Selection phase
        while not node.is_leaf():
            legal_actions = [1, 2]
            if all(a in node.children for a in legal_actions):
                node = max(node.children.values(), key=lambda n: n.ucb1())
                path.append(node)
            else:
                break
                
        # Expansion phase
        current_state = node.state
        if current_state < max_value:
            legal_actions = [1, 2]
            untried = [a for a in legal_actions if a not in node.children]
            if untried:
                action = random.choice(untried)
                next_state = current_state + action
                node.children[action] = MCTSNode(state=next_state, parent=node)
                node = node.children[action]
                path.append(node)
                
        # Simulation phase
        current_sim_state = node.state
        while current_sim_state < max_value:
            action = random.choice([1, 2])
            current_sim_state += action
            
        reward = 1.0 if current_sim_state == max_value else -1.0
        
        # Backpropagation phase
        for n in reversed(path):
            n.visits += 1
            n.value += reward
            reward = -reward
            
    if not root.children:
        return 1
        
    best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_action