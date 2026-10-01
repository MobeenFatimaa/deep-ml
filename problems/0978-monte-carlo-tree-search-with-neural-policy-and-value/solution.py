import math
import numpy as np

class Node:
    def __init__(self, state, num_actions, is_terminal=False, value=0.0):
        self.state = state
        self.is_terminal = is_terminal
        self.value = value  # Only relevant if terminal
        self.N = 0
        self.N_a = np.zeros(num_actions, dtype=np.float64)
        self.W_a = np.zeros(num_actions, dtype=np.float64)
        self.P_a = np.zeros(num_actions, dtype=np.float64)
        self.children = {}  # action -> Node

def mcts(root_state, num_actions, get_legal_actions, apply_action,
         check_terminal, policy_value_fn, c_puct, num_simulations):
    """
    Run MCTS guided by a policy+value function and return the
    normalized visit-count distribution over root actions.
    """
    # Check if root is terminal
    done, root_val = check_terminal(root_state)
    if done:
        return np.zeros(num_actions, dtype=np.float64)

    # Initialize and expand root node
    root = Node(root_state, num_actions)
    _expand_node(root, num_actions, get_legal_actions, policy_value_fn)

    for _ in range(num_simulations):
        node = root
        search_path = []  # Stores tuples of (node, action)

        # 1. Selection Phase
        while True:
            if node.is_terminal:
                leaf_value = node.value
                break

            legal_actions = get_legal_actions(node.state)
            
            # Select action maximizing PUCT score
            best_score = -float('inf')
            best_action = -1

            sqrt_N = math.sqrt(node.N)
            for a in legal_actions:
                q_a = (node.W_a[a] / node.N_a[a]) if node.N_a[a] > 0 else 0.0
                u_a = c_puct * node.P_a[a] * sqrt_N / (1.0 + node.N_a[a])
                score = q_a + u_a

                if score > best_score:
                    best_score = score
                    best_action = a

            search_path.append((node, best_action))

            # Check if selected child node exists
            if best_action not in node.children:
                # 2. Expansion Phase
                next_state = apply_action(node.state, best_action)
                child_done, child_val = check_terminal(next_state)
                
                child_node = Node(next_state, num_actions, is_terminal=child_done, value=child_val)
                node.children[best_action] = child_node

                if not child_done:
                    # 3. Evaluation Phase (Non-terminal)
                    leaf_value = _expand_node(child_node, num_actions, get_legal_actions, policy_value_fn)
                else:
                    # 3. Evaluation Phase (Terminal)
                    leaf_value = child_val

                break
            else:
                node = node.children[best_action]

        # 4. Backup Phase
        # Back up value along the traversed path, flipping signs due to alternating players
        v = leaf_value
        for p_node, action in reversed(search_path):
            v = -v
            p_node.N += 1
            p_node.N_a[action] += 1
            p_node.W_a[action] += v

    # Normalize root visit counts into a probability distribution
    total_visits = np.sum(root.N_a)
    if total_visits > 0:
        return root.N_a / total_visits
    else:
        return np.zeros(num_actions, dtype=np.float64)


def _expand_node(node, num_actions, get_legal_actions, policy_value_fn):
    """Mask priors to legal actions, renormalize, and set initial node properties."""
    legal_actions = get_legal_actions(node.state)
    raw_priors, value = policy_value_fn(node.state)

    # Mask illegal actions
    legal_mask = np.zeros(num_actions, dtype=np.float64)
    legal_mask[legal_actions] = 1.0
    masked_priors = np.array(raw_priors, dtype=np.float64) * legal_mask

    prior_sum = np.sum(masked_priors)
    if prior_sum > 0:
        node.P_a = masked_priors / prior_sum
    else:
        # Uniform distribution over legal actions if legal prior mass is 0
        node.P_a = legal_mask / len(legal_actions)

    return value