def alpha_beta_pruning(tree, is_maximizing=True):
    """
    Perform alpha-beta pruning on a game tree.

    Args:
        tree: A nested list representing the game tree. Leaf nodes are numbers,
              internal nodes are lists of children.
        is_maximizing: Whether the root node is a maximizing player.

    Returns:
        A tuple (value, nodes_evaluated) where value is the optimal minimax
        value and nodes_evaluated is the number of leaf nodes examined.
    """

    def search(node, is_max, alpha, beta):
        # Base case: leaf node
        if isinstance(node, (int, float)):
            return node, 1

        leaves_evaluated = 0

        if is_max:
            value = float("-inf")
            for child in node:
                child_val, leaves = search(child, False, alpha, beta)
                leaves_evaluated += leaves
                value = max(value, child_val)
                alpha = max(alpha, value)

                # Beta cut-off
                if alpha >= beta:
                    break
            return value, leaves_evaluated
        else:
            value = float("inf")
            for child in node:
                child_val, leaves = search(child, True, alpha, beta)
                leaves_evaluated += leaves
                value = min(value, child_val)
                beta = min(beta, value)

                # Alpha cut-off
                if alpha >= beta:
                    break
            return value, leaves_evaluated

    return search(tree, is_maximizing, float("-inf"), float("inf"))