def tree_and_graph_drills(task, *args):
    """
    Dispatcher function to solve four classic coding problems based on the `task` parameter.
    
    Supported tasks:
      - "lca": tree, root, p, q
      - "clone": graph
      - "min_removals": s
      - "word_break": s, word_dict
    """
    
    if task == "lca":
        tree, root, p, q = args
        
        def find_lca(node):
            if node is None or node == p or node == q:
                return node
            
            left_child, right_child = tree.get(node, [None, None])
            
            left = find_lca(left_child)
            right = find_lca(right_child)
            
            if left is not None and right is not None:
                return node
            
            return left if left is not None else right
        
        return find_lca(root)

    elif task == "clone":
        graph = args[0]
        if not graph:
            return {}

        # Dictionary to store mapped copies of nodes
        cloned = {}

        def dfs(node):
            if node in cloned:
                return
            cloned[node] = []
            for neighbor in graph.get(node, []):
                if neighbor not in cloned:
                    dfs(neighbor)
                cloned[node].append(neighbor)

        # Traverse all connected components
        for node in graph:
            if node not in cloned:
                dfs(node)

        # Return deep copy with sorted keys and sorted neighbor lists
        return {
            key: sorted(cloned[key])
            for key in sorted(cloned.keys())
        }

    elif task == "min_removals":
        s = args[0]
        unmatched_open = 0
        unmatched_close = 0
        
        for char in s:
            if char == '(':
                unmatched_open += 1
            elif char == ')':
                if unmatched_open > 0:
                    unmatched_open -= 1
                else:
                    unmatched_close += 1
                    
        return unmatched_open + unmatched_close

    elif task == "word_break":
        s, word_dict = args[0], args[1]
        word_set = set(word_dict)
        n = len(s)
        
        # dp[i] represents whether s[:i] can be segmented using word_set
        dp = [False] * (n + 1)
        dp[0] = True  # Empty string base case
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
                    
        return dp[n]

    else:
        raise ValueError(f"Unknown task: {task}")