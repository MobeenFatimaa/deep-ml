def detect_cycles(ll_next: list[int], graph: list[list[int]]) -> tuple[int, bool]:
    """Detect cycle entry node in a singly linked list and presence of a cycle in a directed graph.

    Args:
        ll_next: List where ll_next[i] is the next node from i, -1 means end of list.
        graph: Adjacency list of a directed graph.

    Returns:
        Tuple (entry_node_or_-1, graph_has_cycle_bool)
    """

    # Part 1: Singly Linked List Cycle Detection (Floyd's Algorithm)
    def find_linked_list_cycle_entry(ll_next: list[int]) -> int:
        if not ll_next or ll_next[0] == -1:
            return -1

        slow = 0
        fast = 0

        # Step 1: Detect if a cycle exists
        has_cycle = False
        while fast != -1 and ll_next[fast] != -1:
            slow = ll_next[slow]
            fast = ll_next[ll_next[fast]]

            if slow == fast:
                has_cycle = True
                break

        if not has_cycle:
            return -1

        # Step 2: Find the entry node of the cycle
        slow = 0
        while slow != fast:
            slow = ll_next[slow]
            fast = ll_next[fast]

        return slow

    # Part 2: Directed Graph Cycle Detection (DFS 3-color approach)
    def has_graph_cycle(graph: list[list[int]]) -> bool:
        n = len(graph)
        # 0: Unvisited, 1: Visiting (in current path), 2: Fully Visited
        state = [0] * n

        def dfs(node: int) -> bool:
            state[node] = 1  # Mark as visiting

            for neighbor in graph[node]:
                if state[neighbor] == 1:
                    return True  # Found a back edge / cycle
                if state[neighbor] == 0:
                    if dfs(neighbor):
                        return True

            state[node] = 2  # Mark as fully visited
            return False

        for i in range(n):
            if state[i] == 0:
                if dfs(i):
                    return True

        return False

    ll_entry = find_linked_list_cycle_entry(ll_next)
    graph_cycle = has_graph_cycle(graph)

    return (ll_entry, graph_cycle)