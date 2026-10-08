def process_operations(operations):
    # Queue initialized as a list
    queue = []

    # Main stack and auxiliary min stack
    stack = []
    min_stack = []

    results = []

    for op in operations:
        action = op[0]

        if action == "enqueue":
            val = op[1]
            queue.append(val)

        elif action == "dequeue":
            # FIFO: remove from the front
            removed_val = queue.pop(0)
            results.append(removed_val)

        elif action == "mpush":
            val = op[1]
            stack.append(val)
            # Track current minimum in min_stack
            if not min_stack or val <= min_stack[-1]:
                min_stack.append(val)
            else:
                min_stack.append(min_stack[-1])

        elif action == "mpop":
            removed_val = stack.pop()
            min_stack.pop()
            results.append(removed_val)

        elif action == "mtop":
            top_val = stack[-1]
            results.append(top_val)

        elif action == "mmin":
            min_val = min_stack[-1]
            results.append(min_val)

    return results