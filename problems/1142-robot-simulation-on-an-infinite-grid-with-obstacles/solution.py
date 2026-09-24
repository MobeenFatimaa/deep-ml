def robot_simulation(commands: list[int], obstacles: list[list[int]]) -> int:
    # Directions: North, East, South, West (represented as dx, dy offsets)
    # Turning right moves index + 1 % 4, turning left moves index - 1 % 4
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir_idx = 0  # Start facing North (directions[0])

    # Convert obstacles list to a set of tuples for O(1) lookup
    obstacle_set = {tuple(obs) for obs in obstacles}

    x, y = 0, 0
    max_dist_sq = 0

    for cmd in commands:
        if cmd == -1:
            # Turn right 90 degrees
            dir_idx = (dir_idx + 1) % 4
        elif cmd == -2:
            # Turn left 90 degrees
            dir_idx = (dir_idx - 1) % 4
        else:
            # Move forward step-by-step
            dx, dy = directions[dir_idx]
            for _ in range(cmd):
                next_x, next_y = x + dx, y + dy
                if (next_x, next_y) in obstacle_set:
                    break  # Stop moving for this command if blocked by obstacle
                x, y = next_x, next_y
                max_dist_sq = max(max_dist_sq, x * x + y * y)

    return max_dist_sq