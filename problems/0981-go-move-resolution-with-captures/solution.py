from collections import deque

def resolve_go_move(board, player, row, col):
    """
    Place a stone and resolve captures under Tromp-Taylor rules.
    Returns the resulting board as a 2D list.
    """
    # Create a deep copy of the board to prevent mutating the input
    new_board = [r[:] for r in board]
    rows, cols = len(new_board), len(new_board[0])
    
    # Place the current player's stone
    new_board[row][col] = player
    opponent = 3 - player  # If player is 1 (Black), opponent is 2 (White), and vice versa

    def get_group_and_liberties(r, c):
        """Finds all stones in the connected group starting at (r, c) and its liberties."""
        color = new_board[r][c]
        group = set()
        liberties = set()
        queue = deque([(r, c)])
        group.add((r, c))

        while queue:
            curr_r, curr_c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if new_board[nr][nc] == 0:
                        liberties.add((nr, nc))
                    elif new_board[nr][nc] == color and (nr, nc) not in group:
                        group.add((nr, nc))
                        queue.append((nr, nc))

        return group, liberties

    def remove_group(group):
        """Removes all stones belonging to the specified group from the board."""
        for r, c in group:
            new_board[r][c] = 0

    # 1. Resolve opponent captures first
    captured_opponent_groups = []
    visited_opponents = set()

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if new_board[nr][nc] == opponent and (nr, nc) not in visited_opponents:
                group, liberties = get_group_and_liberties(nr, nc)
                visited_opponents.update(group)
                if len(liberties) == 0:
                    captured_opponent_groups.append(group)

    # Remove all captured opponent stones
    for group in captured_opponent_groups:
        remove_group(group)

    # 2. Resolve self-capture (suicide rule under Tromp-Taylor)
    own_group, own_liberties = get_group_and_liberties(row, col)
    if len(own_liberties) == 0:
        remove_group(own_group)

    return new_board