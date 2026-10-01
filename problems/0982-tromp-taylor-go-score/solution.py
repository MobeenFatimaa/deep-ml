from collections import deque

def tromp_taylor_score(board):
    """
    Compute Tromp-Taylor scores for a final Go board.

    Args:
        board: 2D list or numpy array of ints (0 empty, 1 black, 2 white)

    Returns:
        [black_score, white_score]
    """
    rows, cols = len(board), len(board[0])
    
    black_score = 0
    white_score = 0

    # 1. Count stones already on the board
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 1:
                black_score += 1
            elif board[r][c] == 2:
                white_score += 1

    # 2. Identify 4-connected regions of empty intersections (0s)
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 0 and (r, c) not in visited:
                # Explore connected component of empty intersections
                region = []
                border_colors = set()
                queue = deque([(r, c)])
                visited.add((r, c))

                while queue:
                    curr_r, curr_c = queue.popleft()
                    region.append((curr_r, curr_c))

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            val = board[nr][nc]
                            if val == 0:
                                if (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                            else:
                                border_colors.add(val)

                # 3. Assign territory based on surrounding stone colors
                if len(border_colors) == 1:
                    owner = border_colors.pop()
                    if owner == 1:
                        black_score += len(region)
                    elif owner == 2:
                        white_score += len(region)

    return [black_score, white_score]