def best_meeting_point(grid: list[list[int]]) -> int:
    """Find the minimum total Manhattan distance to gather all homes at a single cell.

    Args:
        grid: 2D binary grid where 1 represents a home and 0 an empty cell.

    Returns:
        Minimum total Manhattan distance as an integer.
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])

    # 1. Collect row coordinates of all homes in sorted order
    r_coords = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                r_coords.append(r)

    # If no homes exist in the grid
    if not r_coords:
        return 0

    # 2. Collect column coordinates of all homes in sorted order
    c_coords = []
    for c in range(cols):
        for r in range(rows):
            if grid[r][c] == 1:
                c_coords.append(c)

    # 3. Calculate 1D minimum distance using two-pointer approach for sorted arrays
    def min_distance_1d(coords: list[int]) -> int:
        total_dist = 0
        left, right = 0, len(coords) - 1
        while left < right:
            total_dist += coords[right] - coords[left]
            left += 1
            right -= 1
        return total_dist

    return min_distance_1d(r_coords) + min_distance_1d(c_coords)