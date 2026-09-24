import math


def tile_coding_hash(
    state: list,
    num_tilings: int,
    tiles_per_dim: int,
    memory_size: int,
    state_bounds: list = None,
) -> list:
    """Compute active tile indices using tile coding with hash-based index mapping.

    Args:
        state: list of floats, continuous state variables.
        num_tilings: int, number of tilings.
        tiles_per_dim: int, tiles per dimension per tiling.
        memory_size: int, total hash table size.
        state_bounds: list of (low, high) tuples per dimension (default: (0,1)).

    Returns:
        List of int, one active tile index per tiling.
    """
    num_dims = len(state)

    # Set default state bounds to (0.0, 1.0) for each dimension if not provided
    if state_bounds is None:
        state_bounds = [(0.0, 1.0)] * num_dims

    active_indices = []

    for t in range(num_tilings):
        tiling_offset = t / num_tilings

        # Discretize each state dimension with the displacement offset
        coords = [t]
        for i, val in enumerate(state):
            low, high = state_bounds[i]
            # Normalize to [0, tiles_per_dim]
            norm_val = ((val - low) / (high - low)) * tiles_per_dim
            # Add tiling offset and floor to get integer tile coordinate
            tile_coord = math.floor(norm_val + tiling_offset)
            coords.append(tile_coord)

        # Apply djb2 hash function over the coordinate tuple
        h = 5381
        for c in coords:
            h = ((h * 33) + c) & 0xFFFFFFFF  # Mask to 32 bits

        # Map to memory size
        active_indices.append(h % memory_size)

    return active_indices