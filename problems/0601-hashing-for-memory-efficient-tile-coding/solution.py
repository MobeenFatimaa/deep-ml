import numpy as np

def hashed_tile_coding(state: list, num_tilings: int, num_tiles_per_dim: list, state_lows: list, state_highs: list, memory_size: int, weights: np.ndarray = None) -> tuple:
    """
    Compute memory-efficient tile coding features using hashing.
    
    Args:
        state: Continuous state vector, list of floats
        num_tilings: Number of overlapping tilings
        num_tiles_per_dim: Number of tiles per dimension per tiling
        state_lows: Lower bounds of state space per dimension
        state_highs: Upper bounds of state space per dimension
        memory_size: Size of the hash table (number of possible indices)
        weights: Optional weight vector of length memory_size
    
    Returns:
        tuple: (active_tiles, value_estimate)
            active_tiles: Sorted list of hashed tile indices (length num_tilings)
            value_estimate: float, sum of weights at active tiles (0.0 if no weights)
    """
    primes = [509, 521, 523, 541, 547, 557, 563, 569, 571, 577]
    ndim = len(state)
    
    if len(num_tiles_per_dim) == 1 and ndim > 1:
        num_tiles_per_dim = num_tiles_per_dim * ndim
        
    tile_widths = [(state_highs[d] - state_lows[d]) / num_tiles_per_dim[d] for d in range(ndim)]
    
    active_tiles = []
    value_estimate = 0.0
    
    for t in range(num_tilings):
        coords = [t]
        for d in range(ndim):
            tile_width = tile_widths[d]
            offset = t * tile_width / num_tilings
            shifted = state[d] - state_lows[d] + offset
            coord = int(np.floor(shifted / tile_width))
            coords.append(coord)
            
        h = sum(val * primes[i % len(primes)] for i, val in enumerate(coords))
        hashed_index = abs(h) % memory_size
        active_tiles.append(hashed_index)
        
    if weights is not None:
        value_estimate = float(np.sum([weights[idx] for idx in active_tiles]))
        value_estimate = round(value_estimate, 10)
        
    return sorted(active_tiles), value_estimate