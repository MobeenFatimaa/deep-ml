import copy
from itertools import islice

def stream_group_stats(rows, chunk_size: int, state: dict = None, max_chunks: int = None) -> dict:
    """Fold (key, value) rows into per-group count/mean/m2/min/max chunk by chunk, resumably."""
    # 1. Initialize or deepcopy state to avoid mutating the caller's reference
    if state is None:
        state = {
            "offset": 0,
            "done": False,
            "groups": {}
        }
    else:
        state = copy.deepcopy(state)
        
    # If the stream was already marked done, return it unchanged
    if state["done"]:
        return state
        
    # 2. Fast-forward the iterator past rows consumed so far
    offset = state["offset"]
    if offset > 0:
        rows = islice(rows, offset, None)
        
    groups = state["groups"]
    chunks_processed = 0
    
    # 3. Process chunks
    while True:
        if max_chunks is not None and chunks_processed >= max_chunks:
            break
            
        # Read up to chunk_size rows without loading the entire stream into memory
        chunk = list(islice(rows, chunk_size))
        chunks_processed += 1
        num_rows = len(chunk)
        
        offset += num_rows
        state["offset"] = offset
        
        # Fold rows into group statistics using Welford's algorithm
        for key, val in chunk:
            val = float(val)
            if key not in groups:
                groups[key] = {
                    "count": 1,
                    "mean": val,
                    "m2": 0.0,
                    "min": val,
                    "max": val
                }
            else:
                g = groups[key]
                g["count"] += 1
                count = g["count"]
                delta = val - g["mean"]
                g["mean"] += delta / count
                delta2 = val - g["mean"]
                g["m2"] += delta * delta2
                if val < g["min"]:
                    g["min"] = val
                if val > g["max"]:
                    g["max"] = val
                    
        # A chunk with fewer than chunk_size rows means the source is exhausted
        if num_rows < chunk_size:
            state["done"] = True
            break
            
    return state