import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    q = np.array(query_point)
    scored_points = []
    
    for idx, pt in enumerate(points):
        p = np.array(pt)
        # Compute Euclidean distance
        dist = np.linalg.norm(p - q)
        # Store distance, original index (for stable sorting ties), and the point tuple
        scored_points.append((dist, idx, pt))
    
    # Sort primarily by distance (ascending) and secondarily by original index (ascending)
    scored_points.sort(key=lambda x: (x[0], x[1]))
    
    # Extract the top k points
    return [pt for _, _, pt in scored_points[:k]]