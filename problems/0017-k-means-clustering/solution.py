import math

def k_means_clustering(
    points: list[tuple[float, ...]], 
    k: int, 
    initial_centroids: list[tuple[float, ...]], 
    max_iterations: int
) -> list[tuple[float, ...]]:
    
    centroids = [list(c) for c in initial_centroids]
    dimensions = len(points[0]) if points else 0

    for _ in range(max_iterations):
        # Step 1: Assign each point to the nearest centroid
        clusters = [[] for _ in range(k)]
        
        for point in points:
            # Calculate Euclidean distance to each centroid
            distances = [
                math.sqrt(sum((point[d] - centroid[d]) ** 2 for d in range(dimensions)))
                for centroid in centroids
            ]
            nearest_centroid_idx = distances.index(min(distances))
            clusters[nearest_centroid_idx].append(point)

        # Step 2: Update centroids to be the mean of assigned points
        new_centroids = []
        for i in range(k):
            cluster_points = clusters[i]
            if cluster_points:
                # Calculate component-wise mean
                mean_point = [
                    sum(p[d] for p in cluster_points) / len(cluster_points)
                    for d in range(dimensions)
                ]
                new_centroids.append(mean_point)
            else:
                # Keep original centroid if no points were assigned
                new_centroids.append(centroids[i])

        # Check for convergence (if centroids did not change)
        if new_centroids == centroids:
            break
            
        centroids = new_centroids

    # Round final centroid coordinates to 4 decimal places
    final_centroids = [
        tuple(round(val, 4) for val in centroid) 
        for centroid in centroids
    ]
    
    return final_centroids