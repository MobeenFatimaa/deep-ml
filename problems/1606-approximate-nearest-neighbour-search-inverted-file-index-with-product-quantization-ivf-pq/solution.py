import numpy as np


def lloyd(X, k, iters):
    X = np.array(X, dtype=float)
    n, d = X.shape
    centroids = X[:k].copy()
    
    for _ in range(iters):
        # Compute squared Euclidean distances: shape (n, k)
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[assignments == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[j] = centroids[j]
        centroids = new_centroids
        
    return centroids


def build_ivf(X, centroids):
    X = np.array(X, dtype=float)
    centroids = np.array(centroids, dtype=float)
    dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    assignments = np.argmin(dists, axis=1)
    
    ivf = []
    for j in range(len(centroids)):
        indices = np.where(assignments == j)[0].tolist()
        ivf.append(sorted(indices))
    return ivf


def train_pq(X, m, k, iters):
    X = np.array(X, dtype=float)
    n, d = X.shape
    sub_d = d // m
    codebooks = np.zeros((m, k, sub_d))
    
    for j in range(m):
        sub_X = X[:, j * sub_d : (j + 1) * sub_d]
        codebooks[j] = lloyd(sub_X, k, iters)
        
    return codebooks


def encode_pq(X, codebooks):
    X = np.array(X, dtype=float)
    codebooks = np.array(codebooks, dtype=float)
    n, d = X.shape
    m, k, sub_d = codebooks.shape
    
    codes = np.zeros((n, m), dtype=int)
    for j in range(m):
        sub_X = X[:, j * sub_d : (j + 1) * sub_d]
        cb = codebooks[j]
        # Distances: (n, k)
        dists = np.sum((sub_X[:, None, :] - cb[None, :, :]) ** 2, axis=2)
        codes[:, j] = np.argmin(dists, axis=1)
        
    return codes


def adc_distances(q, codes, codebooks):
    q = np.array(q, dtype=float)
    codes = np.array(codes, dtype=int)
    codebooks = np.array(codebooks, dtype=float)
    
    m, k, sub_d = codebooks.shape
    n = codes.shape[0]
    
    # Table of squared distances from each sub-vector of q to each codeword: shape (m, k)
    table = np.zeros((m, k))
    for j in range(m):
        sub_q = q[j * sub_d : (j + 1) * sub_d]
        cb = codebooks[j]
        table[j] = np.sum((sub_q[None, :] - cb) ** 2, axis=1)
        
    # Sum table entries selected by codes: shape (n,)
    # codes is (n, m), table is (m, k) -> we want table[j, codes[:, j]] for each j
    approx_dists = np.zeros(n)
    for j in range(m):
        approx_dists += table[j, codes[:, j]]
        
    return approx_dists


def ivf_pq_search(q, codes, codebooks, centroids, ivf, nprobe, k):
    q = np.array(q, dtype=float)
    centroids = np.array(centroids, dtype=float)
    
    # Find nprobe nearest cells
    centroid_dists = np.sum((centroids - q) ** 2, axis=1)
    nearest_cells = np.argsort(centroid_dists)[:nprobe]
    
    candidate_indices = []
    for cell in nearest_cells:
        candidate_indices.extend(ivf[cell])
        
    if not candidate_indices:
        return []
        
    candidate_indices = np.array(candidate_indices)
    sub_codes = codes[candidate_indices]
    
    dists = adc_distances(q, sub_codes, codebooks)
    
    # Sort candidates by distance, then by original index
    sorted_idx = np.lexsort((candidate_indices, dists))
    best_local = sorted_idx[:k]
    
    return candidate_indices[best_local].tolist()


def recall_at_k(found, exact):
    found_set = set(found)
    exact_set = set(exact[:len(found)])
    if not exact_set:
        return 0.0
    intersection = found_set.intersection(exact_set)
    return round(len(intersection) / len(exact_set), 4)