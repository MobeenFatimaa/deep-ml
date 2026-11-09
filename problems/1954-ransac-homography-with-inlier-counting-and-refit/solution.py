import numpy as np

def ransac_iterations(p, w, s):
    """
    Calculate the number of RANSAC iterations required to ensure with probability p
    that at least one selected sample of size s consists entirely of inliers,
    given an inlier ratio w.
    """
    if w <= 0.0:
        return int(1e9)
    if w >= 1.0:
        return 1
        
    denom = np.log(1.0 - w**s)
    if denom == 0:
        return 1
        
    n = np.log(1.0 - p) / denom
    return max(1, int(np.ceil(n)))

def _normalize_points(pts):
    """
    Hartley normalization for 2D points: shifts centroid to origin and 
    scales mean distance to sqrt(2).
    """
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    
    if mean_dist == 0:
        scale = 1.0
    else:
        scale = np.sqrt(2.0) / mean_dist
        
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=float)
    
    # Apply transformation
    n_pts = pts.shape[0]
    homog_pts = np.hstack([pts, np.ones((n_pts, 1))])
    norm_homog_pts = (T @ homog_pts.T).T
    norm_pts = norm_homog_pts[:, :2] / norm_homog_pts[:, 2:3]
    
    return norm_pts, T

def _fit_homography_dlt(src_pts, dst_pts):
    """
    Fit a homography matrix using the Normalized Direct Linear Transformation (DLT).
    src_pts, dst_pts: (N, 2) arrays
    """
    norm_src, T_src = _normalize_points(src_pts)
    norm_dst, T_dst = _normalize_points(dst_pts)
    
    N = src_pts.shape[0]
    A = []
    
    for i in range(N):
        x, y = norm_src[i]
        xp, yp = norm_dst[i]
        
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        
    A = np.array(A, dtype=float)
    
    # SVD
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    
    # Denormalize: H = T_dst^{-1} * H_norm * T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    
    # Scale H such that H[2, 2] = 1 (if H[2, 2] != 0)
    if H[2, 2] != 0:
        H = H / H[2, 2]
        
    return H

def ransac_homography(src, dst, samples, threshold):
    """
    Estimate homography using RANSAC with provided samples, finding the best inlier set,
    and refitting using normalized DLT on all inliers of the best sample.
    """
    N_pts = src.shape[0]
    best_inlier_count = -1
    best_inliers = None
    best_H = None
    
    for sample_indices in samples:
        src_sample = src[sample_indices]
        dst_sample = dst[sample_indices]
        
        # 1. Fit homography to sample
        try:
            H = _fit_homography_dlt(src_sample, dst_sample)
        except np.linalg.LinAlgError:
            continue
            
        # 2. Score on all correspondences with symmetric transfer error or forward transfer error
        # Here we use forward transfer error: || \pi(H x_i) - x_i' ||
        homog_src = np.hstack([src, np.ones((N_pts, 1))])
        projected = (H @ homog_src.T).T
        
        # Avoid division by zero
        denom = projected[:, 2:3]
        denom = np.where(denom == 0, 1e-12, denom)
        
        proj_pts = projected[:, :2] / denom
        
        errors = np.sqrt(np.sum((proj_pts - dst) ** 2, axis=1))
        inliers = errors < threshold
        inlier_count = np.sum(inliers)
        
        # 3. Keep sample with the most inliers; on tie keep earlier sample
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_H = H
            
    # 4. Refit with normalized DLT on all inliers of the best sample
    inlier_indices = np.where(best_inliers)[0]
    if len(inlier_indices) >= 4:
        refit_src = src[inlier_indices]
        refit_dst = dst[inlier_indices]
        final_H = _fit_homography_dlt(refit_src, refit_dst)
    else:
        final_H = best_H
        
    # Scale H2,2 to 1
    if final_H[2, 2] != 0:
        final_H = final_H / final_H[2, 2]
        
    return final_H, best_inliers