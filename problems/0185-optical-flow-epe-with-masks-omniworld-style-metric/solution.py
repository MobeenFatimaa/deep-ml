from typing import Optional, Union

try:
    import numpy as np
except Exception:
    np = None

ArrayLike = Union[list, "np.ndarray"]

def flow_epe(pred: ArrayLike,
             gt: ArrayLike,
             mask: Optional[ArrayLike] = None,
             max_flow: Optional[float] = None) -> float:
    """
    Compute mean End-Point Error (EPE) between predicted and ground-truth optical flow.

    Args:
        pred, gt: (H, W, 2) lists or NumPy arrays.
        mask: optional (H, W) or broadcastable to (H, W); 1=include, 0=ignore.
        max_flow: optional float; clip per-pixel EPE to this value.

    Returns:
        float: mean EPE over valid pixels. Returns -1 on invalid input or if no valid pixels.
    """
    if np is None:
        return -1

    # 1. Convert inputs to NumPy float arrays safely
    try:
        pred_arr = np.asarray(pred, dtype=np.float64)
        gt_arr = np.asarray(gt, dtype=np.float64)
    except Exception:
        return -1

    # 2. Validate shapes
    if pred_arr.ndim != 3 or pred_arr.shape[2] != 2:
        return -1
    if pred_arr.shape != gt_arr.shape:
        return -1

    H, W, _ = pred_arr.shape

    # 3. Handle validity mask
    valid_mask = np.ones((H, W), dtype=bool)

    if mask is not None:
        try:
            mask_arr = np.asarray(mask)
            # Remove trailing 1s in mask if shape is (H, W, 1)
            if mask_arr.ndim == 3 and mask_arr.shape[2] == 1:
                mask_arr = mask_arr.squeeze(axis=2)
            
            if mask_arr.shape != (H, W):
                return -1
            
            valid_mask = valid_mask & (mask_arr > 0)
        except Exception:
            return -1

    # 4. Exclude NaN and Inf values
    valid_pred = np.isfinite(pred_arr).all(axis=2)
    valid_gt = np.isfinite(gt_arr).all(axis=2)
    valid_mask = valid_mask & valid_pred & valid_gt

    if not np.any(valid_mask):
        return -1.0

    # 5. Compute per-pixel End-Point Error (EPE)
    diff = pred_arr[valid_mask] - gt_arr[valid_mask]
    epe = np.linalg.norm(diff, axis=1)

    # 6. Clip EPE values if max_flow is specified
    if max_flow is not None:
        if not isinstance(max_flow, (int, float)) or max_flow < 0:
            return -1
        epe = np.minimum(epe, float(max_flow))

    return float(np.mean(epe))