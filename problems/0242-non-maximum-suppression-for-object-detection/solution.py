import numpy as np

def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes.
    
    Args:
        boxes: Array-like of shape (N, 4) with boxes in format [x1, y1, x2, y2]
        scores: Array-like of shape (N,) with confidence scores
        iou_threshold: float, IoU threshold for suppression (0 to 1)
    
    Returns:
        List of indices of kept boxes, ordered by descending score
        Returns -1 for invalid inputs
    """
    # 1. Validate iou_threshold
    if not isinstance(iou_threshold, (int, float)) or iou_threshold < 0 or iou_threshold > 1:
        return -1
    
    # 2. Convert to numpy arrays
    try:
        boxes = np.asarray(boxes, dtype=float)
        scores = np.asarray(scores, dtype=float)
    except Exception:
        return -1
    
    # 3. Check for empty input
    if boxes.size == 0 or scores.size == 0:
        if boxes.size == 0 and scores.size == 0:
            return []
        return -1
    
    # 4. Check array shapes and dimensions
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 1:
        return -1
    
    if len(boxes) != len(scores):
        return -1
    
    # 5. Check coordinate validity (x1 <= x2 and y1 <= y2)
    if np.any(boxes[:, 0] > boxes[:, 2]) or np.any(boxes[:, 1] > boxes[:, 3]):
        return -1
    
    # Extract coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute areas of all boxes
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort indices by confidence scores in descending order
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        # Pick the index with the highest confidence score
        i = order[0]
        keep.append(int(i))
        
        if order.size == 1:
            break
            
        # Coordinates of intersection boxes with remaining candidates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Width and height of intersection
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        
        # Compute Intersection over Union (IoU)
        union = areas[i] + areas[order[1:]] - intersection
        
        # Handle zero division (if union is 0, IoU is 0)
        iou = np.zeros_like(intersection)
        non_zero_union = union > 0
        iou[non_zero_union] = intersection[non_zero_union] / union[non_zero_union]
        
        # Keep boxes where IoU is less than or equal to threshold
        inds = np.where(iou <= iou_threshold)[0]
        
        # Offset index by +1 to match the slice order[1:]
        order = order[inds + 1]
        
    return keep