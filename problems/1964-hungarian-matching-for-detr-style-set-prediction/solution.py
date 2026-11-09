import numpy as np

def linear_assignment(cost):
    """
    Solve the linear sum assignment problem using the Hungarian algorithm (Kuhn-Munkres).
    cost: (n, m) array
    Returns (rows, cols) sorted by row index as standard Python ints.
    """
    cost = np.array(cost, dtype=np.float64)
    n, m = cost.shape
    
    transposed = False
    if n > m:
        cost = cost.T
        n, m = cost.shape
        transposed = True

    size = max(n, m)
    C = np.zeros((size, size), dtype=np.float64)
    C[:n, :m] = cost
    
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)
    
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
                
    ans_rows = []
    ans_cols = []
    for j in range(1, size + 1):
        if p[j] != 0:
            r_idx = p[j] - 1
            c_idx = j - 1
            if r_idx < n and c_idx < m:
                if transposed:
                    ans_rows.append(int(c_idx))
                    ans_cols.append(int(r_idx))
                else:
                    ans_rows.append(int(r_idx))
                    ans_cols.append(int(c_idx))
                
    sorted_pairs = sorted(zip(ans_rows, ans_cols), key=lambda x: x[0])
    if sorted_pairs:
        r_out, c_out = zip(*sorted_pairs)
        return list(r_out), list(c_out)
    return [], []


def _box_cxcywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)


def _generalized_box_iou(boxes1, boxes2):
    """
    Compute Generalized IoU between boxes1 (N, 4) and boxes2 (M, 4) in [cx, cy, w, h] format.
    """
    b1_xyxy = _box_cxcywh_to_xyxy(boxes1)
    b2_xyxy = _box_cxcywh_to_xyxy(boxes2)
    
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    iou = np.zeros((N, M), dtype=np.float64)
    union = np.zeros((N, M), dtype=np.float64)
    enclose_area = np.zeros((N, M), dtype=np.float64)
    
    for i in range(N):
        b1 = b1_xyxy[i]
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        for j in range(M):
            b2 = b2_xyxy[j]
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            
            inter_x1 = max(b1[0], b2[0])
            inter_y1 = max(b1[1], b2[1])
            inter_x2 = min(b1[2], b2[2])
            inter_y2 = min(b1[3], b2[3])
            
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            
            u = area1 + area2 - inter_area
            union[i, j] = u
            iou[i, j] = inter_area / (u + 1e-12)
            
            enc_x1 = min(b1[0], b2[0])
            enc_y1 = min(b1[1], b2[1])
            enc_x2 = max(b1[2], b2[2])
            enc_y2 = max(b1[3], b2[3])
            enc_area = max(0.0, enc_x2 - enc_x1) * max(0.0, enc_y2 - enc_y1)
            enclose_area[i, j] = enc_area
            
    giou = iou - (enclose_area - union) / (enclose_area + 1e-12)
    return giou


def detr_cost(pred_logits, pred_boxes, gt_labels, gt_boxes, w_class=1.0, w_l1=5.0, w_giou=2.0):
    """
    Compute the bipartite matching cost matrix C_qg for DETR using high-precision float64.
    """
    Q = pred_logits.shape[0]
    G = gt_boxes.shape[0]
    
    if G == 0:
        return np.zeros((Q, 0), dtype=np.float64)
        
    pred_logits = np.array(pred_logits, dtype=np.float64)
    exp_logits = np.exp(pred_logits - np.max(pred_logits, axis=-1, keepdims=True))
    prob = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (Q, K+1)
    
    cost_class = -prob[:, gt_labels]  # (Q, G)
    cost_l1 = np.sum(np.abs(pred_boxes[:, None, :] - gt_boxes[None, :, :]), axis=-1)  # (Q, G)
    giou = _generalized_box_iou(pred_boxes, gt_boxes)  # (Q, G)
    
    C = w_class * cost_class + w_l1 * cost_l1 - w_giou * giou
    return C


def detr_match(pred_logits, pred_boxes, gt_labels, gt_boxes, w_class=1.0, w_l1=5.0, w_giou=2.0):
    """
    Solve the optimal bipartite matching between predictions and ground-truth objects.
    Returns (pred_idx, gt_idx) sorted by prediction index.
    """
    if gt_boxes.shape[0] == 0:
        return [], []
        
    C = detr_cost(pred_logits, pred_boxes, gt_labels, gt_boxes, w_class, w_l1, w_giou)
    rows, cols = linear_assignment(C)
    
    paired = sorted(zip(rows, cols), key=lambda x: x[0])
    pred_idx, gt_idx = zip(*paired) if paired else ([], [])
    return [int(x) for x in pred_idx], [int(x) for x in gt_idx]