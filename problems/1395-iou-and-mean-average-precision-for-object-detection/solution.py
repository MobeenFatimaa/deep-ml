import numpy as np


def iou(box_a, box_b) -> float:
    """
    Compute Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes.
    Returns 0.0 if they do not overlap.
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    if inter_area == 0.0:
        return 0.0
        
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    
    if union_area <= 0.0:
        return 0.0
        
    return float(inter_area / union_area)


def mean_average_precision(preds, gts, iou_thresh=0.5) -> float:
    """
    Compute Mean Average Precision (mAP) for a single image given predictions and ground truths.
    """
    gt_classes = set(gt[0] for gt in gts)
    if not gt_classes:
        return 0.0

    class_aps = []

    for cls in sorted(list(gt_classes)):
        # Extract ground truth boxes for this class
        cls_gts = [gt[1] for gt in gts if gt[0] == cls]
        num_gts = len(cls_gts)

        # Extract predictions for this class, sorted by score descending
        cls_preds = [p for p in preds if p[0] == cls]
        cls_preds.sort(key=lambda x: x[1], reverse=True)

        if not cls_preds:
            class_aps.append(0.0)
            continue

        gt_matched = [False] * num_gts
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        # Greedy matching of predictions to unmatched ground-truth boxes
        for i, pred in enumerate(cls_preds):
            pred_box = pred[2]
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt_box in enumerate(cls_gts):
                if not gt_matched[j]:
                    current_iou = iou(pred_box, gt_box)
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_gt_idx = j

            if best_iou >= iou_thresh and best_gt_idx != -1:
                gt_matched[best_gt_idx] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        precisions = cum_tp / (cum_tp + cum_fp)
        recalls = cum_tp / num_gts

        # Make precision monotonically non-increasing from the right
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Integrate over recall using the all-point method
        ap = 0.0
        prev_recall = 0.0
        for r, p in zip(recalls, precisions):
            ap += (r - prev_recall) * p
            prev_recall = r

        class_aps.append(ap)

    mean_ap = sum(class_aps) / len(class_aps)
    return round(float(mean_ap), 4)