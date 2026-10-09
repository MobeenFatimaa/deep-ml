import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Count positive and negative instances
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    # Edge case: all labels belong to a single class
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    # Sort instances by predicted score in descending order
    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_indices]
    y_scores_sorted = y_scores[desc_indices]
    
    # Initialize ROC curve coordinates starting at (0, 0)
    fpr_list = [0.0]
    tpr_list = [0.0]
    
    tp = 0
    fp = 0
    
    n_samples = len(y_true)
    
    for i in range(n_samples):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            
        # Add point to ROC curve when score changes or at the final element
        # (Handles tied prediction scores correctly)
        if i == n_samples - 1 or y_scores_sorted[i] != y_scores_sorted[i + 1]:
            fpr_list.append(fp / n_neg)
            tpr_list.append(tp / n_pos)
            
    # Calculate Area Under the Curve using Trapezoidal Integration
    # Formula: np.trapz(tpr_list, fpr_list) or manual summation:
    auc = 0.0
    for i in range(1, len(fpr_list)):
        height = (tpr_list[i] + tpr_list[i - 1]) / 2.0
        width = fpr_list[i] - fpr_list[i - 1]
        auc += height * width
        
    return float(auc)