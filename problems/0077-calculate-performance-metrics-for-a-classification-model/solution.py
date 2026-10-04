def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    # Initialize confusion matrix components
    tp = 0  # True Positive: actual = 1, predicted = 1
    fp = 0  # False Positive: actual = 0, predicted = 1
    fn = 0  # False Negative: actual = 1, predicted = 0
    tn = 0  # True Negative: actual = 0, predicted = 0

    # Count classification outcomes
    for a, p in zip(actual, predicted):
        if a == 1 and p == 1:
            tp += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 1 and p == 0:
            fn += 1
        elif a == 0 and p == 0:
            tn += 1

    # Confusion matrix format: [[TP, FN], [FP, TN]]
    confusion_matrix = [[tp, fn], [fp, tn]]

    total = len(actual)

    # 1. Accuracy = (TP + TN) / Total
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # 2. Precision and Recall for F1 Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 3. Specificity (True Negative Rate) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 4. Negative Predictive Value (NPV) = TN / (TN + FN)
    negativePredictive = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)