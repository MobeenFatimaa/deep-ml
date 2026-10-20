def confusion_matrix(data):
    tp = fn = fp = tn = 0
    for y_true, y_pred in data:
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
    return [[tp, fn], [fp, tn]]