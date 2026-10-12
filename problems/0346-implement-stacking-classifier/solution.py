import numpy as np

def stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        base_classifiers: list, meta_classifier, n_folds: int = 5) -> np.ndarray:
    """
    Implement a stacking classifier ensemble.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        X_test: Test features of shape (m_samples, n_features)
        base_classifiers: List of classifier functions
        meta_classifier: Meta-level classifier function
        n_folds: Number of cross-validation folds
    
    Returns:
        np.ndarray: Final predictions on X_test
    """
    n_samples = len(X_train)
    n_models = len(base_classifiers)
    
    # 1. Generate out-of-fold predictions for training meta-features
    meta_X_train = np.zeros((n_samples, n_models))
    
    # Compute fold sizes and ranges (contiguous folds without shuffling)
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1
    
    fold_indices = []
    current = 0
    for size in fold_sizes:
        fold_indices.append((current, current + size))
        current += size
        
    for fold_idx, (start, end) in enumerate(fold_indices):
        val_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(0, start), np.arange(end, n_samples)])
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val = X_train[val_idx]
        
        for model_idx, clf in enumerate(base_classifiers):
            preds = clf(X_tr, y_tr, X_val)
            meta_X_train[val_idx, model_idx] = preds
            
    # 2. Generate test predictions from base classifiers trained on full training data
    meta_X_test = np.zeros((len(X_test), n_models))
    for model_idx, clf in enumerate(base_classifiers):
        preds = clf(X_train, y_train, X_test)
        meta_X_test[:, model_idx] = preds
        
    # 3. Train meta-classifier on out-of-fold predictions and predict on test meta-features
    final_preds = meta_classifier(meta_X_train, y_train, meta_X_test)
    
    return np.array(final_preds)