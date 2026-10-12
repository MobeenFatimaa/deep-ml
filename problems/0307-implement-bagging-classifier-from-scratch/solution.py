import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_err = float('inf')
        
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            # Consider midpoints between sorted unique values as thresholds, 
    # plus boundaries to handle all thresholds correctly
            sorted_vals = np.sort(np.unique(X_column))
            if len(sorted_vals) > 1:
                thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
                # Also include values slightly below min or above max
                thresholds = np.concatenate([
                    [sorted_vals[0] - 1.0], 
                    thresholds, 
                    [sorted_vals[-1] + 1.0]
                ])
            else:
                thresholds = sorted_vals
            
            for thresh in thresholds:
                for polarity in [1, -1]:
                    predictions = np.zeros(n_samples)
                    if polarity == 1:
                        predictions[X_column >= thresh] = 1
                    else:
                        predictions[X_column < thresh] = 1
                        
                    err = np.sum(predictions != y)
                    
                    if err < best_err:
                        best_err = err
                        self.feature_idx = feat_idx
                        self.threshold = thresh
                        self.polarity = polarity
                        
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.zeros(n_samples)
        if self.polarity == 1:
            predictions[X_column >= self.threshold] = 1
        else:
            predictions[X_column < self.threshold] = 1
        return predictions.astype(int)

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    np.random.seed(seed)
    n_samples = X_train.shape[0]
    
    stumps = []
    for _ in range(n_estimators):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        stump = DecisionStump()
        stump.fit(X_boot, y_boot)
        stumps.append(stump)
        
    test_preds = np.zeros((X_test.shape[0], n_estimators))
    for i, stump in enumerate(stumps):
        test_preds[:, i] = stump.predict(X_test)
        
    # Majority voting with tie-breaking towards 0 or average rule
    final_preds = (np.mean(test_preds, axis=1) >= 0.5).astype(int)
    return final_preds