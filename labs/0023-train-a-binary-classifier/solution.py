import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_train, 30) -- standardized features
        y_train: numpy array of shape (n_train,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized validation features
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # 1. Combine train and validation sets for maximum training data
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    
    # 2. Define linear models with high accuracy on standardized Breast Cancer data
    lr = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
    svm = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    
    # 3. Create a Voting Classifier ensemble for robust predictions
    model = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm)],
        voting='soft'
    )
    
    # 4. Fit the model on the full dataset
    model.fit(X_full, y_full)
    
    # 5. Define the predict closure
    def predict(X):
        return model.predict(X)
    
    return predict