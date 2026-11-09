import numpy as np

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.classes = None
        self.class_log_prior = None
        self.feature_log_prob = None
        self.single_class = None

    def forward(self, X, y):
        """
        Fit the Bernoulli Naive Bayes model to binary features X and labels y.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Handle case where training data contains only one class
        if n_classes == 1:
            self.single_class = self.classes[0]
            return
        
        self.class_log_prior = np.zeros(n_classes)
        self.feature_log_prob = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            
            # Class prior using standard relative frequency (MLE)
            self.class_log_prior[idx] = np.log(n_c / n_samples)
            
            # Feature probability P(x_i = 1 | y = c) with Laplace smoothing
            feature_count_1 = np.sum(X_c, axis=0)
            prob_1 = (feature_count_1 + self.smoothing) / (n_c + 2 * self.smoothing)
            
            # Store log probabilities for feature being 1
            self.feature_log_prob[idx] = np.log(prob_1)

    def predict(self, X):
        """
        Predict class labels for test set X.
        """
        # If trained on a single class, return that class for all test samples
        if self.single_class is not None:
            return np.full(X.shape[0], self.single_class)
            
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for idx, c in enumerate(self.classes):
            log_p_1 = self.feature_log_prob[idx]
            # Use np.log1p(-np.exp(log_p_1)) for numerical stability when calculating log(1 - p)
            log_p_0 = np.log1p(-np.exp(log_p_1))
            
            # Bernoulli log likelihood: sum over features of [x_i * log(p_1) + (1 - x_i) * log(p_0)]
            likelihood = np.dot(X, log_p_1) + np.dot(1 - X, log_p_0)
            
            log_posteriors[:, idx] = self.class_log_prior[idx] + likelihood
            
        pred_indices = np.argmax(log_posteriors, axis=1)
        return self.classes[pred_indices]