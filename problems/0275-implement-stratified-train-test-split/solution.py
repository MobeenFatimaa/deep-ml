import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    X = np.asarray(X)
    y = np.asarray(y)
    
    unique_classes = np.unique(y)
    
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    
    for c in unique_classes:
        # Find indices belonging to the current class
        class_indices = np.where(y == c)[0]
        
        # Shuffle indices for this class
        shuffled_indices = np.random.permutation(class_indices)
        
        n_class_samples = len(shuffled_indices)
        n_test = int(n_class_samples * test_size)
        
        # Split into test and train indices for this class
        test_indices = shuffled_indices[:n_test]
        train_indices = shuffled_indices[n_test:]
        
        # Append corresponding subsets
        X_test_list.append(X[test_indices])
        y_test_list.append(y[test_indices])
        X_train_list.append(X[train_indices])
        y_train_list.append(y[train_indices])
        
    # Concatenate all class subsets together
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # Optional: shuffle the combined train and test sets so classes aren't ordered consecutively
    train_perm = np.random.permutation(len(y_train))
    test_perm = np.random.permutation(len(y_test))
    
    return X_train[train_perm], X_test[test_perm], y_train[train_perm], y_test[test_perm]