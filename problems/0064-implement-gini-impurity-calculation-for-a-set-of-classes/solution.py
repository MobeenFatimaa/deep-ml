import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    if len(y) == 0:
        return 0.0
    
    # Convert to numpy array
    y = np.array(y)
    
    # Get the unique classes and their respective counts
    _, counts = np.unique(y, return_counts=True)
    
    # Calculate probabilities for each class
    probabilities = counts / len(y)
    
    # Calculate Gini impurity: 1 - sum(p_i^2)
    impurity = 1.0 - np.sum(probabilities ** 2)
    
    return round(float(impurity), 3)