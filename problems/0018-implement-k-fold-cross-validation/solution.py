import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Base size per fold and extra remainder samples
    fold_size = n_samples // k
    remainder = n_samples % k
    
    folds = []
    start = 0
    for i in range(k):
        # Distribute extra samples to the first 'remainder' folds
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end = start + current_fold_size
        
        # Select current fold as test indices
        test_idx = indices[start:end].tolist()
        
        # Combine all other folds as train indices
        train_idx = np.concatenate([indices[:start], indices[end:]]).tolist()
        
        folds.append((train_idx, test_idx))
        start = end
        
    return folds