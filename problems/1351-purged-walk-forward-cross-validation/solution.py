import numpy as np


def purged_walk_forward_splits(n_samples: int, n_splits: int, embargo: int) -> list:
    """Build expanding-window train/test splits with an embargo gap.

    Args:
        n_samples (int): total number of chronologically ordered samples.
        n_splits (int): number of test folds.
        embargo (int): samples immediately before each test block to drop from training.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: one (train_idx, test_idx) pair per fold.
    """
    fold_size = n_samples // (n_splits + 1)
    splits = []
    
    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        # The last fold's test block runs all the way to n_samples
        if k == n_splits:
            test_end = n_samples
        else:
            test_end = (k + 1) * fold_size
            
        test_idx = np.arange(test_start, test_end)
        
        # Training includes everything before the test block minus the embargo, clamped at 0
        train_end = max(0, test_start - embargo)
        train_idx = np.arange(0, train_end)
        
        splits.append((train_idx, test_idx))
        
    return splits