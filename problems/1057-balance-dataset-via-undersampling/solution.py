from collections import Counter

def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    if not data:
        return []
    
    # Step 1: Count occurrences of each label
    label_counts = Counter(label for _, label in data)
    
    # Step 2: Determine the minimum count across all class labels
    min_count = min(label_counts.values())
    
    # Step 3: Keep track of how many samples we've kept per class
    kept_counts = Counter()
    balanced_data = []
    
    # Step 4: Iterate through input once to preserve original order
    for sample, label in data:
        if kept_counts[label] < min_count:
            balanced_data.append((sample, label))
            kept_counts[label] += 1
            
    return balanced_data