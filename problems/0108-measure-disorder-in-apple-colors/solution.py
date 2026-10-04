import math
from collections import Counter

def disorder(apples: list) -> float:
    """
    Compute the disorder in a basket of apples using Shannon Entropy.
    
    Args:
        apples: List of integers representing apple colors.
        
    Returns:
        Entropy value as a float representing disorder.
    """
    if not apples:
        return 0.0
    
    total = len(apples)
    counts = Counter(apples)
    
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
            
    return round(entropy, 4)