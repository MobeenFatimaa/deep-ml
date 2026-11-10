from collections import Counter

def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    # 1. Tokenize by lowercasing and splitting on whitespace
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    # 2. Count token frequencies
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    
    # 3. Compute clipped overlap counts
    overlap = 0
    for w, count_cand in cand_counts.items():
        if w in ref_counts:
            overlap += min(ref_counts[w], count_cand)
            
    len_ref = len(ref_tokens)
    len_cand = len(cand_tokens)
    
    # 4. Compute precision, recall, and F1 with division-by-zero protection
    precision = (overlap / len_cand) if len_cand > 0 else 0.0
    recall = (overlap / len_ref) if len_ref > 0 else 0.0
    
    if (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }