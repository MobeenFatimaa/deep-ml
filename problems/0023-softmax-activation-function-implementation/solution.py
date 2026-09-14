import math

def softmax(scores: list[float]) -> list[float]:
    """Computes numerically stable softmax probabilities for a list of scores."""
    if not scores:
        return []
    
    # Subtract max(scores) for numerical stability (prevents overflow during exp)
    max_score = max(scores)
    exp_scores = [math.exp(x - max_score) for x in scores]
    
    sum_exp = sum(exp_scores)
    return [exp_score / sum_exp for exp_score in exp_scores]


# Example usage
scores = [1, 2, 3]
probs = softmax(scores)
print([round(p, 4) for p in probs])  # Output: [0.09, 0.2447, 0.6652]