def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    best_candidates = []
    for prompt_candidates, prompt_scores in zip(candidates, scores):
        # Find the index of the highest score. 
        # max() preserves the first occurrence in case of ties.
        best_idx = max(range(len(prompt_scores)), key=lambda i: prompt_scores[i])
        best_candidates.append(prompt_candidates[best_idx])
        
    return best_candidates