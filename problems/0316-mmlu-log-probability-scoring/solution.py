import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    predictions = []
    correct_probs = []
    
    for lp, correct in zip(log_probs, correct_answers):
        lp_arr = np.array(lp, dtype=float)
        
        # 1. Determine predicted answer choice (argmax of log-probabilities)
        pred = int(np.argmax(lp_arr))
        predictions.append(pred)
        
        # 2. Convert log-probabilities to probabilities using numerically stable softmax
        shifted_lp = lp_arr - np.max(lp_arr)
        exp_lp = np.exp(shifted_lp)
        probs = exp_lp / np.sum(exp_lp)
        
        # 3. Store probability assigned to the correct answer choice
        correct_probs.append(probs[correct])
        
    # Compute metrics
    accuracy = float(np.mean(np.array(predictions) == np.array(correct_answers)))
    avg_correct_prob = float(np.mean(correct_probs))
    
    return {
        'accuracy': round(accuracy, 4),
        'predictions': predictions,
        'avg_correct_prob': round(avg_correct_prob, 4)
    }