from collections import Counter
import numpy as np


def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    """Calculate METEOR score for machine translation evaluation.

    Args:
        reference: Reference translation string
        candidate: Candidate translation string
        alpha: Weight for precision vs recall in F-mean (default 0.9)
        beta: Exponent for fragmentation penalty (default 3)
        gamma: Maximum penalty coefficient (default 0.5)

    Returns:
        METEOR score between 0 and 1 rounded to 3 decimal places
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not ref_tokens or not cand_tokens:
        return 0.0

    # Step 1: Find exact unigram matches
    # Match candidate tokens to reference tokens greedily from left to right
    matched_cand_indices = []
    matched_ref_indices = []

    ref_used = [False] * len(ref_tokens)

    for i, c_word in enumerate(cand_tokens):
        for j, r_word in enumerate(ref_tokens):
            if not ref_used[j] and c_word == r_word:
                matched_cand_indices.append(i)
                matched_ref_indices.append(j)
                ref_used[j] = True
                break

    m = len(matched_cand_indices)
    if m == 0:
        return 0.0

    # Step 2: Compute Precision, Recall, and Weighted F-mean
    precision = m / len(cand_tokens)
    recall = m / len(ref_tokens)

    # Harmonic mean weighted heavily towards recall (default alpha = 0.9)
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    # Step 3: Count Chunks (contiguous, in-order matches in reference)
    # Sort matches by candidate token order
    cand_ref_pairs = sorted(zip(matched_cand_indices, matched_ref_indices))

    chunks = 1
    for i in range(1, len(cand_ref_pairs)):
        curr_cand, curr_ref = cand_ref_pairs[i]
        prev_cand, prev_ref = cand_ref_pairs[i - 1]

        # A chunk breaks if reference indices or candidate indices are not contiguous
        if curr_cand != prev_cand + 1 or curr_ref != prev_ref + 1:
            chunks += 1

    # Step 4: Compute Fragmentation Penalty
    penalty = gamma * ((chunks / m) ** beta)

    # Step 5: Final METEOR Score
    score = f_mean * (1 - penalty)

    return float(np.round(score, 3))