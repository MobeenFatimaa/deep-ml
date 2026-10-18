import math
from collections import Counter


def bleu_score(
    candidate: list[str], references: list[list[str]], max_n: int = 4
) -> float:
    """Calculate BLEU score for candidate text against reference texts.

    Args:
        candidate: Candidate sentence as a list of tokens.
        references: List of reference sentences, each as a list of tokens.
        max_n: Maximum n-gram order to evaluate.

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    cand_len = len(candidate)

    if cand_len == 0 or not references:
        return 0.0

    # 1. Compute modified n-gram precisions for n = 1 to max_n
    precisions = []

    for n in range(1, max_n + 1):
        if cand_len < n:
            return 0.0

        # Extract n-grams for candidate
        cand_ngrams = [
            tuple(candidate[i : i + n]) for i in range(cand_len - n + 1)
        ]
        cand_ngram_counts = Counter(cand_ngrams)

        # Find maximum count of each n-gram across all references
        max_ref_counts = {}
        for ref in references:
            ref_ngrams = [
                tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)
            ]
            ref_ngram_counts = Counter(ref_ngrams)

            for ngram in cand_ngram_counts:
                max_ref_counts[ngram] = max(
                    max_ref_counts.get(ngram, 0), ref_ngram_counts.get(ngram, 0)
                )

        # Clip candidate n-gram counts
        clipped_count = sum(
            min(count, max_ref_counts.get(ngram, 0))
            for ngram, count in cand_ngram_counts.items()
        )
        total_cand_ngrams = len(cand_ngrams)

        if clipped_count == 0:
            return 0.0

        p_n = clipped_count / total_cand_ngrams
        precisions.append(p_n)

    # 2. Select effective reference length (r) for Brevity Penalty (BP)
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lengths, key=lambda r: (abs(r - cand_len), r)
    )

    # Compute Brevity Penalty
    if cand_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (closest_ref_len / cand_len))

    # 3. Combine precisions using geometric mean and apply BP
    log_sum = sum(math.log(p) for p in precisions)
    geo_mean = math.exp(log_sum / max_n)

    bleu = bp * geo_mean
    return bleu


# Example verification
if __name__ == "__main__":
    result = bleu_score(
        ["the", "cat", "sat", "on", "the", "mat"],
        [["the", "cat", "sat", "on", "the", "mat"]],
        max_n=4,
    )
    print(round(result, 4))  # Output: 1.0