import numpy as np


def calculate_perplexity(probabilities: list[float]) -> float:
    """Calculate the perplexity of a language model given token probabilities.

    Args:
        probabilities: List of probabilities P(token_i | context) for each token
          in the sequence, where each probability is in (0, 1].

    Returns:
        Perplexity value as a float.
    """
    probs = np.asarray(probabilities, dtype=float)

    # Calculate average negative log-likelihood using natural logarithm
    avg_neg_log_likelihood = -np.mean(np.log(probs))

    # Perplexity is exponentiated average negative log-likelihood
    perplexity = np.exp(avg_neg_log_likelihood)

    return float(perplexity)


# Example test run
if __name__ == "__main__":
    probs = [0.5, 0.5, 0.5, 0.5]
    print(calculate_perplexity(probs))  # Output: 2.0