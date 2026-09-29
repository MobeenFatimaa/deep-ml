def unigram_probability(corpus: str, word: str) -> float:
    """Calculate unigram probability of a target word in a corpus.

    Args:
        corpus: String of sentences containing <s> and </s> markers.
        word: The target word to calculate the probability for.

    Returns:
        Probability rounded to 4 decimal places.
    """
    # Tokenize corpus by splitting on whitespace
    tokens = corpus.split()

    if not tokens:
        return 0.0

    # Count occurrences of the target word
    word_count = tokens.count(word)

    # Total number of tokens including <s> and </s>
    total_tokens = len(tokens)

    # Calculate probability
    prob = word_count / total_tokens

    return round(prob, 4)


# Example test run
if __name__ == "__main__":
    corpus = "<s> Jack I like </s> <s> Jack I do like </s>"
    word = "Jack"
    print(
        unigram_probability(corpus, word)
    )  # Output: 0.1818 (2 occurrences / 11 tokens)