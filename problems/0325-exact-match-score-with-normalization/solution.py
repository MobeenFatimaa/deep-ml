import re
import string


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and collapsing whitespace."""
    # Convert text to lowercase
    text = text.lower()

    # Remove all punctuation characters
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse multiple whitespace characters and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """Calculate the exact match score between predictions and references.

    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings

    Returns:
        Exact match score as a float between 0.0 and 1.0
    """
    if not predictions or not references or len(predictions) != len(references):
        return 0.0

    matches = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        if normalize_text(pred) == normalize_text(ref):
            matches += 1

    return matches / total


# Example verification
if __name__ == "__main__":
    predictions = ["Hello, World!", "The answer is 42"]
    references = ["hello world", "the answer is 42"]

    score = exact_match_score(predictions, references)
    print(score)  # Output: 1.0