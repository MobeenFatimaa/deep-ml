def vector_sum(a: list[int | float], b: list[int | float]) -> list[int | float] | int:
    """
    Returns the element-wise sum of vectors 'a' and 'b'.
    If vectors have different lengths, returns -1.
    """
    if len(a) != len(b):
        return -1
    return [x + y for x, y in zip(a, b)]