def label_encode_ordinal(values: list, order: list) -> list:
    """
    Encode ordinal categorical values to integers based on specified order.
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    
    Returns:
        List of integers representing the encoded values
    """
    # Create a mapping dictionary for O(1) lookups
    mapping = {category: rank for rank, category in enumerate(order)}
    
    # Map each value to its rank index, defaulting to -1 if unknown
    return [mapping.get(val, -1) for val in values]