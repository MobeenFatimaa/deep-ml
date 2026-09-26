def find_duplicates(records):
    """
    Find duplicate items in records, maintaining the order of their second occurrence.
    
    Args:
        records: list of hashable items (ints or strings)
        
    Returns:
        list of items that appear more than once, each listed once
    """
    seen = set()
    duplicates_seen = set()
    result = []
    
    for item in records:
        if item in seen:
            if item not in duplicates_seen:
                duplicates_seen.add(item)
                result.append(item)
        else:
            seen.add(item)
            
    return result