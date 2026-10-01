from collections import defaultdict

def run_etl(csv_text: str) -> list[tuple[str, float]]:
    """Run a simple ETL pipeline over CSV text with header user_id,event_type,value.

    Returns a sorted list of (user_id, total_value) for event_type == "purchase".
    """
    user_totals = defaultdict(float)

    # 1. Extract: Split lines and handle blank lines
    lines = csv_text.strip().splitlines()
    if not lines:
        return []

    # Skip header
    data_lines = lines[1:]

    for line in data_lines:
        line = line.strip()
        if not line:
            continue  # Ignore blank lines

        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 3:
            continue  # Skip malformed lines

        user_id, event_type, value_str = parts

        # 2. Transform: Filter, Type Cast, Aggregate
        if event_type == "purchase":
            try:
                val = float(value_str)
                user_totals[user_id] += val
            except ValueError:
                # Drop invalid numeric values (e.g., 'not_a_number')
                continue

    # 3. Load: Format and sort by user_id ascending
    sorted_results = sorted(user_totals.items(), key=lambda x: x[0])
    return sorted_results