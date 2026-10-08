import numpy as np


def _is_valid_type(val, expected_type: str, nullable: bool) -> bool:
    """Helper function to validate data types against schema specifications."""
    # Null handling
    if val is None:
        return nullable

    # Python bool inherits from int (isinstance(True, int) is True),
    # so we explicitly exclude bools from numeric types.
    if expected_type == "numeric":
        if isinstance(val, bool):
            return False
        if isinstance(val, (int, float)):
            return not np.isnan(val)
        return False
    elif expected_type == "categorical":
        return isinstance(val, str)
    elif expected_type == "boolean":
        return isinstance(val, bool)

    return False


def calculate_data_quality_score(data: list, schema: dict) -> dict:
    """Calculate data quality metrics for ML pipeline monitoring.

    Args:
        data: list of dictionaries representing rows of data schema: dictionary
        defining expected columns and their types {'column_name': {'type':
        'numeric'|'categorical'|'boolean', 'nullable': True|False}}

    Returns:
        dict with keys: 'completeness', 'type_validity', 'uniqueness_ratio',
        'overall_score' All values as percentages (0-100), rounded to 2 decimal
        places.
    """
    if not data:
        return {}

    n_records = len(data)
    schema_cols = list(schema.keys())
    total_expected_cells = n_records * len(schema_cols)

    if total_expected_cells == 0:
        return {}

    non_null_count = 0
    type_valid_count = 0

    for record in data:
        for col in schema_cols:
            val = record.get(col)
            spec = schema[col]

            # 1. Completeness Check
            if val is not None:
                non_null_count += 1

            # 2. Type Validity Check
            if _is_valid_type(val, spec["type"], spec["nullable"]):
                type_valid_count += 1

    completeness = (non_null_count / total_expected_cells) * 100.0
    type_validity = (type_valid_count / total_expected_cells) * 100.0

    # 3. Uniqueness Ratio Check
    # Convert records into hashable tuples of sorted key-value pairs
    unique_records = len(
        set(tuple(sorted(record.items())) for record in data)
    )
    uniqueness_ratio = (unique_records / n_records) * 100.0

    # 4. Overall Score (40% completeness, 40% type validity, 20% uniqueness)
    overall_score = (
        0.4 * completeness + 0.4 * type_validity + 0.2 * uniqueness_ratio
    )

    return {
        "completeness": round(completeness, 2),
        "type_validity": round(type_validity, 2),
        "uniqueness_ratio": round(uniqueness_ratio, 2),
        "overall_score": round(overall_score, 2),
    }