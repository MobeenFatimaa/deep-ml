def conditional_probability(joint_distribution: dict) -> float:
    """
    Compute conditional probability P(A|B) from a joint probability distribution.

    Args:
        joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

    Returns:
        float: Conditional probability P(A|B)
    """
    p_a_and_b = float(joint_distribution.get(("A", "B"), 0.0))
    p_not_a_and_b = float(joint_distribution.get(("`A", "B"), 0.0))

    p_b = p_a_and_b + p_not_a_and_b
    if p_b == 0.0:
        raise ZeroDivisionError("Marginal probability P(B) is zero; P(A|B) is undefined.")

    return round(p_a_and_b / p_b, 4)