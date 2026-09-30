def prepare_eval_input(tokens: list, mode: str, reserved_output: int) -> list:
    """
    Truncate a token list to fit the context window of the given reasoning mode.
    """
    context_budgets = {
        "non-think": 8192,
        "high": 131072,
        "max": 393216
    }
    
    if mode not in context_budgets:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {list(context_budgets.keys())}")
        
    context_window = context_budgets[mode]
    max_input_length = context_window - reserved_output
    
    # If reserved output takes up the full budget or more
    if max_input_length <= 0:
        return []
        
    # Left-truncate to keep the most recent context if it exceeds max input length
    if len(tokens) > max_input_length:
        return tokens[-max_input_length:]
        
    return tokens