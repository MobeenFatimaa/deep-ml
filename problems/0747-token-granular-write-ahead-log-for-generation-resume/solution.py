def resume_generation(prompt, wal, max_new_tokens, vocab_size, eos):
    """
    Resume deterministic token generation using a Write-Ahead Log.

    Args:
        prompt: list of ints, initial prompt tokens.
        wal: list of ints, tokens already generated before interruption.
        max_new_tokens: int, max total number of generated tokens (not counting prompt).
        vocab_size: int, modulus for the next-token rule.
        eos: int, end-of-sequence token id.

    Returns:
        list of ints: prompt followed by the full generated sequence.
    """
    # Check if WAL already contains the EOS token
    if eos in wal:
        eos_index = wal.index(eos)
        truncated_wal = wal[:eos_index + 1]
        return prompt + truncated_wal

    # Otherwise, start generation using prompt + existing wal
    generated = list(wal)
    context = list(prompt) + generated

    while len(generated) < max_new_tokens:
        # Determine the next token using the context state
        n = len(context)
        if n >= 2:
            next_token = (context[-1] + context[-2]) % vocab_size
        elif n == 1:
            next_token = context[-1] % vocab_size
        else:
            next_token = 0

        # Append to context and generated tokens
        context.append(next_token)
        generated.append(next_token)

        # Stop if EOS token is produced
        if next_token == eos:
            break

    return prompt + generated