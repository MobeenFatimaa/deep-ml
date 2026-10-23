import numpy as np

def collate_preference_batch(batch, pad_token_id=0):
    """
    Collate a list of preference examples into padded tensors/lists and masks for DPO.
    
    Args:
        batch: list of dicts with keys 'prompt', 'chosen', 'rejected'
        pad_token_id: int, token id used for padding
        
    Returns:
        dict with keys 'chosen_input_ids', 'rejected_input_ids', 'chosen_mask', 'rejected_mask'
        as nested Python lists.
    """
    if not batch:
        return {
            'chosen_input_ids': [],
            'rejected_input_ids': [],
            'chosen_mask': [],
            'rejected_mask': []
        }

    chosen_sequences = []
    rejected_sequences = []
    prompt_lens = []
    chosen_lens = []
    rejected_lens = []

    for item in batch:
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        prompt_len = len(prompt)
        prompt_lens.append(prompt_len)
        chosen_lens.append(len(chosen))
        rejected_lens.append(len(rejected))
        
        chosen_sequences.append(prompt + chosen)
        rejected_sequences.append(prompt + rejected)

    max_len = max(
        max(len(seq) for seq in chosen_sequences),
        max(len(seq) for seq in rejected_sequences)
    )

    chosen_input_ids = []
    rejected_input_ids = []
    chosen_mask = []
    rejected_mask = []

    for i in range(len(batch)):
        c_seq = chosen_sequences[i]
        r_seq = rejected_sequences[i]
        p_len = prompt_lens[i]
        c_len = chosen_lens[i]
        r_len = rejected_lens[i]

        # Right pad chosen sequence
        c_padded = c_seq + [pad_token_id] * (max_len - len(c_seq))
        chosen_input_ids.append(c_padded)

        # Right pad rejected sequence
        r_padded = r_seq + [pad_token_id] * (max_len - len(r_seq))
        rejected_input_ids.append(r_padded)

        # Build chosen mask
        c_m = [False] * max_len
        for t in range(p_len, p_len + c_len):
            c_m[t] = True
        chosen_mask.append(c_m)

        # Build rejected mask
        r_m = [False] * max_len
        for t in range(p_len, p_len + r_len):
            r_m[t] = True
        rejected_mask.append(r_m)

    return {
        'chosen_input_ids': chosen_input_ids,
        'rejected_input_ids': rejected_input_ids,
        'chosen_mask': chosen_mask,
        'rejected_mask': rejected_mask
    }