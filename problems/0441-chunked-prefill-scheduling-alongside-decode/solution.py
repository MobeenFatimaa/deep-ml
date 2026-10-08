def chunked_prefill_schedule(
    prefill_requests: list,
    decode_requests: list,
    max_batch_tokens: int,
    max_chunk_size: int,
) -> list:
    """Simulate chunked prefill scheduling alongside decode requests.

    Args:
        prefill_requests: list of {'id': str, 'prompt_tokens': int,
          'decode_tokens': int}
        decode_requests: list of {'id': str, 'remaining_decode': int}
        max_batch_tokens: int, max tokens per step
        max_chunk_size: int, max prefill chunk size

    Returns:
        list of step dicts with 'step', 'prefill_chunks', 'decode_ids',
        'total_tokens'
    """
    # Active decode pool: list of dicts with 'id' and 'remaining_decode'
    active_decode = [
        {"id": req["id"], "remaining_decode": req["remaining_decode"]}
        for req in decode_requests
        if req["remaining_decode"] > 0
    ]

    # Active prefill queue: list of dicts tracking remaining prompt tokens
    prefill_queue = [
        {
            "id": req["id"],
            "remaining_prompt": req["prompt_tokens"],
            "decode_tokens": req["decode_tokens"],
        }
        for req in prefill_requests
        if req["prompt_tokens"] > 0
    ]

    schedule = []
    step = 0

    while active_decode or prefill_queue:
        step_prefill_chunks = []
        step_decode_ids = []

        # 1. Schedule Decode Requests First (1 token per active decode request)
        num_decodes = len(active_decode)
        tokens_budget = max_batch_tokens - num_decodes

        for req in active_decode:
            step_decode_ids.append(req["id"])
            req["remaining_decode"] -= 1

        # Track requests transitioning from prefill to decode for the NEXT step
        newly_completed_prefills = []

        # 2. Schedule Prefill Chunks with Remaining Token Budget
        idx = 0
        while idx < len(prefill_queue) and tokens_budget > 0:
            p_req = prefill_queue[idx]

            # Determine chunk size: min(remaining_prompt, max_chunk_size, tokens_budget)
            chunk_size = min(
                p_req["remaining_prompt"], max_chunk_size, tokens_budget
            )

            if chunk_size > 0:
                step_prefill_chunks.append((p_req["id"], chunk_size))
                p_req["remaining_prompt"] -= chunk_size
                tokens_budget -= chunk_size

                # Check if prefill finished for this request
                if p_req["remaining_prompt"] == 0:
                    completed = prefill_queue.pop(idx)
                    if completed["decode_tokens"] > 0:
                        newly_completed_prefills.append(
                            {
                                "id": completed["id"],
                                "remaining_decode": completed["decode_tokens"],
                            }
                        )
                    continue  # Do not increment idx since an item was popped

            idx += 1

        # 3. Clean up Decode Requests that finished during this step
        active_decode = [
            req for req in active_decode if req["remaining_decode"] > 0
        ]

        # 4. Add newly completed prefill requests to decode pool starting next step
        active_decode.extend(newly_completed_prefills)

        # 5. Record Step Summary
        total_tokens = sum(c[1] for c in step_prefill_chunks) + len(
            step_decode_ids
        )
        schedule.append(
            {
                "step": step,
                "prefill_chunks": step_prefill_chunks,
                "decode_ids": step_decode_ids,
                "total_tokens": total_tokens,
            }
        )

        step += 1

    return schedule