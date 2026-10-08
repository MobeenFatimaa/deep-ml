def batch_requests(
    requests: list, max_batch_size: int, max_wait_time: float
) -> list:
    """Group inference requests into batches based on size and time constraints.

    Args:
        requests: List of dicts with 'id', 'timestamp', 'features'
        max_batch_size: Maximum number of requests per batch
        max_wait_time: Maximum time to wait before processing a batch

    Returns:
        List of tuples: (request_ids, batched_features, process_time)
    """
    if not requests:
        return []

    # Sort requests chronologically by timestamp
    sorted_requests = sorted(requests, key=lambda req: req["timestamp"])

    batches = []
    current_ids = []
    current_features = []
    batch_start_time = None

    for req in sorted_requests:
        req_id = req["id"]
        req_time = req["timestamp"]
        req_feat = req["features"]

        # First request in a new batch
        if not current_ids:
            current_ids.append(req_id)
            current_features.append(req_feat)
            batch_start_time = req_time
        else:
            # Check if including this request exceeds the max wait time from the batch start
            if (req_time - batch_start_time) > max_wait_time:
                # Finalize the current batch with the timestamp of its last included request
                last_req_time = sorted_requests[
                    sorted_requests.index(req) - 1
                ]["timestamp"]
                batches.append(
                    (current_ids, current_features, round(last_req_time, 4))
                )

                # Start a new batch with the current request
                current_ids = [req_id]
                current_features = [req_feat]
                batch_start_time = req_time
            else:
                current_ids.append(req_id)
                current_features.append(req_feat)

        # Check if the batch reached maximum capacity
        if len(current_ids) == max_batch_size:
            batches.append(
                (current_ids, current_features, round(req_time, 4))
            )
            current_ids = []
            current_features = []
            batch_start_time = None

    # Flush any remaining requests into a final batch
    if current_ids:
        batches.append(
            (
                current_ids,
                current_features,
                round(sorted_requests[-1]["timestamp"], 4),
            )
        )

    return batches