def wave_pipeline_makespan(transfer_times, compute_times, send_times):
    """
    Compute the makespan of the three-stage wave pipeline.

    Args:
        transfer_times: list of per-wave token transfer durations
        compute_times: list of per-wave expert compute durations
        send_times: list of per-wave result send durations

    Returns:
        Total makespan (number).
    """
    if not transfer_times:
        return 0

    N = len(transfer_times)
    
    x_end = 0.0
    c_end = 0.0
    s_end = 0.0

    for i in range(N):
        t_trans = transfer_times[i]
        t_comp = compute_times[i]
        t_send = send_times[i]

        # Transfer stage
        x_start = x_end
        x_end = x_start + t_trans

        # Compute stage
        c_start = max(x_end, c_end)
        c_end = c_start + t_comp

        # Send stage
        s_start = max(c_end, s_end)
        s_end = s_start + t_send

    return s_end