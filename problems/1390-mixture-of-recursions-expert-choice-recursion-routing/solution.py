import torch


def expert_choice_recursion(h, router_w, block, n_recursions, capacity):
    """
    Mixture-of-Recursions expert-choice routing.

    Args:
        h: Token states of shape (T, d)
        router_w: Router weights of shape (n_recursions, d)
        block: Callable mapping (k, d) -> (k, d)
        n_recursions: Maximum number of recursion steps
        capacity: Single int used at every step, or list of ints (length n_recursions)

    Returns:
        (h, depths): Updated states of shape (T, d) and int64 step counts of shape (T,)
    """
    h = h.clone()
    num_tokens = h.shape[0]
    depths = torch.zeros(num_tokens, dtype=torch.int64, device=h.device)
    active = torch.ones(num_tokens, dtype=torch.bool, device=h.device)

    if isinstance(capacity, (list, tuple)):
        capacities = list(capacity)
    else:
        capacities = [int(capacity)] * n_recursions

    for r in range(n_recursions):
        num_active = int(active.sum().item())
        step_cap = int(capacities[r])
        k = min(step_cap, num_active)
        if k <= 0:
            break

        scores = torch.sigmoid(h @ router_w[r])
        masked_scores = torch.where(
            active,
            scores,
            torch.full_like(scores, float("-inf")),
        )

        selected_scores, selected_idx = torch.topk(masked_scores, k=k, largest=True)

        selected_h = h[selected_idx]
        block_out = block(selected_h)
        h[selected_idx] = selected_h + selected_scores.unsqueeze(-1) * block_out

        depths[selected_idx] += 1

        next_active = torch.zeros_like(active)
        next_active[selected_idx] = True
        active = next_active

    return h, depths