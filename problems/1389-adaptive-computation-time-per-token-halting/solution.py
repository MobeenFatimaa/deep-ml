import torch


def act_halting(halt_probs, threshold=0.99):
    """
    Implements Adaptive Computation Time (ACT) per-token halting rule.
    
    Args:
        halt_probs: Tensor of shape (T, S) containing halting probabilities p_t(n)
        threshold: float, threshold for halting (default: 0.99)
        
    Returns:
        weights: Tensor of shape (T, S), weights w_t(n) summing to 1 per token
        n_steps: Tensor of shape (T,), int64, number of loops actually run (N_t + 1)
        ponder: Tensor of shape (T,), float, ponder cost (n_steps + R_t)
    """
    T, S = halt_probs.shape
    device = halt_probs.device
    
    # Compute cumulative sum along dim=1 (loops)
    cumsum = torch.cumsum(halt_probs, dim=1)
    
    # Find where cumulative sum >= threshold
    # To handle cases where it never reaches threshold, we can check if any exceed threshold
    reached = cumsum >= threshold
    
    # Find the first index where cumulative sum >= threshold for each row
    # If not reached anywhere, argmax returns 0, so we handle it using a mask or indicator
    has_reached = reached.any(dim=1)
    
    # First index of True along dim 1
    # argmax on boolean tensor returns the first True index
    n_t = torch.argmax(reached.to(torch.int), dim=1)
    
    # If a token never reaches the threshold, its stopping index should be S - 1
    n_t = torch.where(has_reached, n_t, torch.tensor(S - 1, device=device))
    
    # Create weights tensor of zeros
    weights = torch.zeros_like(halt_probs)
    
    # Build a mask for n < n_t
    # We can create a range tensor of shape (1, S)
    steps = torch.arange(S, device=device).unsqueeze(0).expand(T, S)
    
    # n < n_t mask
    lt_mask = steps < n_t.unsqueeze(1)
    weights = torch.where(lt_mask, halt_probs, weights)
    
    # For n == n_t, we need the remainder R_t = 1 - sum_{n < N_t} p_t(n)
    # Sum for n < N_t can be obtained from cumsum at n_t - 1, or 0 if n_t == 0
    # Let's compute running sum for n < n_t safely:
    # If n_t == 0, sum_{n < 0} is 0. Otherwise it is cumsum[t, n_t - 1]
    prev_cumsum = torch.zeros(T, device=device)
    # indices for n_t - 1 where n_t > 0
    gt_zero = n_t > 0
    valid_indices = torch.clamp(n_t - 1, min=0)
    prev_cumsum = torch.where(gt_zero, cumsum[torch.arange(T, device=device), valid_indices], torch.zeros_like(prev_cumsum))
    
    r_t = 1.0 - prev_cumsum
    
    # Assign remainder to weights at n_t
    # Create a mask for n == n_t
    eq_mask = steps == n_t.unsqueeze(1)
    weights = torch.where(eq_mask, r_t.unsqueeze(1), weights)
    
    # n_steps = N_t + 1 (int64)
    n_steps = n_t.to(torch.int64) + 1
    
    # ponder = n_steps + R_t
    ponder = n_steps.to(torch.float32) + r_t
    
    return weights, n_steps, ponder


def act_output(states, weights):
    """
    Computes the weighted read-out state.
    
    Args:
        states: Tensor of shape (T, S, d)
        weights: Tensor of shape (T, S)
        
    Returns:
        Weighted state tensor of shape (T, d)
    """
    # weights[:, :, None] expands shape to (T, S, 1) for broadcasting
    return torch.sum(states * weights.unsqueeze(-1), dim=1)
    