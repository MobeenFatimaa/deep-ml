def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    # Dense layer computes across all experts/parameters equivalent
    dense_flops = n_experts * d_in * d_out
    
    # MoE layer only computes for the k active experts per token
    moe_flops = k_active * d_in * d_out
    
    # Calculate percentage savings
    savings_percentage = ((dense_flops - moe_flops) / dense_flops) * 100.0
    
    return savings_percentage