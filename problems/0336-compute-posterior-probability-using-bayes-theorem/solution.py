def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.
    
    Args:
        priors: Prior probabilities P(H_i) for each hypothesis
        likelihoods: Likelihoods P(E|H_i) for each hypothesis
        
    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis
    """
    # Calculate unnormalized posteriors: P(E|H_i) * P(H_i)
    unnormalized = [p * l for p, l in zip(priors, likelihoods)]
    
    # Calculate total evidence P(E) = sum(P(E|H_j) * P(H_j))
    total_evidence = sum(unnormalized)
    
    if total_evidence == 0:
        return [0.0 for _ in priors]
        
    # Normalize to get posterior probabilities P(H_i|E)
    return [val / total_evidence for val in unnormalized]