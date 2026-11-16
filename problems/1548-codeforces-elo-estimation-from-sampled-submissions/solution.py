import numpy as np


def submission_score(outcomes, penalty_medians):
    """
    Score of a submission sequence (booleans, in order): penalty_medians[min(k, len - 1)] for the first
    solved submission after k failures, else 0.0.
    """
    failures = 0
    for solved in outcomes:
        if solved:
            idx = min(failures, len(penalty_medians) - 1)
            return float(penalty_medians[idx])
        else:
            failures += 1
    return 0.0


def contest_rank(score, human_scores):
    """
    1 + number of human contest scores strictly greater than score.
    """
    return 1 + sum(1 for h in human_scores if h > score)


def performance_rating(rank, human_ratings, lo=0.0, hi=4000.0, iters=50):
    """
    Rating R with seed(R) = 1 + sum_j 1 / (1 + 10 ** ((R - r_j) / 400)) equal to rank,
    by `iters` bisection steps on [lo, hi]; returns the final midpoint.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        seed = 1.0 + np.sum(1.0 / (1.0 + 10.0 ** ((mid - np.array(human_ratings)) / 400.0)))
        if seed > rank:
            # Expected seed is higher than target rank -> rating R is too low
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def estimate_elo(results, penalty_medians, human_scores, human_ratings, n_samples=100, n_pick=10, seed=0):
    """
    Monte-Carlo expected rating: n_samples contests, each sampling n_pick of the candidates per problem
    without replacement in random order with np.random.default_rng(seed).
    Returns {"expected_rating", "mean_score", "min_rating", "max_rating"} rounded to 2 decimals.
    """
    rng = np.random.default_rng(seed)
    ratings = []
    scores = []
    
    for _ in range(n_samples):
        contest_score = 0.0
        for p, outcomes in enumerate(results):
            n_c = len(outcomes)
            k = min(n_pick, n_c)
            # Sample candidate submission indices without replacement in random order
            perm = rng.permutation(n_c)[:k]
            sub_outcomes = [outcomes[i] for i in perm]
            score = submission_score(sub_outcomes, penalty_medians[p])
            contest_score += score
            
        rank = contest_rank(contest_score, human_scores)
        rating = performance_rating(rank, human_ratings)
        ratings.append(rating)
        scores.append(contest_score)
        
    return {
        "expected_rating": round(float(np.mean(ratings)), 2),
        "mean_score": round(float(np.mean(scores)), 2),
        "min_rating": round(float(np.min(ratings)), 2),
        "max_rating": round(float(np.max(ratings)), 2)
    }