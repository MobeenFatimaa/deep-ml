import bisect


def random_pick_with_weight(weights, r):
    total = sum(weights)
    target = r * total

    # Compute prefix sums
    prefix_sums = []
    curr = 0
    for w in weights:
        curr += w
        prefix_sums.append(curr)

    # Find the smallest index i such that prefix_sums[i] > target
    # bisect_right finds the insertion point to the right, which gives elements strictly greater than target
    return bisect.bisect_right(prefix_sums, target)