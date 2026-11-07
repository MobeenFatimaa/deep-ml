import itertools
from collections import defaultdict

def apriori(transactions, min_support=0.5, max_length=None):
    """
    Returns: dict mapping frozenset(itemset) -> support (float)
    """
    txs = [set(t) for t in transactions]
    num_transactions = len(txs)
    if num_transactions == 0:
        raise ValueError("Transactions list cannot be empty.")

    frequent_itemsets = {}
    
    # Step 1: Find frequent 1-itemsets
    item_counts = defaultdict(int)
    for tx in txs:
        for item in tx:
            item_counts[frozenset([item])] += 1

    current_k_itemsets = {}
    for itemset, count in item_counts.items():
        support = count / num_transactions
        if support >= min_support:
            current_k_itemsets[itemset] = support
            frequent_itemsets[itemset] = support

    k = 2
    while current_k_itemsets:
        if max_length is not None and k > max_length:
            break

        # Step 2: Generate candidate k-itemsets from frequent (k-1)-itemsets
        items = list(current_k_itemsets.keys())
        candidates = set()
        n = len(items)
        
        for i in range(n):
            for j in range(i + 1, n):
                union_set = items[i] | items[j]
                if len(union_set) == k:
                    # Apriori pruning: all (k-1) subsets must be frequent
                    subsets = [frozenset(comb) for comb in itertools.combinations(union_set, k - 1)]
                    if all(sub in current_k_itemsets for sub in subsets):
                        candidates.add(union_set)

        if not candidates:
            break

        # Step 3: Count support for candidates across all transactions
        candidate_counts = defaultdict(int)
        for tx in txs:
            for cand in candidates:
                if cand.issubset(tx):
                    candidate_counts[cand] += 1

        # Step 4: Filter candidates by min_support
        current_k_itemsets = {}
        for cand, count in candidate_counts.items():
            support = count / num_transactions
            if support >= min_support:
                current_k_itemsets[cand] = support
                frequent_itemsets[cand] = support

        k += 1

    return frequent_itemsets