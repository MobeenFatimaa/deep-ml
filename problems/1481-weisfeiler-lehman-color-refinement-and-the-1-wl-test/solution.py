from collections import Counter


def wl_refine(adj, colors):
    """Refine node colours for one round of the 1-WL test."""
    signatures = []
    for i in range(len(adj)):
        neigh_colors = tuple(sorted(colors[j] for j in adj[i]))
        signatures.append((colors[i], neigh_colors))
    
    sorted_unique_sigs = sorted(list(set(signatures)))
    sig_to_color = {sig: idx for idx, sig in enumerate(sorted_unique_sigs)}
    
    return [sig_to_color[sig] for sig in signatures]


def wl_iterate(adj, colors=None):
    """Iterate 1-WL colour refinement until the number of distinct colours stops growing.

    Returns (colors, rounds).
    """
    if colors is None:
        colors = [0] * len(adj)
    else:
        colors = list(colors)
        
    rounds = 0
    current_count = len(set(colors))
    
    while True:
        next_colors = wl_refine(adj, colors)
        next_count = len(set(next_colors))
        if next_count > current_count:
            rounds += 1
            colors = next_colors
            current_count = next_count
        else:
            break
            
    return colors, rounds


def color_histogram(colors):
    """Return the sorted list of class sizes from a list of colours."""
    counts = Counter(colors)
    return sorted(list(counts.values()))


def disjoint_union(adj1, adj2):
    """Construct the disjoint union adjacency list of two graphs side by side,

    with the second graph's nodes offset by len(adj1).
    """
    n1 = len(adj1)
    n2 = len(adj2)
    combined = []
    for i in range(n1):
        combined.append(list(adj1[i]))
    for j in range(n2):
        combined.append([v + n1 for v in adj2[j]])
    return combined


def wl_test(adj1, adj2):
    """Run 1-WL isomorphism test on two graphs via disjoint union.

    Returns True if the two graphs end with identical multisets of colours, and False otherwise.
    """
    if len(adj1) != len(adj2):
        return False
        
    n1 = len(adj1)
    combined = disjoint_union(adj1, adj2)
    final_colors, _ = wl_iterate(combined, colors=None)
    
    colors1 = final_colors[:n1]
    colors2 = final_colors[n1:]
    
    return Counter(colors1) == Counter(colors2)