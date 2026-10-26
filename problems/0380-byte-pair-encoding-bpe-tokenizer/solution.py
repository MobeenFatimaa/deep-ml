def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    merges = []
    
    # Convert corpus keys to tuples of tokens for easier manipulation
    current_corpus = {}
    for word, freq in corpus.items():
        tokens = tuple(word.split())
        current_corpus[tokens] = freq

    for _ in range(num_merges):
        # 1. Count all adjacent token pairs weighted by word frequency
        pair_counts = {}
        for tokens, freq in current_corpus.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        
        if not pair_counts:
            break
            
        # 2. Find the most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        
        # 3. Record the merge operation
        merges.append(best_pair)
        
        # 4. Merge that pair everywhere it appears in the corpus
        new_corpus = {}
        for tokens, freq in current_corpus.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_corpus[tuple(new_tokens)] = freq
            
        current_corpus = new_corpus
        
    return merges