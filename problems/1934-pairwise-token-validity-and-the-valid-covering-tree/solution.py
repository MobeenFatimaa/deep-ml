import re

def apply_merge(seq, left, right):
    new_seq = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == left and seq[i+1] == right:
            new_seq.append(left + right)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return new_seq


def bpe_encode(data, merges):
    """
    Start from single bytes and apply the merges in order, 
    each one to the whole sequence, left to right without overlap.
    """
    if isinstance(data, bytes):
        seq = [bytes([b]) for b in data]
    else:
        seq = list(data)
    for left, right in merges:
        seq = apply_merge(seq, left, right)
    return seq


def is_valid_pair(left, right, merges):
    """
    Return True if the two-token sequence [left, right] is valid.
    """
    encoded = bpe_encode(left + right, merges)
    return encoded == [left, right]


def is_valid(tokens, merges):
    """
    Return whether the whole list is valid using Proposition 3.4 
    (every adjacent pair must be valid).
    """
    if len(tokens) <= 1:
        return True
    for i in range(len(tokens) - 1):
        if not is_valid_pair(tokens[i], tokens[i+1], merges):
            return False
    return True


def valid_covering_tree(prefix, merges):
    """
    Return, sorted, the list of all token sequences (as tuples of bytes) 
    that form the leaves of the Valid Covering Tree for the given prefix.
    """
    if not prefix:
        raise ValueError("prefix cannot be empty")
    
    # Collect vocabulary: 256 single bytes + all merge results
    vocab = set(bytes([b]) for b in range(256))
    for left, right in merges:
        vocab.add(left + right)
        
    leaves = set()
    
    def dfs(current_tokens):
        acc = b"".join(current_tokens)
        
        # If the accumulated bytes cover or exceed the prefix length
        if len(acc) >= len(prefix):
            if acc.startswith(prefix):
                # Ensure t1...tn-1 are strictly shorter than prefix
                if len(current_tokens) == 1 or len(b"".join(current_tokens[:-1])) < len(prefix):
                    leaves.add(tuple(current_tokens))
            return
        
        # Prune if acc is not a prefix of prefix
        if not prefix.startswith(acc):
            return
            
        for t in vocab:
            if len(current_tokens) > 0:
                if not is_valid_pair(current_tokens[-1], t, merges):
                    continue
            dfs(current_tokens + [t])
            
    dfs([])
    return sorted(list(leaves))


def trunk(leaves):
    """
    Return the longest list of tokens that every leaf starts with 
    ([] for an empty list of leaves).
    """
    if not leaves:
        return []
    
    prefix_len = 0
    first_leaf = leaves[0]
    while True:
        if prefix_len >= len(first_leaf):
            break
        token = first_leaf[prefix_len]
        match = True
        for leaf in leaves:
            if prefix_len >= len(leaf) or leaf[prefix_len] != token:
                match = False
                break
        if match:
            prefix_len += 1
        else:
            break
            
    return list(first_leaf[:prefix_len])