class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

def segment_stats(s, words):
    """Return (number of splits of s into words, fewest pieces or -1)."""
    # 1. Deduplicate words into a set (vocabulary)
    unique_words = set(words)
    
    # 2. Build the Trie
    root = TrieNode()
    for word in unique_words:
        curr = root
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.is_word = True
        
    n = len(s)
    
    # DP arrays for ways and fewest pieces from index i to end
    ways = [0] * (n + 1)
    fewest = [float('inf')] * (n + 1)
    
    # Base case for empty suffix
    ways[n] = 1
    fewest[n] = 0
    
    # 3. DP from right to left
    for i in range(n - 1, -1, -1):
        curr = root
        total_ways = 0
        min_pieces = float('inf')
        
        # Walk the Trie with characters starting from index i
        for j in range(i, n):
            char = s[j]
            if char not in curr.children:
                break
            curr = curr.children[char]
            
            # If a valid dictionary word ends here
            if curr.is_word:
                next_ways = ways[j + 1]
                if next_ways > 0:
                    total_ways += next_ways
                    min_pieces = min(min_pieces, 1 + fewest[j + 1])
                    
        ways[i] = total_ways
        fewest[i] = min_pieces
        
    # 4. Format final output
    total_ways = ways[0]
    fewest_pieces = fewest[0] if fewest[0] != float('inf') else -1
    
    return (total_ways, fewest_pieces)