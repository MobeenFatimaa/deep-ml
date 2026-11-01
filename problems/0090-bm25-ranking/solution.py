from collections import Counter
import numpy as np

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    """
    Calculate BM25 ranking scores for a corpus given a query.
    
    Args:
        corpus: List of tokenized documents (lists of strings)
        query: List of query tokens
        k1: Term frequency saturation parameter
        b: Document length normalization parameter
        
    Returns:
        np.array: BM25 scores per document, rounded to 3 decimals
    """
    if not corpus or not query:
        raise ValueError("Corpus and query must not be empty.")

    N = len(corpus)
    doc_len = np.array([len(doc) for doc in corpus])
    avgdl = np.mean(doc_len) if N > 0 else 1.0
    if avgdl == 0:
        avgdl = 1.0

    # Calculate document frequency (nt) for each unique query term
    nt = {}
    for term in set(query):
        nt[term] = sum(1 for doc in corpus if term in doc)

    # Calculate IDF using the exact formula: ln((N + 1) / (nt + 1))
    idf = {}
    for term in set(query):
        idf[term] = np.log((N + 1) / (nt[term] + 1))

    scores = np.zeros(N)

    # Compute BM25 scores for each document
    for idx, doc in enumerate(corpus):
        doc_counter = Counter(doc)
        doc_score = 0.0
        length_factor = 1.0 - b + b * (doc_len[idx] / avgdl)

        # Iterate over query tokens as given (handling duplicate occurrences)
        for term in query:
            tf = doc_counter[term]
            if tf > 0:
                term_idf = idf[term]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * length_factor
                doc_score += term_idf * (numerator / denominator)

        scores[idx] = doc_score

    return np.round(scores, 3)