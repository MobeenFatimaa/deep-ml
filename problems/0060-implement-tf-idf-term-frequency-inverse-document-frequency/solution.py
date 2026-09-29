import math


def compute_tf_idf(corpus, query):
    """Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each
        document
    """
    if not corpus:
        return []

    N = len(corpus)

    # Calculate Inverse Document Frequency (IDF) with standard smoothing:
    # idf(t) = ln((N + 1) / (df(t) + 1)) + 1
    idfs = []
    for term in query:
        # Count how many documents contain the term
        df = sum(1 for doc in corpus if term in doc)
        idf = math.log((N + 1) / (df + 1)) + 1.0
        idfs.append(idf)

    scores = []
    for doc in corpus:
        doc_len = len(doc)
        doc_scores = []

        for i, term in enumerate(query):
            if doc_len == 0:
                tf = 0.0
            else:
                # Term Frequency: count of term / total terms in document
                tf = doc.count(term) / doc_len

            # TF-IDF calculation
            tf_idf = tf * idfs[i]
            doc_scores.append(round(tf_idf, 5))

        scores.append(doc_scores)

    return scores