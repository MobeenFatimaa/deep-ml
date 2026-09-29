import hashlib
import numpy as np


def minhash_near_duplicates(
    documents: list[str],
    num_hashes: int,
    threshold: float,
    shingle_size: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Detect near-duplicate document pairs using MinHash.

    Returns a list of (i, j) index pairs (i < j) whose estimated Jaccard
    similarity meets or exceeds the given threshold.
    """
    p = (1 << 61) - 1

    # Generate deterministic coefficients for hash functions
    rng = np.random.default_rng(seed)
    a = rng.integers(1, p, size=num_hashes, dtype=np.int64)
    b = rng.integers(0, p, size=num_hashes, dtype=np.int64)

    signatures = []

    for doc in documents:
        tokens = doc.lower().split()

        # Generate shingles
        if not tokens:
            shingles = set()
        elif len(tokens) < shingle_size:
            shingles = {" ".join(tokens)}
        else:
            shingles = {
                " ".join(tokens[i : i + shingle_size])
                for i in range(len(tokens) - shingle_size + 1)
            }

        # If document is empty or yields no shingles
        if not shingles:
            signatures.append(np.full(num_hashes, p, dtype=np.int64))
            continue

        # Compute base hash for each shingle
        base_hashes = []
        for s in shingles:
            md5_val = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
            base_hashes.append(md5_val % p)

        base_hashes = np.array(base_hashes, dtype=np.int64)  # Shape: (S,)

        # Compute hash values for all shingles across all hash functions
        # Vectorized calculation: (num_hashes, 1) * (1, S) + (num_hashes, 1) mod p
        hash_matrix = (
            a[:, None] * base_hashes[None, :] + b[:, None]
        ) % p  # Shape: (num_hashes, S)

        # MinHash signature is the minimum along the shingle axis
        sig = np.min(hash_matrix, axis=1)
        signatures.append(sig)

    # Estimate Jaccard similarity for all pairs (i, j) with i < j
    near_duplicates = []
    n_docs = len(documents)

    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            sim = np.mean(signatures[i] == signatures[j])
            if sim >= threshold:
                near_duplicates.append((i, j))

    return near_duplicates


# Example test run
if __name__ == "__main__":
    documents = [
        "the cat sat on the mat",
        "the cat sat on the mat",
        "a dog ran across the field",
    ]
    num_hashes = 32
    threshold = 0.7
    shingle_size = 2
    seed = 0

    result = minhash_near_duplicates(
        documents, num_hashes, threshold, shingle_size, seed
    )
    print(result)  # Output: [(0, 1)]