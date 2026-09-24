def hamming_distance_kanerva(
    state: list[int], prototypes: list[list[int]], threshold: int
) -> tuple[list[int], list[int]]:
    distances = []
    active_indices = []

    for i, proto in enumerate(prototypes):
        # Calculate Hamming distance by counting differing bits
        dist = sum(s != p for s, p in zip(state, proto))
        distances.append(dist)

        # Check if prototype is active (within threshold)
        if dist <= threshold:
            active_indices.append(i)

    return distances, active_indices