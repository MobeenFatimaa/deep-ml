import numpy as np

class SparseDistributedMemory:
    def __init__(self, address_length: int, word_length: int, num_hard_locations: int, activation_radius: int, seed: int = 42):
        """
        Initialize the Sparse Distributed Memory system.
        
        Args:
            address_length: Length of binary address vectors.
            word_length: Length of binary word vectors.
            num_hard_locations: Number of hard storage locations.
            activation_radius: Maximum Hamming distance for activation.
            seed: Random seed for reproducible hard location generation.
        """
        self.address_length = address_length
        self.word_length = word_length
        self.num_hard_locations = num_hard_locations
        self.activation_radius = activation_radius
        
        rng = np.random.RandomState(seed)
        # Generate random binary address vectors (num_hard_locations, address_length)
        self.hard_locations = rng.randint(0, 2, size=(num_hard_locations, address_length))
        
        # Initialize integer counter matrix to zeros (num_hard_locations, word_length)
        self.counters = np.zeros((num_hard_locations, word_length), dtype=int)
    
    def _get_activated_indices(self, address) -> np.ndarray:
        """Helper to find indices of hard locations within the activation radius."""
        addr_arr = np.array(address, dtype=int)
        # Compute Hamming distance: sum of element-wise XOR between address and hard locations
        hamming_distances = np.sum(self.hard_locations != addr_arr, axis=1)
        return np.where(hamming_distances <= self.activation_radius)[0]
    
    def write(self, address, word) -> int:
        """
        Given a binary address and a binary word, update activated hard locations.
        
        Args:
            address: List or array of 0s and 1s representing the address.
            word: List or array of 0s and 1s representing the word.
            
        Returns:
            Number of activated locations as an integer.
        """
        activated_indices = self._get_activated_indices(address)
        
        if len(activated_indices) > 0:
            word_arr = np.array(word, dtype=int)
            # Map 0s to -1s and 1s to +1s for counter updates
            update_vector = np.where(word_arr == 1, 1, -1)
            # Increment/decrement counters for all activated locations
            self.counters[activated_indices] += update_vector
            
        return len(activated_indices)
    
    def read(self, address) -> list:
        """
        Given a binary address, read from all activated locations and threshold.
        
        Args:
            address: List or array of 0s and 1s representing the address.
            
        Returns:
            Recovered binary word as a list of integers (0s and 1s).
        """
        activated_indices = self._get_activated_indices(address)
        
        if len(activated_indices) == 0:
            return [0] * self.word_length
            
        # Sum counter vectors element-wise across all activated locations
        summed_counters = np.sum(self.counters[activated_indices], axis=0)
        
        # Threshold: non-negative sums map to 1, negative sums map to 0
        binary_output = np.where(summed_counters >= 0, 1, 0)
        
        return binary_output.tolist()