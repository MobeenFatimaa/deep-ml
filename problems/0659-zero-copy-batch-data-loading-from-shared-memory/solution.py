import numpy as np
import math

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        """Store data in a flat contiguous buffer simulating shared memory."""
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D NumPy array.")
        self.n_samples, self.n_features = data.shape
        self.batch_size = batch_size
        # Ensure the buffer is contiguous and flattened to 1D
        self.buffer = np.ascontiguousarray(data).ravel()

    def num_batches(self) -> int:
        """Return total number of batches."""
        if self.batch_size <= 0:
            return 0
        return math.ceil(self.n_samples / self.batch_size)

    def get_batch(self, batch_idx: int) -> np.ndarray:
        """Return batch as a zero-copy view into the buffer."""
        total_batches = self.num_batches()
        if not (0 <= batch_idx < total_batches):
            raise IndexError(f"Batch index {batch_idx} out of range for {total_batches} batches.")
        
        start_row = batch_idx * self.batch_size
        end_row = min((batch_idx + 1) * self.batch_size, self.n_samples)
        batch_rows = end_row - start_row
        
        # Calculate start and end indices in the flat 1D buffer
        start_idx = start_row * self.n_features
        end_idx = end_row * self.n_features
        
        # Slice the 1D buffer and reshape to 2D. 
        # Slicing a 1D contiguous array and reshaping it creates a memory view.
        return self.buffer[start_idx:end_idx].reshape(batch_rows, self.n_features)

    def is_zero_copy(self, batch_idx: int) -> bool:
        """Check whether the batch shares memory with the buffer."""
        batch = self.get_batch(batch_idx)
        return np.shares_memory(batch, self.buffer)

    def get_batch_means(self) -> list:
        """Return list of per-batch mean values, each rounded to 4 decimals."""
        means = []
        for i in range(self.num_batches()):
            batch = self.get_batch(i)
            batch_mean = round(float(np.mean(batch)), 4)
            means.append(batch_mean)
        return means

    def write_to_buffer(self, row: int, col: int, value: float) -> None:
        """Write a value directly into the flat buffer at (row, col)."""
        if not (0 <= row < self.n_samples and 0 <= col < self.n_features):
            raise IndexError(f"Position ({row}, {col}) is out of bounds for shape ({self.n_samples}, {self.n_features}).")
        
        flat_idx = row * self.n_features + col
        self.buffer[flat_idx] = value