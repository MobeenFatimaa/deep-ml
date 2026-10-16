import numpy as np

def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack even-length nibble array (0..15) into uint8 bytes: lo | (hi<<4)."""
    q_arr = np.asarray(q, dtype=np.uint8)
    # Reshape into pairs: (N // 2, 2) where column 0 is low nibble, column 1 is high nibble
    q_pairs = q_arr.reshape(-1, 2)
    # Pack: lo | (hi << 4)
    packed = q_pairs[:, 0] | (q_pairs[:, 1] << 4)
    return packed.astype(np.uint8)

def unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack bytes to length-n int nibbles."""
    packed_arr = np.asarray(packed, dtype=np.uint8)
    # Extract low nibbles
    lo = packed_arr & 0x0F
    # Extract high nibbles
    hi = (packed_arr >> 4) & 0x0F
    
    # Interleave low and high nibbles
    unpacked = np.empty((packed_arr.size * 2,), dtype=np.int32)
    unpacked[0::2] = lo
    unpacked[1::2] = hi
    
    # Trim to original length n
    return unpacked[:n]

def w4a16_matmul(x, packed_w, scales, in_features: int):
    """
    x: (batch, in_features) float
    packed_w: (out_features, in_features//2) uint8
    scales: (out_features,)
    Dequant: W = scale * (q - 8), y = x @ W.T
    """
    x_arr = np.asarray(x, dtype=np.float32)
    packed_w_arr = np.asarray(packed_w, dtype=np.uint8)
    scales_arr = np.asarray(scales, dtype=np.float32)
    
    out_features = packed_w_arr.shape[0]
    
    # Unpack weights for each row
    unpacked_w = np.empty((out_features, in_features), dtype=np.float32)
    for i in range(out_features):
        unpacked_w[i] = unpack_int4(packed_w_arr[i], in_features)
        
    # Dequantize weights: W = scale * (q - 8)
    # scales_arr is (out_features,), broadcast across in_features
    W = scales_arr[:, np.newaxis] * (unpacked_w - 8.0)
    
    # Compute matmul: y = x @ W.T
    y = np.matmul(x_arr, W.T)
    return y