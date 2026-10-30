import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k, v):
        """
        Concatenate the new keys and values onto the cache along the sequence dimension (dim=2)
        and return the full (k, v) so far.
        """
        if self.k is None or self.v is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def __len__(self):
        """
        Number of cached positions (0 when empty).
        """
        if self.k is None:
            return 0
        return self.k.shape[2]

    def reset(self):
        """
        Empty the cache.
        """
        self.k = None
        self.v = None


def attend_with_cache(q, k_new, v_new, cache):
    """
    Append k_new/v_new to the cache, then attend q over everything cached so far.
    """
    # 1. Update cache with new keys and values
    k, v = cache.append(k_new, v_new)
    
    # 2. Get head dimension for scaling
    head_dim = q.shape[-1]
    
    # 3. Scaled dot-product attention (no causal mask needed during inference decode)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
    
    # Softmax over the key/value sequence dimension
    attn_weights = F.softmax(scores, dim=-1)
    
    # Compute attention output
    output = torch.matmul(attn_weights, v)
    
    return output