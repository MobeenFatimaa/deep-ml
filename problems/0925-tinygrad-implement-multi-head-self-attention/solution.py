from tinygrad import Tensor, nn

class MultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        
        # 1. Project to queries, keys, and values, then 2. reshape and transpose into heads
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        # 3-6. Scaled dot-product attention scores QK^T / sqrt(d_head)
        scores = q.matmul(k.transpose(2, 3)) / (self.d_head ** 0.5)
        
        # 7. Add mask if provided
        if mask is not None:
            scores = scores + mask
            
        # 8. Softmax along the last axis
        attn_weights = scores.softmax(axis=-1)
        
        # 9. Apply weights to V and reshape back
        out = attn_weights.matmul(v).transpose(1, 2).reshape(B, T, C)
        
        # 10. Project through out_proj
        return self.out_proj(out)