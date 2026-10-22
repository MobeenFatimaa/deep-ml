import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, d_model = x.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape into heads: (B, T, num_heads, d_head) -> (B, num_heads, T, d_head)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        # 3. Scaled dot-product attention scores
        scale = 1.0 / (self.d_head ** 0.5)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        # 4. Apply mask if provided
        if mask is not None:
            scores = scores + mask
            
        # 5. Softmax along the last axis
        attn_weights = F.softmax(scores, dim=-1)
        
        # 6. Apply weights to V
        context = torch.matmul(attn_weights, v)
        
        # 7. Reshape back to (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, d_model)
        
        # 8. Output projection
        out = self.out_proj(context)
        
        return out