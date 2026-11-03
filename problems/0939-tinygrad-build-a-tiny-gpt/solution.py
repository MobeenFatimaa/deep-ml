from tinygrad import Tensor, nn

class TinyTransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Pre-LN layers
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp1 = nn.Linear(d_model, d_ff)
        self.mlp2 = nn.Linear(d_ff, d_model)

    def __call__(self, x):
        # 1. Causal Self-Attention with Pre-LN and Residual Connection
        h = self.ln1(x)
        B, T, _ = h.shape
        
        # Project and reshape to (B, num_heads, T, head_dim)
        q = self.q_proj(h).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot product attention with causal mask
        attn_out = q.scaled_dot_product_attention(k, v, is_causal=True)
        
        # Reshape back to (B, T, d_model) and apply output projection
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, -1)
        x = x + self.out_proj(attn_out)
        
        # 2. MLP with Pre-LN and Residual Connection
        h2 = self.ln2(x)
        mlp_out = self.mlp2(self.mlp1(h2).gelu())
        x = x + mlp_out
        
        return x

class TinyGPT:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Plain Python list for transformer blocks (automatically tracked by tinygrad)
        self.blocks = [TinyTransformerBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)]
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        
        # 1. Embed tokens and add positional embeddings
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(Tensor.arange(T))
        x = tok_emb + pos_emb
        
        # 2. Pass through each transformer block
        for block in self.blocks:
            x = block(x)
            
        # 3. Final LayerNorm and output projection head
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits