import torch
import torch.nn as nn
import torch.nn.functional as F


class DCStage(nn.Module):
    def __init__(self, d_model, main):
        """W_q and W_k start as the identity, residual starts at zero; all three are bias-free."""
        super().__init__()
        self.main = main
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.residual = nn.Linear(d_model, d_model, bias=False)
        
        nn.init.eye_(self.W_q.weight)
        nn.init.eye_(self.W_k.weight)
        nn.init.zeros_(self.residual.weight)

    def forward(self, x_hat):
        """x_hat (B, L, D). Returns z (B, L, D), p float (B, L), b bool (B, L)."""
        B, L, D = x_hat.shape
        
        # 1. Route
        q = self.W_q(x_hat)
        k = self.W_k(x_hat)
        
        # k_prev with padding for t=0 (prepending k[:, :1] avoids division by zero in cosine similarity)
        k_prev = torch.cat([k[:, :1], k[:, :-1]], dim=1)
        
        cos_sim = F.cosine_similarity(q, k_prev, dim=-1)
        p = (1.0 - cos_sim) / 2.0
        p[:, 0] = 1.0
        p = torch.clamp(p, 0.0, 1.0)
        
        b = p >= 0.5
        
        # 2. Compact
        b_int = b.long()
        counts = b_int.sum(dim=1)
        M = int(counts.max().item())
        if M == 0:
            M = 1
            
        compacted_x = x_hat.new_zeros(B, M, D)
        compacted_p = p.new_zeros(B, M)
        
        for i in range(B):
            valid_idx = b[i].nonzero(as_tuple=True)[0]
            k_len = len(valid_idx)
            if k_len > 0:
                compacted_x[i, :k_len] = x_hat[i, valid_idx]
                compacted_p[i, :k_len] = p[i, valid_idx]
                
        # 3. Main network
        z_compacted = self.main(compacted_x)
        
        # 4. Upsample
        c_sum = torch.cumsum(b_int, dim=1)
        chunk_indices = c_sum - 1
        # Clamp indices to valid range [0, M-1] to prevent out-of-bounds gathering
        chunk_indices = torch.clamp(chunk_indices, 0, M - 1)
        
        gather_idx = chunk_indices.unsqueeze(-1).expand(B, L, D)
        z_hat = torch.gather(z_compacted, dim=1, index=gather_idx)
        
        # Smooth
        z_smoothed = []
        z_prev = torch.zeros(B, D, device=x_hat.device, dtype=x_hat.dtype)
        for t in range(L):
            pt = p[:, t].unsqueeze(-1)
            z_t = z_hat[:, t, :]
            z_curr = pt * z_t + (1.0 - pt) * z_prev
            z_smoothed.append(z_curr)
            z_prev = z_curr
        z = torch.stack(z_smoothed, dim=1)
        
        # 5. Confidence
        c = torch.where(b, p, 1.0 - p)
        factor = 1.0 + (c - c.detach())
        z = z * factor.unsqueeze(-1)
        
        # 6. Residual
        z = z + self.residual(x_hat)
        
        return z, p, b