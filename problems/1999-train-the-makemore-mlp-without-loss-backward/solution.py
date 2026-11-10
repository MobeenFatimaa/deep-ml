import torch
import torch.nn.functional as F

# ---- provided: parameters and the forward pass from the video (do not change) ----
def init_params(vocab_size=27, block_size=3, n_embd=10, n_hidden=64, seed=42, dtype=torch.float64):
    """Small random values everywhere, including the biases, so a wrong gradient can't hide behind zeros."""
    g = torch.Generator().manual_seed(seed)
    fan_in = n_embd * block_size
    return {
        "C": torch.randn((vocab_size, n_embd), generator=g, dtype=dtype),
        "W1": torch.randn((fan_in, n_hidden), generator=g, dtype=dtype) * (5 / 3) / fan_in ** 0.5,
        "b1": torch.randn(n_hidden, generator=g, dtype=dtype) * 0.1,
        "W2": torch.randn((n_hidden, vocab_size), generator=g, dtype=dtype) * 0.1,
        "b2": torch.randn(vocab_size, generator=g, dtype=dtype) * 0.1,
        "bngain": torch.randn((1, n_hidden), generator=g, dtype=dtype) * 0.1 + 1.0,
        "bnbias": torch.randn((1, n_hidden), generator=g, dtype=dtype) * 0.1,
    }


def forward(params, Xb, Yb):
    """The makemore MLP with batch norm. Returns the loss and every intermediate you may need."""
    C, W1, b1, W2, b2 = params["C"], params["W1"], params["b1"], params["W2"], params["b2"]
    bngain, bnbias = params["bngain"], params["bnbias"]
    n = Xb.shape[0]
    emb = C[Xb]                                           # (n, block_size, n_embd)
    embcat = emb.view(n, -1)                              # (n, block_size * n_embd)
    hprebn = embcat @ W1 + b1                             # (n, n_hidden)
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = ((hprebn - bnmean) ** 2).sum(0, keepdim=True) / (n - 1)   # Bessel's correction
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)                               # (n, n_hidden)
    logits = h @ W2 + b2                                  # (n, vocab_size)
    loss = F.cross_entropy(logits, Yb)
    cache = {"emb": emb, "embcat": embcat, "hprebn": hprebn, "bnvar_inv": bnvar_inv, "bnraw": bnraw,
             "h": h, "logits": logits}
    return loss, cache
# ---- end provided ----


def backward(params, cache, Xb, Yb):
    """Gradients of forward()'s loss w.r.t. every parameter, written by hand: no autograd.

    Return a dict with the same keys and shapes as params.
    """
    C, W1, b1, W2, b2 = params["C"], params["W1"], params["b1"], params["W2"], params["b2"]
    bngain, bnbias = params["bngain"], params["bnbias"]
    emb = cache["emb"]
    embcat = cache["embcat"]
    hprebn = cache["hprebn"]
    bnvar_inv = cache["bnvar_inv"]
    bnraw = cache["bnraw"]
    h = cache["h"]
    logits = cache["logits"]
    
    n = Xb.shape[0]
    
    # 1. Cross Entropy Loss backprop w.r.t. logits
    # F.cross_entropy combines log_softmax and NLLLoss.
    # Derivative of cross_entropy w.r.t. logits is (softmax(logits) - one_hot(Yb)) / n
    probs = F.softmax(logits, dim=1)
    dlogits = probs.clone()
    dlogits[torch.arange(n), Yb] -= 1.0
    dlogits /= n
    
    # 2. Linear layer 2: logits = h @ W2 + b2
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(dim=0)
    dh = dlogits @ W2.T
    
    # 3. Tanh activation: h = torch.tanh(hpreact)
    dhpreact = dh * (1.0 - h ** 2)
    
    # 4. Batch norm scaling and shifting: hpreact = bngain * bnraw + bnbias
    dbngain = (dhpreact * bnraw).sum(dim=0, keepdim=True)
    dbnbias = dhpreact.sum(dim=0, keepdim=True)
    dbnraw = dhpreact * bngain
    
    # 5. Batch normalization backward (with Bessel's correction: denominator is n - 1)
    # bnraw = (hprebn - bnmean) * bnvar_inv
    # bnvar = sum((hprebn - bnmean)**2) / (n - 1)
    # bnvar_inv = (bnvar + 1e-5)**-0.5
    bnmean = hprebn.mean(dim=0, keepdim=True)
    
    dbnvar_inv = (dbnraw * (hprebn - bnmean)).sum(dim=0, keepdim=True)
    dbnvar = dbnvar_inv * (-0.5) * (bnvar_inv ** 3)
    
    # hprebn gradient components
    dhprebn1 = dbnraw * bnvar_inv
    dhprebn2 = dbnvar * 2.0 * (hprebn - bnmean) / (n - 1)
    dhprebn3 = (- (dhprebn1 + dhprebn2).sum(dim=0, keepdim=True)) / n
    
    dhprebn = dhprebn1 + dhprebn2 + dhprebn3
    
    # 6. Linear layer 1: hprebn = embcat @ W1 + b1
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(dim=0)
    dembcat = dhprebn @ W1.T
    
    # 7. Embedding reshape and lookup: embcat = emb.view(n, -1), emb = C[Xb]
    demb = dembcat.view(emb.shape)
    
    # C gradient: scatter_add indices from Xb into C shape
    dC = torch.zeros_like(C)
    dC.index_add_(0, Xb.view(-1), demb.view(-1, C.shape[1]))
    
    return {
        "C": dC,
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "bngain": dbngain,
        "bnbias": dbnbias
    }


def train(params, X, Y, steps, batch_size=32, lr=0.1, seed=0):
    """Minibatch SGD using your backward(). Update params in place; return each step's loss (before its update)."""
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    losses = []
    
    for _ in range(steps):
        # Draw batch_size row indices uniformly from 0 .. N-1
        ix = torch.randint(0, N, (batch_size,), generator=g)
        Xb = X[ix]
        Yb = Y[ix]
        
        # Forward pass
        loss, cache = forward(params, Xb, Yb)
        losses.append(loss.item())
        
        # Backward pass
        grads = backward(params, cache, Xb, Yb)
        
        # Update parameters in place using SGD: param -= lr * grad
        with torch.no_grad():
            for k in params:
                params[k] -= lr * grads[k]
                
    return losses