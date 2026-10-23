import numpy as np


def sae_forward(X, W_enc, b_enc, W_dec, b_dec):
    pre = X @ W_enc + b_enc
    Z = np.maximum(pre, 0)
    Xhat = Z @ W_dec + b_dec
    return pre, Z, Xhat


def sae_loss_and_grads(X, W_enc, b_enc, W_dec, b_dec, lam):
    """(loss, {"W_enc", "b_enc", "W_dec", "b_dec"}) with exact gradients."""
    B, d = X.shape
    
    pre, Z, Xhat = sae_forward(X, W_enc, b_enc, W_dec, b_dec)
    
    diff = Xhat - X
    recon_loss = np.sum(diff ** 2) / B
    l1_loss = lam * np.sum(np.abs(Z)) / B
    loss = recon_loss + l1_loss
    
    d_Xhat = (2.0 / B) * diff
    
    d_W_dec = Z.T @ d_Xhat
    d_b_dec = np.sum(d_Xhat, axis=0)
    
    d_Z = d_Xhat @ W_dec.T + (lam / B)
    d_pre = d_Z * (pre > 0)
    
    d_W_enc = X.T @ d_pre
    d_b_enc = np.sum(d_pre, axis=0)
    
    grads = {
        "W_enc": d_W_enc,
        "b_enc": d_b_enc,
        "W_dec": d_W_dec,
        "b_dec": d_b_dec
    }
    
    return loss, grads


def sae_step(params, X, lam, lr):
    """One gradient step, then unit-normalise decoder rows; returns a new dict."""
    _, grads = sae_loss_and_grads(
        X, params["W_enc"], params["b_enc"], params["W_dec"], params["b_dec"], lam
    )
    new_params = {
        "W_enc": params["W_enc"] - lr * grads["W_enc"],
        "b_enc": params["b_enc"] - lr * grads["b_enc"],
        "W_dec": params["W_dec"] - lr * grads["W_dec"],
        "b_dec": params["b_dec"] - lr * grads["b_dec"],
    }
    
    row_norms = np.linalg.norm(new_params["W_dec"], axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    new_params["W_dec"] = new_params["W_dec"] / row_norms
    
    return new_params


def sae_diagnostics(Z, X, Xhat, dense_thresh=0.5, decimals=4):
    B = X.shape[0]
    n = Z.shape[1]
    
    active_mask = (Z > 0)
    l0 = float(np.mean(np.sum(active_mask, axis=1)))
    
    residual_ss = np.sum((X - Xhat) ** 2)
    total_ss = np.sum((X - np.mean(X, axis=0, keepdims=True)) ** 2)
    explained_variance = float(1.0 - (residual_ss / total_ss) if total_ss > 0 else 0.0)
    
    latent_active_counts = np.sum(active_mask, axis=0)
    dead_count = np.sum(latent_active_counts == 0)
    dead_frac = float(dead_count / n)
    
    latent_active_fracs = latent_active_counts / B
    dense_count = np.sum(latent_active_fracs > dense_thresh)
    dense_frac = float(dense_count / n)
    
    return {
        "l0": round(l0, decimals),
        "explained_variance": round(explained_variance, decimals),
        "dead_frac": round(dead_frac, decimals),
        "dense_frac": round(dense_frac, decimals)
    }