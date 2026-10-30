import numpy as np

def nextlat_loss(H, E_tok, targets, W_out, b_out, Wh, Wx, d,
                 lam_h, lam_kl, beta=1.0):
    """
    The full NextLat objective.

    Args:
        H:      (T, D) hidden states
        E_tok:   (T, D) token embeddings; E_tok[t] embeds the token at position t
        targets: (T,) int array; targets[t] is the token following position t
        W_out:   (D, V) output head weights
        b_out:   (V,) output head bias
        Wh, Wx:  (D, D) each; transition tanh(h @ Wh + e @ Wx)
        d:       rollout horizon
        lam_h:   weight on the next-hidden term
        lam_kl:  weight on the KL term
        beta:    Smooth L1 beta

    Returns:
        dict with float values under 'next_token', 'next_h', 'kl', 'total'
    """
    H = np.asarray(H, dtype=float)
    E_tok = np.asarray(E_tok, dtype=float)
    targets = np.asarray(targets, dtype=int)
    W_out = np.asarray(W_out, dtype=float)
    b_out = np.asarray(b_out, dtype=float)
    Wh = np.asarray(Wh, dtype=float)
    Wx = np.asarray(Wx, dtype=float)

    T, D = H.shape

    # 1. Next-Token Loss (Cross-Entropy)
    # Logits = H @ W_out + b_out -> shape (T, V)
    logits = H @ W_out + b_out
    # Stable softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Cross-entropy over all T positions
    # CE = -log(probs[t, targets[t]])
    chosen_probs = probs[np.arange(T), targets]
    next_token_loss = float(-np.mean(np.log(np.clip(chosen_probs, 1e-12, 1.0))))

    # 2. Rollout and Next-H / KL Losses
    next_h_losses = []
    kl_losses = []

    # Valid start indices t where t + d <= T - 1
    valid_starts = [t for t in range(T) if t + d <= T - 1]

    if len(valid_starts) == 0:
        next_h_loss = 0.0
        kl_loss = 0.0
    else:
        for t in valid_starts:
            step_next_h = []
            step_kl = []
            
            h_hat = H[t]
            for i in range(1, d + 1):
                # Transition: tau(h, e) = tanh(h @ Wh + e @ Wx)
                h_hat = np.tanh(h_hat @ Wh + E_tok[t + i] @ Wx)
                h_target = H[t + i]

                # Smooth L1 loss (PyTorch convention)
                # abs_diff = |h_hat - h_target|
                # if abs_diff < beta: 0.5 * abs_diff^2 / beta
                # else: abs_diff - 0.5 * beta
                diff = np.abs(h_hat - h_target)
                smooth_l1 = np.where(
                    diff < beta,
                    0.5 * (diff ** 2) / beta,
                    diff - 0.5 * beta
                )
                step_next_h.append(np.mean(smooth_l1))

                # KL divergence: KL(softmax(logits(H[t+i])) || softmax(logits(h_hat)))
                # logits for actual H[t+i]
                logits_true = H[t + i] @ W_out + b_out
                # logits for rollout h_hat
                logits_pred = h_hat @ W_out + b_out

                # Stable softmax probabilities
                p_true = np.exp(logits_true - np.max(logits_true))
                p_true /= np.sum(p_true)

                p_pred = np.exp(logits_pred - np.max(logits_pred))
                p_pred /= np.sum(p_pred)

                # KL divergence = sum(p_true * (log(p_true) - log(p_pred)))
                kl_val = np.sum(p_true * (np.log(np.clip(p_true, 1e-12, 1.0)) - np.log(np.clip(p_pred, 1e-12, 1.0))))
                step_kl.append(kl_val)

            next_h_losses.append(np.mean(step_next_h))
            kl_losses.append(np.mean(step_kl))

        next_h_loss = float(np.mean(next_h_losses))
        kl_loss = float(np.mean(kl_losses))

    total_loss = next_token_loss + lam_h * next_h_loss + lam_kl * kl_loss

    return {
        'next_token': next_token_loss,
        'next_h': next_h_loss,
        'kl': kl_loss,
        'total': total_loss
    }