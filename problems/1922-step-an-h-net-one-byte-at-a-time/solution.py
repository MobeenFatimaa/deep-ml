import torch
import torch.nn.functional as F


def init_stream_state(batch, d, enc, main, dec):
    """Build the streaming state dict for a batch."""
    enc_state = enc.init_state(batch)
    if isinstance(enc_state, torch.Tensor):
        device = enc_state.device
        dtype = enc_state.dtype
    else:
        device = enc_state[0].device
        dtype = enc_state[0].dtype

    return {
        'enc': enc_state,
        'main': main.init_state(batch),
        'dec': dec.init_state(batch),
        'last_x': torch.zeros(batch, d, device=device, dtype=dtype),
        'zbar': torch.zeros(batch, d, device=device, dtype=dtype),
        'seen': torch.zeros(batch, dtype=torch.bool, device=device),
    }


def stream_step(x_t, st, enc, w_q, w_k, main, dec, res):
    """Advance one byte. Returns (y_t, st, n_main)."""
    device = x_t.device
    dtype = x_t.dtype

    # 1. Step the encoder to get x^t
    x_hat_t, new_enc_state = enc.step(x_t, st['enc'])

    # 2. Compute p_t
    q = w_q(x_hat_t)
    k_prev = w_k(st['last_x'])

    cos_sim = F.cosine_similarity(q, k_prev, dim=-1)
    p = (1.0 - cos_sim) / 2.0

    # Elements that have not seen a byte yet have p = 1
    p = torch.where(st['seen'], p, torch.tensor(1.0, device=device, dtype=dtype))
    p = torch.clamp(p, 0.0, 1.0)

    b = p >= 0.5
    main_indices = b.nonzero(as_tuple=True)[0]
    n_main = len(main_indices)

    zbar = st['zbar'].clone()
    main_state = st['main']

    # 3. On the sub-batch with b true only, step the main network
    if n_main > 0:
        sub_x_hat = x_hat_t[main_indices]
        if isinstance(main_state, torch.Tensor):
            sub_main_state = main_state[main_indices]
        elif isinstance(main_state, (list, tuple)):
            sub_main_state = type(main_state)([s[main_indices] for s in main_state])
        else:
            sub_main_state = main_state

        sub_z_hat, sub_new_main_state = main.step(sub_x_hat, sub_main_state)

        # 4. Update zbar for the sub-batch
        sub_p = p[main_indices].unsqueeze(-1)
        sub_zbar_old = zbar[main_indices]
        sub_zbar_new = sub_p * sub_z_hat + (1.0 - sub_p) * sub_zbar_old
        zbar[main_indices] = sub_zbar_new

        # Write back main state
        if isinstance(main_state, torch.Tensor):
            new_main_state = main_state.clone()
            new_main_state[main_indices] = sub_new_main_state
        elif isinstance(main_state, (list, tuple)):
            new_main_state = type(main_state)([
                s.clone() if isinstance(s, torch.Tensor) else s for s in main_state
            ])
            for idx, s in enumerate(sub_new_main_state):
                new_main_state[idx][main_indices] = s
        else:
            new_main_state = main_state
    else:
        new_main_state = main_state

    # 5. Decoder input is zbar plus res applied to x^t
    dec_input = zbar + res(x_hat_t)

    # Step the decoder
    y_t, new_dec_state = dec.step(dec_input, st['dec'])

    # Update streaming state
    new_st = {
        'enc': new_enc_state,
        'main': new_main_state,
        'dec': new_dec_state,
        'last_x': x_hat_t,
        'zbar': zbar,
        'seen': torch.ones_like(st['seen']),
    }

    return y_t, new_st, n_main