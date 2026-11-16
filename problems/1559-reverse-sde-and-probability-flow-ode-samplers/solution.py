import numpy as np


def reverse_sde_step(x, t, dt, beta_fn, score_fn, rng):
    """
    One Euler-Maruyama step of the VP reverse SDE (Song et al. 2021, eq. 6) from t to t - dt.
    The noise z is a single rng.standard_normal(x.shape) draw. Returns the new (N, d) array.
    """
    beta = beta_fn(t)
    score = score_fn(x, t)
    
    # Reverse SDE drift coefficient moving backwards in time (dt > 0)
    drift = 0.5 * beta * x + beta * score
    
    # Diffusion coefficient term for Euler-Maruyama
    noise_scale = np.sqrt(beta * dt)
    z = rng.standard_normal(x.shape)
    
    return x + drift * dt + noise_scale * z


def pf_ode_step(x, t, dt, beta_fn, score_fn):
    """
    One Euler step of the VP probability-flow ODE (eq. 13) from t to t - dt.
    Returns the new (N, d) array.
    """
    beta = beta_fn(t)
    score = score_fn(x, t)
    
    # Probability-flow ODE drift coefficient moving backwards in time (dt > 0)
    drift = 0.5 * beta * x + 0.5 * beta * score
    
    return x + drift * dt


def sample_vp(x_T, ts, beta_fn, score_fn, mode="ode", rng=None):
    """
    Run the chosen sampler over a decreasing time grid ts (step i: t = ts[i], dt = ts[i] - ts[i+1]).
    mode is "ode" or "sde" (rng required for "sde"). Returns the final (N, d) array.
    """
    x = np.copy(x_T)
    
    for i in range(len(ts) - 1):
        t = ts[i]
        dt = ts[i] - ts[i + 1]
        
        if mode == "ode":
            x = pf_ode_step(x, t, dt, beta_fn, score_fn)
        elif mode == "sde":
            if rng is None:
                raise ValueError("An RNG must be provided when mode is 'sde'.")
            x = reverse_sde_step(x, t, dt, beta_fn, score_fn, rng)
        else:
            raise ValueError(f"Unsupported sampling mode: {mode}")
            
    return x