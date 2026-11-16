import numpy as np


def karras_grid(sigma_min, sigma_max, N, rho=7.0):
    """
    Karras et al. (2022) eq. 5: N noise levels from sigma_max down to sigma_min with exponent rho,
    followed by a final 0. Returns an array of length N + 1.
    """
    if N == 1:
        return np.array([sigma_max, 0.0])
    i = np.arange(N)
    t = (sigma_max ** (1.0 / rho) + i / (N - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    return np.append(t, 0.0)


def heun_step(f, x, t, t_next):
    """
    One step of Algorithm 1. Returns (x_next, nfe): Euler only (nfe = 1) when t_next == 0,
    otherwise the Heun correction with the derivative at the Euler point (nfe = 2).
    """
    dt = t_next - t
    d = f(x, t)
    x_euler = x + dt * d
    
    # Final step to t = 0 is Euler-only per Karras et al. Algorithm 1
    if t_next == 0.0 or t_next == 0:
        return x_euler, 1
        
    d_prime = f(x_euler, t_next)
    x_next = x + dt * 0.5 * (d + d_prime)
    return x_next, 2


def sample_ode(f, x_T, t_grid, method="heun"):
    """
    Integrate dx/dt = f(x, t) along t_grid with "euler" or "heun".
    Returns {"x": final array, "nfe": number of calls to f}. Raise ValueError for other methods.
    """
    x = np.copy(x_T)
    nfe = 0
    
    for i in range(len(t_grid) - 1):
        t = t_grid[i]
        t_next = t_grid[i + 1]
        dt = t_next - t
        
        if method == "euler":
            d = f(x, t)
            x = x + dt * d
            nfe += 1
        elif method == "heun":
            x, step_nfe = heun_step(f, x, t, t_next)
            nfe += step_nfe
        else:
            raise ValueError(f"Unsupported ODE sampling method: {method}")
            
    return {"x": x, "nfe": nfe}


def compare_at_budget(f, x_T, x_exact, sigma_min, sigma_max, nfe, rho=7.0):
    """
    Euler on karras_grid(N = nfe) and Heun on karras_grid(N = (nfe + 1) // 2), both costing nfe.
    Returns {"euler": {"x", "nfe", "error"}, "heun": {"x", "nfe", "error"}} with error the RMSE vs x_exact.
    """
    # Euler run (cost = nfe intervals = nfe evaluations)
    grid_euler = karras_grid(sigma_min, sigma_max, N=nfe, rho=rho)
    res_euler = sample_ode(f, x_T, grid_euler, method="euler")
    err_euler = float(np.sqrt(np.mean((res_euler["x"] - x_exact) ** 2)))
    
    # Heun run (cost = 2(N - 1) + 1 = nfe evaluations when N = (nfe + 1) // 2)
    N_heun = (nfe + 1) // 2
    grid_heun = karras_grid(sigma_min, sigma_max, N=N_heun, rho=rho)
    res_heun = sample_ode(f, x_T, grid_heun, method="heun")
    err_heun = float(np.sqrt(np.mean((res_heun["x"] - x_exact) ** 2)))
    
    return {
        "euler": {"x": res_euler["x"], "nfe": res_euler["nfe"], "error": err_euler},
        "heun": {"x": res_heun["x"], "nfe": res_heun["nfe"], "error": err_heun}
    }