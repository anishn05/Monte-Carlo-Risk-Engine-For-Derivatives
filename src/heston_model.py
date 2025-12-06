# src/heston_model.py
import numpy as np

def correlated_normals(rng, steps, n_paths, rho):
    """Return Zs of shape (steps, n_paths, 2): [Zs_price, Zs_var] correlated per-step."""
    Zs1 = rng.standard_normal(size=(steps, n_paths))
    Zs2 = rng.standard_normal(size=(steps, n_paths))
    Zs2 = rho * Zs1 + np.sqrt(1 - rho**2) * Zs2
    return Zs1, Zs2

def heston_simulate_full_truncation(S0, v0, r, kappa, theta, xi, rho,
                                    T, steps, n_paths, rng_seed=None):
    """
    Simulate Heston via full-truncation Euler.
    Returns S_paths (steps+1, n_paths), v_paths (steps+1, n_paths).
    Notes: Use even n_paths for antithetic outside if desired.
    """
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    rng = np.random.default_rng(rng_seed)

    S = np.empty((steps + 1, n_paths), dtype=float)
    v = np.empty((steps + 1, n_paths), dtype=float)
    S[0, :] = S0
    v[0, :] = v0

    Z1, Z2 = correlated_normals(rng, steps, n_paths, rho)

    for t in range(steps):
        vt = v[t]
        # ensure non-neg in diffusion sqrt, but drift uses vt (or truncated vt)
        vt_pos = np.maximum(vt, 0.0)
        dv = kappa * (theta - vt_pos) * dt + xi * np.sqrt(vt_pos) * sqrt_dt * Z2[t]
        v_next = vt + dv
        # enforce non-negative
        v_next = np.maximum(v_next, 0.0)
        v[t + 1] = v_next

        # Spot update using log-Euler with instantaneous variance (use v_next or vt_pos; full trunc uses v_next)
        S[t + 1] = S[t] * np.exp((r - 0.5 * v_next) * dt + np.sqrt(v_next) * sqrt_dt * Z1[t])

    return S, v

def mc_price_heston(S0, K, r, kappa, theta, xi, rho, v0, T,
                    steps=252, n_paths=200_000, option='call', rng_seed=None):
    """
    Monte Carlo price an European option under Heston using full-truncation Euler.
    Returns (price_est, se, discounted_payoffs).
    """
    S_paths, v_paths = heston_simulate_full_truncation(S0, v0, r, kappa, theta, xi, rho, T, steps, n_paths, rng_seed=rng_seed)
    S_T = S_paths[-1]
    if option == 'call':
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = float(discounted.mean())
    se = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))
    return price, se, discounted, S_T
