# src/sabr_model.py
import numpy as np

def sabr_simulate(F0, alpha0, beta, rho, nu, T, steps, n_paths, rng_seed=None):
    """
    Euler-ish simulation for SABR forward dynamics:
      dF = alpha * F^beta dW
      dalpha = nu * alpha dZ  (log-Euler for positivity)
      corr(dW,dZ) = rho
    Returns F_paths, alpha_paths (shape steps+1, n_paths).
    """
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    rng = np.random.default_rng(rng_seed)
    Z1 = rng.standard_normal(size=(steps, n_paths))
    Z2 = rng.standard_normal(size=(steps, n_paths))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    F = np.empty((steps + 1, n_paths))
    alpha = np.empty((steps + 1, n_paths))
    F[0, :] = F0
    alpha[0, :] = alpha0

    for t in range(steps):
        # log Euler for alpha
        alpha[t + 1, :] = alpha[t, :] * np.exp(-0.5 * nu**2 * dt + nu * sqrt_dt * Z2[t])
        # Euler for F (suitable for small dt); use positivity checks
        F[t + 1, :] = F[t, :] + alpha[t, :] * (F[t, :] ** beta) * sqrt_dt * Z1[t]
        # prevent negative F
        F[t + 1, :] = np.maximum(F[t + 1, :], 1e-12)
    return F, alpha

def hagan_sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """
    Hagan et al. (2002) SABR implied vol approximation.
    Returns implied volatility (decimal).
    This is vectorized for K as array or scalar.
    """
    F = float(F)
    K_arr = np.array(K, ndmin=1)
    ivs = np.zeros_like(K_arr, dtype=float)
    eps = 1e-07
    for i, Ki in enumerate(K_arr):
        if abs(F - Ki) < eps:
            # ATM asymptotic
            FK = F
            z = (nu / alpha) * (F ** (1 - beta))
            xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
            ivs[i] = (alpha / (F ** (1 - beta))) * (1 + (( (1 - beta)**2 * alpha**2 )/(24 * F**(2 - 2*beta)) + (rho * beta * nu * alpha)/(4 * F**(1 - beta)) + (2 - 3 * rho**2) * nu**2 / 24) * T)
        else:
            FK = F * Ki
            z = (nu / alpha) * (FK ** ((1 - beta) / 2)) * np.log(F / Ki)
            xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
            prefactor = alpha / ((F * Ki) ** ((1 - beta) / 2) * (1 + ((1 - beta)**2 / 24) * (np.log(F / Ki))**2 + ((1 - beta)**4 / 1920) * (np.log(F / Ki))**4))
            ivs[i] = prefactor * (z / xz) * (1 + (( (1 - beta)**2 * alpha**2 )/(24 * (FK)**(1 - beta)) + (rho * beta * nu * alpha)/(4 * (FK)**((1 - beta)/2)) + (2 - 3 * rho**2) * nu**2 / 24) * T)
    return float(ivs[0]) if np.isscalar(K) else ivs
