# src/mc_pricer.py
import numpy as np
from scipy.stats import norm

# ----------------------------
# Black-Scholes analytic
# ----------------------------
def bs_price(S0, K, r, sigma, T, option='call'):
    """Black-Scholes price for European call/put."""
    if T <= 0:
        return float(max(S0 - K, 0.0) if option == 'call' else max(K - S0, 0.0))
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == 'call':
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))

# ----------------------------
# Monte Carlo pricer (vectorized)
# ----------------------------
def mc_price_bs(S0, K, r, sigma, T, n_sims=200_000, option='call', seed=None, antithetic=True, return_paths=False):
    """
    Vectorized Monte Carlo under Black-Scholes lognormal terminal price.
    Returns (price_estimate, std_error, discounted_payoffs) by default.
    If return_paths=True returns (price, se, discounted_payoffs, S_T).
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_sims // 2
        Z_half = rng.standard_normal(size=half)
        Z = np.concatenate([Z_half, -Z_half])
        if len(Z) < n_sims:
            Z = np.concatenate([Z, rng.standard_normal(size=1)])
    else:
        Z = rng.standard_normal(size=n_sims)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S0 * np.exp(drift + diffusion)

    if option == 'call':
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoffs
    price = float(discounted.mean())
    std_error = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))
    if return_paths:
        return price, std_error, discounted, S_T
    return price, std_error, discounted

# ----------------------------
# Finite differences (central) for Greeks
# ----------------------------
def mc_greeks_finite_differences(S0, K, r, sigma, T, n_sims=200_000, option='call',
                                 bump_S=1e-2, bump_sigma=1e-4, seed=1234, antithetic=True):
    """Return dictionary with price, price_se, delta, vega (finite differences, central)."""
    price0, se0, _ = mc_price_bs(S0, K, r, sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    price_up, _, _ = mc_price_bs(S0 + bump_S, K, r, sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    price_dn, _, _ = mc_price_bs(S0 - bump_S, K, r, sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    delta = (price_up - price_dn) / (2 * bump_S)

    price_su, _, _ = mc_price_bs(S0, K, r, sigma + bump_sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    price_sd, _, _ = mc_price_bs(S0, K, r, max(1e-8, sigma - bump_sigma), T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    vega = (price_su - price_sd) / (2 * bump_sigma)

    return {'price': price0, 'price_se': se0, 'delta': delta, 'vega': vega}

# ----------------------------
# Pathwise Delta (for calls/puts)
# ----------------------------
def mc_delta_pathwise(S0, K, r, sigma, T, n_sims=200_000, option='call', seed=None, antithetic=True):
    """Return dict with price, delta_pw, delta_pw_se, price_se."""
    price, se, discounted, S_T = mc_price_bs(S0, K, r, sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic, return_paths=True)
    if option == 'call':
        indicator = (S_T > K).astype(float)
        pw = np.exp(-r * T) * indicator * (S_T / S0)
    else:
        indicator = (S_T < K).astype(float)
        pw = np.exp(-r * T) * (-indicator * (S_T / S0))
    delta = float(pw.mean())
    delta_se = float(pw.std(ddof=1) / np.sqrt(len(pw)))
    return {'price': price, 'delta_pw': delta, 'delta_pw_se': delta_se, 'price_se': se}

# ----------------------------
# Likelihood Ratio Delta
# ----------------------------
def mc_delta_lr(S0, K, r, sigma, T, n_sims=200_000, option='call', seed=None, antithetic=True):
    """LR estimator for delta; may have higher variance but works for discontinuous payoffs."""
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_sims // 2
        Z_half = rng.standard_normal(size=half)
        Z = np.concatenate([Z_half, -Z_half])
        if len(Z) < n_sims:
            Z = np.concatenate([Z, rng.standard_normal(size=1)])
    else:
        Z = rng.standard_normal(size=n_sims)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S0 * np.exp(drift + diffusion)
    if option == 'call':
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    discounted = np.exp(-r * T) * payoffs
    # WT used as internal Brownian representation
    WT = (np.log(S_T / S0) - (r - 0.5 * sigma**2) * T) / sigma
    score = WT / (S0 * sigma * np.sqrt(T))
    lr_samples = discounted * score
    lr = float(lr_samples.mean())
    lr_se = float(lr_samples.std(ddof=1) / np.sqrt(len(lr_samples)))
    return {'price': float(discounted.mean()), 'delta_lr': lr, 'delta_lr_se': lr_se, 'price_se': float(discounted.std(ddof=1)/np.sqrt(len(discounted)))}
