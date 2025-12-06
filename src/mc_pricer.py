# src/mc_pricer.py
import numpy as np
from scipy.stats import norm
from src.heston_model import mc_price_heston
from src.sabr_model import sabr_simulate, hagan_sabr_implied_vol

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
def mc_greeks_finite_differences(S0, K, r, sigma, T,
                                 n_sims=200_000, option='call',
                                 bump_S=1e-2, bump_sigma=1e-4, bump_T=1/365,
                                 seed=1234, antithetic=True):
    """
    Compute Greeks by central finite differences using common random numbers.
    Returns dict: price, price_se, delta, gamma, vega, theta.
    theta returned is conventional (dV/dt) so typically negative.
    bump_T is in years (default 1 calendar day ~ 1/365).
    """
    # baseline
    price0, se0, _ = mc_price_bs(S0, K, r, sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)

    # Delta (central)
    price_up, _, _ = mc_price_bs(S0 + bump_S, K, r, sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    price_dn, _, _ = mc_price_bs(S0 - bump_S, K, r, sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    delta = (price_up - price_dn) / (2 * bump_S)

    # Gamma (central second derivative)
    gamma = (price_up - 2 * price0 + price_dn) / (bump_S ** 2)

    # Vega (central)
    price_su, _, _ = mc_price_bs(S0, K, r, sigma + bump_sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    price_sd, _, _ = mc_price_bs(S0, K, r, max(1e-12, sigma - bump_sigma), T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    vega = (price_su - price_sd) / (2 * bump_sigma)

    # Theta: dV/dt. We compute derivative wrt time to maturity T and return theta = -dP/dT (conventional)
    # Use central difference in T (ensure T - bump_T > 0)
    T_up = T + bump_T
    T_dn = max(1e-12, T - bump_T)
    price_T_up, _, _ = mc_price_bs(S0, K, r, sigma, T_up, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    price_T_dn, _, _ = mc_price_bs(S0, K, r, sigma, T_dn, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic)
    # derivative dP/dT approximated by central diff
    dPdT = (price_T_up - price_T_dn) / (T_up - T_dn)
    theta = -dPdT

    return {'price': price0, 'price_se': se0, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

# ----------------------------
# Pathwise estimators
# ----------------------------
def mc_greeks_pathwise(S0, K, r, sigma, T,
                       n_sims=200_000, option='call',
                       seed=None, antithetic=True):
    """
    Pathwise estimators for Delta, Vega, Theta.
    Returns dict: price, price_se, delta_pw, delta_pw_se, vega_pw, vega_pw_se, theta_pw, theta_pw_se.
    Note: pathwise Gamma is not implemented because for vanilla payoff the payoff is not smooth at the strike
    (indicator function), and the second derivative pathwise is not defined in the pointwise sense.
    Use finite differences for Gamma.
    """
    # Generate paths and S_T
    price, se, discounted, S_T = mc_price_bs(S0, K, r, sigma, T, n_sims=n_sims, option=option, seed=seed, antithetic=antithetic, return_paths=True)
    n = len(discounted)
    # recover the standard normal representation W from S_T:
    # Z = (ln(S_T/S0) - (r - 0.5 sigma^2) T) / (sigma sqrt(T))
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (np.log(S_T / S0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # Pathwise delta (call): e^{-rT} * I(S_T>K) * S_T / S0
    if option == 'call':
        indicator = (S_T > K).astype(float)
        pw_delta_samples = np.exp(-r * T) * indicator * (S_T / S0)
    else:
        indicator = (S_T < K).astype(float)
        pw_delta_samples = np.exp(-r * T) * (-indicator * (S_T / S0))
    delta_pw = float(pw_delta_samples.mean())
    delta_pw_se = float(pw_delta_samples.std(ddof=1) / np.sqrt(n))

    # Pathwise Vega:
    # dS_T/dsigma = S_T * ( -sigma * T + sqrt(T) * Z )
    # so per-path vega sample = e^{-rT} * I(S_T>K) * dS_T/dsigma
    common_indicator = ((S_T > K).astype(float) if option == 'call' else (S_T < K).astype(float))
    dS_dsigma = S_T * (-sigma * T + np.sqrt(T) * Z)
    pw_vega_samples = np.exp(-r * T) * common_indicator * dS_dsigma
    vega_pw = float(pw_vega_samples.mean())
    vega_pw_se = float(pw_vega_samples.std(ddof=1) / np.sqrt(n))

    # Pathwise Theta:
    # We compute derivative of discounted payoff wrt T (time to maturity). Theta = -dP/dT (conventional).
    # d/dT [ e^{-rT} * payoff(S_T) ] = e^{-rT} * [ -r * payoff + d(payoff)/dT ]
    # d(payoff)/dT = indicator * dS_T/dT
    # dS_T/dT = S_T * (r - 0.5 sigma^2 + (sigma * Z) / (2 sqrt(T)) )
    dS_dT = S_T * ( (r - 0.5 * sigma**2) + (sigma * Z) / (2.0 * np.sqrt(np.maximum(T, 1e-12))) )
    pw_dpayoff_dT = common_indicator * dS_dT
    pw_theta_samples = - ( np.exp(-r * T) * ( -r * (np.maximum(S_T - K, 0.0) if option == 'call' else np.maximum(K - S_T, 0.0)) + pw_dpayoff_dT ) )
    # Explanation: pw_theta_samples already returned with conventional sign (theta).
    theta_pw = float(pw_theta_samples.mean())
    theta_pw_se = float(pw_theta_samples.std(ddof=1) / np.sqrt(n))

    return {
        'price': price, 'price_se': se,
        'delta_pw': delta_pw, 'delta_pw_se': delta_pw_se,
        'vega_pw': vega_pw, 'vega_pw_se': vega_pw_se,
        'theta_pw': -theta_pw, 'theta_pw_se': theta_pw_se,
        'gamma_pw': None  # not available via pathwise for vanilla options (see docstring)
    }

# ----------------------------
# Likelihood Ratio estimators
# ----------------------------
def mc_greeks_lr(S0, K, r, sigma, T,
                 n_sims=200_000, option='call',
                 seed=None, antithetic=True):
    """
    Likelihood-ratio (score function) estimators for Delta, Vega, Theta (first derivatives).
    Returns dict with price, price_se, delta_lr, delta_lr_se, vega_lr, vega_lr_se, theta_lr, theta_lr_se.
    Gamma (second derivative) is not returned — second-order score function is possible but often high-variance and not included.
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
    n = len(discounted)

    # Common quantities for LR
    Y = np.log(S_T)                                # observed log-price
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T     # mean of Y
    var = (sigma**2) * T                           # variance of Y

    # WT (standard normal representation)
    with np.errstate(divide='ignore', invalid='ignore'):
        WT = (Y - mu) / (sigma * np.sqrt(T))
        WT = np.nan_to_num(WT, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- Delta (LR)
    score_delta = WT / (S0 * sigma * np.sqrt(T))
    delta_samples = discounted * score_delta
    delta_lr = float(delta_samples.mean())
    delta_lr_se = float(delta_samples.std(ddof=1) / np.sqrt(n))

    # ---------------- Vega (LR) -- corrected
    # Derived expression (numerically stable using WT):
    # score_vega = (WT**2 - 1)/sigma - sqrt(T)*WT
    score_vega = (WT**2 - 1.0) / sigma - np.sqrt(T) * WT
    vega_samples = discounted * score_vega
    vega_lr = float(vega_samples.mean())
    vega_lr_se = float(vega_samples.std(ddof=1) / np.sqrt(n))

    # ---------------- Theta (LR) - corrected
    # Use analytic derivative of log-pdf wrt T:
    # dvar_dT = sigma^2
    # dmu_dT = r - 0.5 * sigma^2
    # score_theta = -0.5*(1/var)*dvar_dT + ((Y - mu) ** 2)/(2*var**2)*dvar_dT + (Y - mu)*dmu_dT/var
    dvar_dT = sigma ** 2
    dmu_dT = r - 0.5 * sigma ** 2
    # compute safely
    term1 = -0.5 * (1.0 / var) * dvar_dT
    term2 = ((Y - mu) ** 2) / (2.0 * (var ** 2)) * dvar_dT
    term3 = (Y - mu) * dmu_dT / var
    score_theta = term1 + term2 + term3

    # Combine with discount derivative -r * payoff
    theta_samples = np.exp(-r * T) * ( -r * payoffs + payoffs * score_theta )
    theta_lr = float(theta_samples.mean())
    theta_lr_se = float(theta_samples.std(ddof=1) / np.sqrt(n))

    return {
        'price': price, 'price_se': float(discounted.std(ddof=1) / np.sqrt(n)),
        'delta_lr': delta_lr, 'delta_lr_se': delta_lr_se,
        'vega_lr': vega_lr, 'vega_lr_se': vega_lr_se,
        'theta_lr': theta_lr, 'theta_lr_se': theta_lr_se,
        'gamma_lr': None  # not provided (second-order LR omitted)
    }

# ----------------------------
# Convenience wrapper: compute all three-method Greeks
# ----------------------------
def mc_all_greeks(S0, K, r, sigma, T, n_sims=200_000, option='call', seed=1234, antithetic=True):
    """
    Compute Greeks by: finite differences, pathwise, likelihood-ratio.
    Returns dict of dicts:
      {
        'finite': {...},
        'pathwise': {...},
        'lr': {...}
      }
    """
    finite = mc_greeks_finite_differences(S0, K, r, sigma, T,
                                          n_sims=n_sims, option=option,
                                          seed=seed, antithetic=antithetic)
    pathwise = mc_greeks_pathwise(S0, K, r, sigma, T,
                                  n_sims=n_sims, option=option,
                                  seed=seed, antithetic=antithetic)
    lr = mc_greeks_lr(S0, K, r, sigma, T,
                      n_sims=n_sims, option=option,
                      seed=seed, antithetic=antithetic)
    return {'finite': finite, 'pathwise': pathwise, 'lr': lr}

def price_option_heston(S0, K, r, kappa, theta, xi, rho, v0, T, steps=252, n_paths=100000, option='call', seed=42):
    """
    Wrapper that returns price and se by Monte Carlo using Heston.
    """
    price, se, discounted, S_T = mc_price_heston(S0, K, r, kappa, theta, xi, rho, v0, T, steps=steps, n_paths=n_paths, option=option, rng_seed=seed)
    return price, se

def price_option_sabr_via_hagan(S0, K, r, T, alpha, beta, rho, nu, option='call'):
    """
    For vanilla: get implied vol from Hagan SABR and price with Black (quick).
    This DOES NOT simulate SABR paths; it's used for quoting/pricing/IV surface generation.
    """
    # For forward F, if dividend yield q: adjust; here assume F ≈ S0*exp((r-q)T) — for simplicity assume no div
    F = S0 * np.exp(r * T)  # crude forward; if q known, F=S0*exp((r-q)T)
    iv = hagan_sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
    # price via Black under forward (or use bs_price with S0, r)
    price = bs_price(S0, K, r, iv, T, option=option)
    return price, iv