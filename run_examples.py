# run_examples.py
from src.utils import fetch_spot_history, historical_volatility, get_risk_free_rate_from_fred
from src.mc_pricer import bs_price, mc_price_bs, mc_greeks_finite_differences, mc_delta_pathwise, mc_delta_lr

def quick_run():
    series = fetch_spot_history('AAPL', period='2y')
    S0 = float(series.iloc[-1])
    sigma = historical_volatility(series)
    #r = 0.01
    r = get_risk_free_rate_from_fred()
    K = S0
    T = 30/252
    print(f"S0={S0:.2f}, sigma={sigma:.4f}, r={r}, K={K:.2f}, T={T:.4f}")

    bs = bs_price(S0, K, r, sigma, T, option='call')
    mc_price, mc_se, _ = mc_price_bs(S0, K, r, sigma, T, n_sims=100_000, option='call', seed=42, antithetic=True)
    print(f"BS price: {bs:.4f}")
    print(f"MC price: {mc_price:.4f} (SE {mc_se:.4f})")

    fd = mc_greeks_finite_differences(S0, K, r, sigma, T, n_sims=100_000, option='call', bump_S=0.5, bump_sigma=1e-3, seed=42)
    pw = mc_delta_pathwise(S0, K, r, sigma, T, n_sims=100_000, option='call', seed=42)
    lr = mc_delta_lr(S0, K, r, sigma, T, n_sims=100_000, option='call', seed=42)
    print("Finite diff:", fd)
    print("Pathwise:", pw)
    print("Likelihood ratio:", lr)

if __name__ == '__main__':
    quick_run()