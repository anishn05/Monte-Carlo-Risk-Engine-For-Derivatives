# run_examples.py
from src.utils import fetch_spot_history, historical_volatility, get_risk_free_rate_from_fred
from src.mc_pricer import bs_price, mc_price_bs, mc_greeks_finite_differences, mc_greeks_pathwise, mc_greeks_lr, price_option_sabr_via_hagan, mc_price_heston
from src.exotics import mc_price_asian

def quick_run():
    ser = fetch_spot_history('AAPL', period='2y')
    S0 = float(ser.iloc[-1])
    sigma = historical_volatility(ser)
    r = 0.037
    #r = get_risk_free_rate_from_fred()
    K = S0
    T = 30/252
    print(f"S0={S0:.2f}, sigma={sigma:.4f}, r={r}, K={K:.2f}, T={T:.4f}")

    bs = bs_price(S0, K, r, sigma, T, option='call')
    mc_price, mc_se, _ = mc_price_bs(S0, K, r, sigma, T, n_sims=100_000, option='call', seed=42, antithetic=True)
    print(f"BS price: {bs:.4f}")
    print(f"MC price: {mc_price:.4f} (SE {mc_se:.4f})")

    fd = mc_greeks_finite_differences(S0, K, r, sigma, T, n_sims=100_000, option='call', bump_S=0.5, bump_sigma=1e-3, seed=42)
    pw = mc_greeks_pathwise(S0, K, r, sigma, T, n_sims=100_000, option='call', seed=42)
    lr = mc_greeks_lr(S0, K, r, sigma, T, n_sims=100_000, option='call', seed=42)
    print("Finite diff:", fd)
    print("Pathwise:", pw)
    print("Likelihood ratio:", lr)

def run_with_stoch_vols():
    hparams = {'kappa':1.2, 'theta':0.08, 'xi':0.5, 'rho':-0.7, 'v0':0.2786**2}
    price, se = mc_price_asian(284.15, 284.15, 0.037, None, 0.119, steps=30, n_paths=50000,
                           option='call', avg_type='arithmetic', model='heston',
                           heston_params=hparams, seed=42, batch_size=10000)
    print(price, se)

if __name__ == '__main__':
    run_with_stoch_vols()