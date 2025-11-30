# tests/test_mc.py
import math
from src.mc_pricer import bs_price, mc_price_bs

def test_mc_vs_bs():
    S0 = 100.0
    K = 100.0
    r = 0.01
    sigma = 0.2
    T = 30/252
    bs = bs_price(S0, K, r, sigma, T, option='call')
    mc, se, _ = mc_price_bs(S0, K, r, sigma, T, n_sims=100000, seed=123, antithetic=True)
    # Accept within 4 standard errors
    assert abs(mc - bs) < 4 * se