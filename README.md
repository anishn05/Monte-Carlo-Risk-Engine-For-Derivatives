# Monte Carlo Derivatives Pricer (Python)

This repo implements a vectorized Monte Carlo engine for pricing European options
under Black–Scholes (GBM), and implements three estimators for Greeks:
- finite-difference (central bump),
- pathwise estimator,
- likelihood ratio estimator.

It also demonstrates variance reduction (antithetic sampling) and validation vs
the Black–Scholes closed-form formula.

## Quickstart
1. Create venv and install requirements: `pip install -r requirements.txt`
2. Run `python run_examples.py` for a quick check.
3. Open `notebooks/mc_pricer.ipynb` to run experiments and produce plots.

## Data sources
- Underlying price history: Yahoo Finance via `yfinance`
- Risk-free rates (optional): FRED via `pandas_datareader`
