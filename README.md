# Monte Carlo Pricing Engine for Options

## Overview
This project implements a suite of Monte Carlo simulation techniques to price vanilla and exotic options and compute their sensitivities (Greeks). The focus is on stochastic volatility models, path-dependent options, and variance reduction methods to produce accurate and numerically robust results.

The notebook demonstrates end-to-end workflows, including simulation, numerical validation, and reporting of pricing and Greek estimates with standard errors and convergence checks.

---

## Implemented Methodologies

### Monte Carlo Simulation for Vanilla Options
- European Call and Put options
- Finite difference and pathwise Greeks computation

### Stochastic Volatility Models
- Heston model simulation
- SABR model simulation and pricing

### Exotic Options
- Asian options (arithmetic & geometric averages)
- Barrier options (up-and-out, down-and-out, up-and-in, down-and-in)

### Variance Reduction Techniques
- Antithetic variates
- Control variates
- Common random numbers (CRN)

### Greeks Computation
- Delta, Gamma, Vega, Theta
- Finite differences, pathwise, and likelihood ratio methods

### Numerical Validation & Reporting
- Standard error computation
- Convergence plots vs. number of Monte Carlo paths


