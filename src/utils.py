# src/utils.py
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

def fetch_spot_history(ticker='SPY', period='2y', interval='1d', verbose=False):
    """
    Fetch adjusted close price series for ticker using yfinance.
    Returns a pandas Series of adjusted close prices.
    """
    # yfinance sometimes needs override for pandas_datareader users
    try:
        yf.pdr_override()
    except Exception:
        # not critical if override fails
        pass
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    # use Adjusted Close if present
    if 'Adj Close' in df.columns:
        series = df['Adj Close'].dropna()
    else:
        series = df['Close'].dropna()
    if verbose:
        print(f"Fetched {len(series)} rows for {ticker} (last={series.index[-1].date()})")
    return series

def historical_volatility(price_series, trading_days=252):
    """
    Compute annualized historical volatility from log returns.
    Uses sample std (ddof=1).
    """
    logrets = np.log(price_series / price_series.shift(1)).dropna()
    daily_std = logrets.std(ddof=1)
    annualized = daily_std * np.sqrt(trading_days)
    return float(annualized)

def get_risk_free_rate_from_fred(series='DGS1', start=None, end=None, verbose=False):
    """
    Fetch yield (percentage) from FRED and convert to decimal (e.g. 1.2 -> 0.012).
    Requires pandas-datareader. May return None if series not available.
    """
    try:
        df = pdr.DataReader(series, 'fred', start=start, end=end)
    except Exception as e:
        if verbose:
            print("Failed to fetch FRED series:", e)
        return None
    if df is None or df.empty:
        return None
    val = df.iloc[-1].values[0]
    if pd.isna(val):
        return None
    return float(val) / 100.0