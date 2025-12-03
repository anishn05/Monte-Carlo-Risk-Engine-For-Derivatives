# robust function to replace fetch_spot_history in src/utils.py
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

def fetch_spot_history(ticker='SPY', period='2y', interval='1d', verbose=False,
                       retries=3, pause=1.0, threads=False):
    """
    Robust spot fetcher:
    - tries yf.download() with threads option
    - falls back to Ticker.history()
    - final fallback to pandas_datareader stooq (if available)
    Returns a pandas Series of adjusted close or close prices (ascending index).
    """
    last_exc = None

    # 1) Try yf.download with a couple retries
    for attempt in range(1, retries + 1):
        try:
            if verbose:
                print(f"Attempt {attempt}: yf.download({ticker}, period={period}, interval={interval})")
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=threads)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                # prefer Adj Close if present
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                series = df[col].dropna()
                if len(series) == 0:
                    # sometimes exists but adj close empty -> treat as failure
                    last_exc = RuntimeError("download returned empty Adjusted/Close column")
                else:
                    if verbose:
                        print(f"yf.download succeeded, rows={len(series)}")
                    return series.sort_index()
            else:
                last_exc = RuntimeError("download returned empty dataframe")
        except Exception as e:
            last_exc = e
            if verbose:
                print(f"yf.download attempt {attempt} exception:", type(e).__name__, e)
        time.sleep(pause)

    # 2) Try Ticker.history() fallback
    try:
        if verbose:
            print("Trying fallback: yf.Ticker().history()")
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        if isinstance(hist, pd.DataFrame) and len(hist) > 0:
            col = 'Close' if 'Close' in hist.columns else hist.columns[0]
            series = hist[col].dropna()
            if len(series) > 0:
                if verbose:
                    print("Ticker.history() succeeded, rows=", len(series))
                return series.sort_index()
            else:
                last_exc = RuntimeError("Ticker.history returned empty price column")
        else:
            last_exc = RuntimeError("Ticker.history returned empty dataframe")
    except Exception as e:
        last_exc = e
        if verbose:
            print("Ticker.history() exception:", type(e).__name__, e)

    # 3) Final fallback: pandas_datareader stooq (if available)
    if pdr is not None:
        try:
            if verbose:
                print("Trying fallback: pandas_datareader stooq")
            st = pdr.DataReader(ticker, 'stooq')
            if isinstance(st, pd.DataFrame) and len(st) > 0:
                # stooq returns descending index; switch to ascending
                st = st.sort_index()
                col = 'Close' if 'Close' in st.columns else st.columns[0]
                series = st[col].dropna()
                if len(series) > 0:
                    if verbose:
                        print("pandas_datareader(stooq) succeeded, rows=", len(series))
                    return series
                else:
                    last_exc = RuntimeError("stooq returned no close column")
            else:
                last_exc = RuntimeError("stooq returned empty dataframe")
        except Exception as e:
            last_exc = e
            if verbose:
                print("stooq exception:", type(e).__name__, e)

    # If we get here, everything failed
    msg = f"All data sources failed for ticker {ticker}. Last exception: {repr(last_exc)}"
    raise RuntimeError(msg)

def historical_volatility(price_series, trading_days=252):
    """
    Compute annualized historical volatility from log returns.
    Uses sample std (ddof=1).
    """
    logrets = np.log(price_series / price_series.shift(1)).dropna()
    daily_std = logrets.std(ddof=1)
    annualized = daily_std * np.sqrt(trading_days)
    return float(annualized)

import os
from fredapi import Fred

def get_risk_free_rate_from_fred(series="DGS3MO"):
    """
    Fetch the most recent risk-free rate from FRED.
    
    series: FRED series ID (default = 3M T-bill rate)
    Returns the latest value as a decimal (e.g., 0.052 for 5.2%)
    """
    api_key = os.environ.get("FRED_API_KEY")
    if api_key is None:
        raise ValueError("FRED_API_KEY environment variable is not set.")
    
    fred = Fred(api_key=api_key)
    
    # Fetch entire series
    data = fred.get_series(series)
    latest_rate = data.dropna().iloc[-1]

    # Convert percent to decimal
    return float(latest_rate) / 100.0