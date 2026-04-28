import os
import numpy as np
import pandas as pd
import streamlit as st
from config import STOCKS_FILE, INFLATION_FILE, TICKERS, DEFENSIVE, CYCLICAL, INFLATION_COL

@st.cache_data(show_spinner=False)
def download_stocks():
    """Download PSX stock data via yfinance and cache to CSV."""
    import yfinance as yf
    os.makedirs("data", exist_ok=True)
    prices = {}
    for name, tick in TICKERS.items():
        df = yf.download(tick, start="2010-01-01", end="2024-12-31", progress=False)
        if not df.empty:
            prices[name] = df["Close"].squeeze()
    raw = pd.DataFrame(prices)
    raw.to_csv(STOCKS_FILE)
    return raw

@st.cache_data(show_spinner=False)
def load_data():
    """
    Load (or download) all data → merge monthly → normalize 0–100.
    Same logic as original project: resample to monthly start, inner join,
    forward-fill, drop NaNs, then min-max scale everything to 0–100.
    """
    stocks_raw = (pd.read_csv(STOCKS_FILE, index_col=0, parse_dates=True)
                  if os.path.exists(STOCKS_FILE) else download_stocks())
    if not os.path.exists(INFLATION_FILE):
        st.error(f"Missing: `{INFLATION_FILE}`. Please place the official CPI CSV in data/.")
        st.stop()
    inflation = pd.read_csv(INFLATION_FILE, index_col=0, parse_dates=True)

    data = (stocks_raw.resample("MS").mean()
        .join(inflation, how="inner")
        .ffill().dropna())
    data_norm = (data - data.min()) / (data.max() - data.min()) * 100
    return data, data_norm

def sector_df(dn):
    """Sector averages + normalized Defensive/Cyclical ratio (Lipstick Effect proxy)."""
    s = pd.DataFrame(index=dn.index)
    s["Defensive"] = dn[DEFENSIVE].mean(axis=1)
    s["Cyclical"]  = dn[CYCLICAL].mean(axis=1)
    s["Inflation"] = dn[INFLATION_COL]
    r = s["Defensive"] / (s["Cyclical"] + 1e-4)
    s["Ratio"] = (r - r.min()) / (r.max() - r.min()) * 100
    return s

def ols_forecast(data, months=24):
    """OLS Linear Regression via numpy.polyfit on CPI → 24-month forward projection."""
    y = data[INFLATION_COL].values
    X = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(X, y, 1)
    fX = np.arange(len(y), len(y) + months, dtype=float)
    return dict(
        dates=data.index, actual=y, trend=slope * X + intercept,
        fdates=pd.date_range(data.index[-1], periods=months, freq="MS"),
        values=slope * fX + intercept, slope=slope,
    )

def compute_kpis(data, dn, fc):
    return dict(
        avg_inf   = round(data[INFLATION_COL].mean(), 2),
        peak_inf  = round(data[INFLATION_COL].max(), 1),
        peak_date = data[INFLATION_COL].idxmax().strftime("%b %Y"),
        cur_inf   = round(fc["actual"][-1], 2),
        fcast_inf = round(fc["values"][-1], 2),
        def_gain  = round(dn[DEFENSIVE].iloc[-1].mean() - dn[DEFENSIVE].iloc[0].mean(), 1),
        cyc_gain  = round(dn[CYCLICAL].iloc[-1].mean()  - dn[CYCLICAL].iloc[0].mean(),  1),
        months    = len(data),
    )
