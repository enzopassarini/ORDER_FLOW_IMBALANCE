"""
Download intraday (1h) bars for 10 Brazilian large caps from Yahoo Finance.

Yahoo limits intraday history: 1h bars go back ~730 days, 5m/15m only ~60 days.
A 5-day time stop needs many multi-day trades, so 1h over 2 years is used.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3",
           "ABEV3", "WEGE3", "B3SA3", "ITSA4", "BPAC11"]
DATA_DIR = Path(__file__).parent / "data"


def fetch(interval: str = "1h", period: str = "730d") -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(exist_ok=True)
    out = {}
    for t in TICKERS:
        df = yf.download(f"{t}.SA", period=period, interval=interval,
                         progress=False, auto_adjust=True, multi_level_index=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = df.index.tz_convert("America/Sao_Paulo").tz_localize(None)
        df = df[df["Volume"] > 0]
        df.to_csv(DATA_DIR / f"{t}_{interval}.csv")
        out[t] = df
        print(f"{t:7s} {len(df):5d} bars  {df.index[0]} -> {df.index[-1]}")
    return out


def load(interval: str = "1h") -> dict[str, pd.DataFrame]:
    return {t: pd.read_csv(DATA_DIR / f"{t}_{interval}.csv",
                           index_col=0, parse_dates=True) for t in TICKERS}


if __name__ == "__main__":
    fetch()
