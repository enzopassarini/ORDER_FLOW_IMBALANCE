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


def relative_to_basket(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Each stock divided by an equal-weight basket of all tickers (rebalanced every bar).
    Trading the ratio approximates long stock / short basket (or the reverse).
    The basket gets its own open (from constituents' opens) so ratio opens are tradable.
    Intrabar ratio High/Low are unknown: use the conservative of stock High/Low over the
    basket's open and close, bounded by the ratio open/close.
    """
    close = pd.DataFrame({t: d["Close"] for t, d in data.items()}).sort_index()
    opn = pd.DataFrame({t: d["Open"] for t, d in data.items()}).reindex(close.index)
    prev = close.ffill().shift(1)
    gap = (opn / prev - 1).mean(axis=1).fillna(0)          # basket prev close -> open
    intra = (close / opn - 1).mean(axis=1).fillna(0)       # basket open -> close
    b_close = ((1 + gap) * (1 + intra)).cumprod()
    b_open = b_close.shift(1).fillna(1.0) * (1 + gap)
    out = {}
    for t, d in data.items():
        bo, bc = b_open.reindex(d.index), b_close.reindex(d.index)
        o, c = d["Open"] / bo, d["Close"] / bc
        h = pd.concat([o, c, pd.concat([d["High"] / bo, d["High"] / bc], axis=1).min(axis=1)], axis=1).max(axis=1)
        l = pd.concat([o, c, pd.concat([d["Low"] / bo, d["Low"] / bc], axis=1).max(axis=1)], axis=1).min(axis=1)
        out[t] = pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=d["Volume"]))
    return out
