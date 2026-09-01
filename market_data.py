import yfinance as yf
import pandas as pd

def get_hourly_candles(ticker="PETR4.SA", period="60d", interval="60m"):
    """
    Downloads OHLCV candles from Yahoo Finance for the given ticker.

    Yahoo Finance limits intraday intervals (<1d) to the last 730 days,
    and tickers on B3 (Brazilian exchange) use the ".SA" suffix, e.g.
    "PETR4.SA" for Petrobras PN.
    """
    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' with interval '{interval}'")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data
