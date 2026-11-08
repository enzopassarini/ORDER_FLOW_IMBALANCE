"""
Real-rate signal: weekly rebalance vs. 4 staggered 21-day tranches (no costs).

Tranches: each Friday survey opens a tranche = signal / 4 held 4 weeks from the Monday close,
so position = mean of the last 4 weekly signals. Also re-runs the GDP + real-rate equal-risk
aggregate with each version. Same sample as aggregate3.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
spread = (px["SMLL"].pct_change() - px["IBOV"].pct_change()).dropna()
w = pd.read_csv(DATA / "focus_channels_signal.csv", index_col=0, parse_dates=True)["real_rate_r5"].dropna()
gdp = pd.read_csv(DATA / "focus_tranches_daily.csv", index_col=0, parse_dates=True)["4 tranches x 21d"]


def daily(s):
    p = pd.Series(np.nan, index=spread.index)
    for d, v in s.dropna().items():
        i0 = px.index.searchsorted(d, side="right")
        if i0 < len(px):
            p.loc[p.index > px.index[i0]] = v
    return p


pos = pd.DataFrame({"Real rate weekly": daily(w), "Real rate 4 tranches": daily(w.rolling(4).mean()),
                    "GDP": gdp}).dropna()
sp = spread.loc[pos.index]
ret = pos.mul(sp, axis=0)
for rr in ("Real rate weekly", "Real rate 4 tranches"):
    vol = ret[["GDP", rr]].expanding(min_periods=252).std().shift(1)
    inv = 1 / vol
    wt = inv.div(inv.sum(axis=1), axis=0).fillna(0.5)
    k = f"GDP + {rr}"
    pos[k] = (wt * pos[["GDP", rr]]).sum(axis=1)
    ret[k] = pos[k] * sp


def perf(r, p):
    eq = (1 + r).cumprod()
    wk = p.resample("W-FRI").last()
    return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "Vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min(),
            "weekly turnover": wk.diff().abs().mean()}


mid = len(ret) // 2
cols = ["Real rate weekly", "Real rate 4 tranches", "GDP + Real rate weekly", "GDP + Real rate 4 tranches"]
tab = pd.DataFrame({k: {**perf(ret[k], pos[k]), "Sharpe H1": perf(ret[k].iloc[:mid], pos[k].iloc[:mid])["Sharpe"],
                        "Sharpe H2": perf(ret[k].iloc[mid:], pos[k].iloc[mid:])["Sharpe"]} for k in cols}).T
tab.to_csv(DATA / "real_rate_tranches_stats.csv", float_format="%.4f")
pd.options.display.width = 200
print(f"{ret.index[0].date()} -> {ret.index[-1].date()}")
print(tab.round(3).to_string())
print(f"\ncorr(daily returns) weekly vs tranches: {ret['Real rate weekly'].corr(ret['Real rate 4 tranches']):.3f}")
print(f"weekly signal autocorrelation: lag1 {w.autocorr(1):.3f}, lag4 {w.autocorr(4):.3f}")
