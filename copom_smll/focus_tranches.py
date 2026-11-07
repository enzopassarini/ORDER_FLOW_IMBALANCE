"""
Weekly signal vs. 4 staggered 21-day tranches (quick test, no costs).

Tranche version: each Friday survey opens a tranche of size signal_t / 4, held 4 weeks
(~21 sessions), so position_t = mean(signal_t, signal_t-1w, signal_t-2w, signal_t-3w).
Consecutive same-side signals stack up to full size; a single signal is 1/4 size.
Also: next-week return regressed on the current and lagged signals (does an old signal
still predict once you know the new one?).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
sig = pd.read_csv(DATA / "focus_signal_weekly.csv", index_col=0, parse_dates=True)["sig_cap075"]
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
spread = (px["SMLL"].pct_change() - px["IBOV"].pct_change()).dropna()

variants = {"Weekly (current)": sig, "4 tranches x 21d": sig.rolling(4).mean()}


def daily_pos(s):
    p = pd.Series(np.nan, index=spread.index)
    for d, v in s.dropna().items():
        i0 = px.index.searchsorted(d, side="right")
        if i0 < len(px):
            p.loc[p.index > px.index[i0]] = v
    return p


pos = pd.DataFrame({k: daily_pos(v) for k, v in variants.items()}).dropna()
sp = spread.loc[pos.index]
ret = pos.mul(sp, axis=0)


def perf(r):
    eq = (1 + r).cumprod()
    return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "Vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min()}


mid = len(ret) // 2
rows = []
for k in ret:
    rows.append({"variant": k, **perf(ret[k]), "Sharpe 1st half": perf(ret[k].iloc[:mid])["Sharpe"],
                 "Sharpe 2nd half": perf(ret[k].iloc[mid:])["Sharpe"],
                 "avg |pos|": pos[k].abs().mean(),
                 "weekly turnover": pos[k].resample("W-FRI").last().diff().abs().mean()})
tab = pd.DataFrame(rows).set_index("variant")
tab.round(3).to_csv(DATA / "focus_tranches_stats.csv")
pd.options.display.width = 200
print(f"{ret.index[0].date()} -> {ret.index[-1].date()}")
print(tab.round(3))
print(f"corr(daily returns) = {ret.iloc[:, 0].corr(ret.iloc[:, 1]):.3f}")

# lag regressions on the next-week return (non-overlapping)
nw = pd.read_csv(DATA / "focus_fwd_nextweek_cap075_deciles.csv", index_col=0, parse_dates=True)
X = pd.DataFrame({f"sig_lag{l}w": sig.shift(l) for l in range(4)}).reindex(nw.index)
d = X.join(nw["fwd_rel"] * 100).dropna()
print(f"\nAutocorrelation of the weekly signal: lag1 {sig.autocorr(1):.2f}, lag2 {sig.autocorr(2):.2f}, "
      f"lag3 {sig.autocorr(3):.2f}, lag4 {sig.autocorr(4):.2f}")
for cols, label in [(["sig_lag0w"], "current only"), (list(X.columns), "current + 3 lags")]:
    m = sm.OLS(d["fwd_rel"], sm.add_constant(d[cols])).fit()
    print(f"\nNext-week return (pp) on {label}: n={int(m.nobs)} R2={m.rsquared:.4f}")
    print(pd.DataFrame({"coef": m.params, "t": m.tvalues}).round(3).to_string())
m = sm.OLS(d["fwd_rel"], sm.add_constant(d[list(X.columns)].mean(axis=1).rename("avg4"))).fit()
print(f"\nNext-week return on 4-week average signal: coef {m.params['avg4']:.3f} t {m.tvalues['avg4']:.2f} R2 {m.rsquared:.4f}")
