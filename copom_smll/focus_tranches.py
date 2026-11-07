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

# --- plot: tranche backtest with position over time ---------------------------------------
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
AQUA, ORANGE, GRAY = "#1baf7a", "#eb6834", "#9b9a95"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
T, W = "4 tranches x 21d", "Weekly (current)"
eq = (1 + ret).cumprod()
fig, (ax, axp, axd) = plt.subplots(3, 1, figsize=(10, 8.6), dpi=200, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.4, 1.1], "hspace": 0.1})
for a in (ax, axp, axd):
    a.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax.plot(eq.index, eq[W], color=GRAY, lw=1.3, label=f"Weekly rebalance (Sharpe {tab.loc[W, 'Sharpe']:.2f})")
ax.plot(eq.index, eq[T], color=AQUA, lw=2.2, label=f"4 tranches × 21 days (Sharpe {tab.loc[T, 'Sharpe']:.2f})")
hi = T if eq[T].iloc[-1] >= eq[W].iloc[-1] else W
for k, c in ((T, INK2), (W, MUTED)):
    ax.annotate(f"{eq[k].iloc[-1]:.2f}×", (eq.index[-1], eq[k].iloc[-1]), xytext=(6, 6 if k == hi else -6),
                textcoords="offset points", fontsize=9, color=c, va="center")
split = ret.index[mid]
for a in (ax, axp, axd):
    a.axvline(split, color=MUTED, lw=1, ls="--")
ax.annotate("half split", (split, 1), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
ax.axhline(1, color=MUTED, lw=1)
ax.set_ylabel("Growth of 1")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
p = pos[T]
axp.fill_between(p.index, 0, p.clip(lower=0), color=AQUA, lw=0, step="post", label="Long SMLL / short IBOV")
axp.fill_between(p.index, 0, p.clip(upper=0), color=ORANGE, lw=0, step="post", label="Short SMLL / long IBOV")
axp.axhline(0, color=MUTED, lw=1)
for y in (-0.5, 0.5):
    axp.axhline(y, color=GRID, lw=0.8)
axp.set_ylim(-1.05, 1.05)
axp.set_ylabel("Position\n(avg of 4 tranches)")
axp.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
dd = (eq / eq.cummax() - 1) * 100
axd.plot(dd.index, dd[W], color=GRAY, lw=0.9)
axd.plot(dd.index, dd[T], color=AQUA, lw=1.2)
axd.set_ylabel("Drawdown (%)")
t = tab.loc[T]
fig.suptitle("Focus GDP-revision backtest: 4 staggered 21-day tranches (no costs)", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"Each Friday survey opens a tranche = clip(rev, ±0.75 pp)/0.75 ÷ 4, held 4 weeks from Monday close; "
             f"position = sum of live tranches.\nTranches: CAGR {t.CAGR:+.1%}, vol {t.Vol:.1%}, Sharpe {t.Sharpe:.2f} "
             f"(1st half {t['Sharpe 1st half']:.2f}, 2nd half {t['Sharpe 2nd half']:.2f}), max DD {t.MaxDD:.0%}, "
             f"avg |position| {t['avg |pos|']:.2f}.", loc="left", fontsize=8.5, color=INK2, pad=10)
fig.text(0.07, 0.01, "Return = position × (r_SMLL − r_IBOV), daily-rebalanced dollar-neutral spread. "
         "Sources: BCB Focus, B3.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.1, right=0.93, top=0.87, bottom=0.07)
fig.savefig(HERE / "focus_tranches_backtest.png")
print("saved", HERE / "focus_tranches_backtest.png")
