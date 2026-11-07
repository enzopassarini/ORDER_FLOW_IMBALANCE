"""
Quick and dirty backtest of the Focus GDP-revision signal, no costs.

Signal (weekly, Friday survey t): rev = (rev4 + rev8) / 2 on next-year GDP (pp),
position = clip(rev, -0.75, 0.75) / 0.75 in [-1, 1]  (upgrade -> long SMLL / short IBOV).
Entered at the close of the first session after Friday (Monday, when the report is
public), held until the next week's position replaces it. Daily return =
position * (r_SMLL - r_IBOV), a daily-rebalanced dollar-neutral spread.
Reads data/focus_signal_weekly.csv (written by focus_analysis.py).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

sig = pd.read_csv(DATA / "focus_signal_weekly.csv", index_col=0, parse_dates=True)["sig_cap075"].dropna()
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
spread = (px["SMLL"].pct_change() - px["IBOV"].pct_change()).dropna()

pos = pd.Series(np.nan, index=spread.index)
for d, s in sig.items():
    i0 = px.index.searchsorted(d, side="right")       # Monday = first session after Friday survey
    if i0 < len(px):
        pos.loc[pos.index > px.index[i0]] = s           # earns returns from the day after entry
pos = pos.dropna()
spread = spread.loc[pos.index]
strat = pos * spread
bt = pd.DataFrame({"position": pos, "spread_ret": spread, "strategy_ret": strat})
bt["equity"] = (1 + strat).cumprod()
bt.to_csv(DATA / "focus_bt_cap075_daily.csv")


def perf(r):
    eq = (1 + r).cumprod()
    return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "Vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min(),
            "Total": eq.iloc[-1] - 1}


mid = len(strat) // 2
tab = pd.DataFrame({
    "Full": perf(strat),
    f"First half ({strat.index[0]:%Y}–{strat.index[mid - 1]:%Y})": perf(strat.iloc[:mid]),
    f"Second half ({strat.index[mid]:%Y}–{strat.index[-1]:%Y})": perf(strat.iloc[mid:]),
    "Constant avg position": perf(spread * pos.mean()),
}).T
tab.round(3).to_csv(DATA / "focus_bt_cap075_stats.csv")
pd.options.display.width = 200
print(f"{strat.index[0].date()} -> {strat.index[-1].date()}; avg position {pos.mean():+.3f}, avg |pos| {pos.abs().mean():.3f}")
print(tab.round(3))

# --- plot ---
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
AQUA, ORANGE = "#1baf7a", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (ax, axp, axd) = plt.subplots(3, 1, figsize=(10, 8.4), dpi=200, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.1, 1.1], "hspace": 0.1})
for a in (ax, axp, axd):
    a.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax.plot(bt.index, bt["equity"], color=AQUA, lw=2)
ax.axhline(1, color=MUTED, lw=1)
ax.axvline(strat.index[mid], color=MUTED, lw=1, ls="--")
ax.annotate("half split", (strat.index[mid], ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
            fontsize=8, color=MUTED)
ax.annotate(f"{bt['equity'].iloc[-1]:.2f}×", (bt.index[-1], bt["equity"].iloc[-1]), xytext=(6, 0),
            textcoords="offset points", fontsize=9, color=INK2, va="center")
ax.set_ylabel("Growth of 1")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
axp.fill_between(pos.index, 0, pos.clip(lower=0), color=AQUA, lw=0, step="post")
axp.fill_between(pos.index, 0, pos.clip(upper=0), color=ORANGE, lw=0, step="post")
axp.axhline(0, color=MUTED, lw=1)
axp.set_ylim(-1.05, 1.05)
axp.set_ylabel("Position")
dd = (bt["equity"] / bt["equity"].cummax() - 1) * 100
axd.plot(dd.index, dd, color=AQUA, lw=1)
axd.set_ylabel("Drawdown (%)")
f, h1, h2 = tab.iloc[0], tab.iloc[1], tab.iloc[2]
fig.suptitle("Focus GDP-revision backtest: SMLL vs. IBOV spread (no costs)", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"Position = clip(rev, ±0.75 pp) / 0.75, set weekly at Monday close.  Full: CAGR {f.CAGR:+.1%}, "
             f"Sharpe {f.Sharpe:.2f}, max DD {f.MaxDD:.0%}\nFirst half Sharpe {h1.Sharpe:.2f} · Second half Sharpe "
             f"{h2.Sharpe:.2f} · Constant average position ({pos.mean():+.2f}×) Sharpe {tab.iloc[3].Sharpe:.2f}",
             loc="left", fontsize=8.5, color=INK2, pad=10)
fig.text(0.07, 0.01, "Return = position × (r_SMLL − r_IBOV), daily-rebalanced dollar-neutral spread. "
         "Sources: BCB Focus, B3.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.09, right=0.93, top=0.87, bottom=0.07)
fig.savefig(HERE / "focus_bt_cap075.png")
print("saved")
