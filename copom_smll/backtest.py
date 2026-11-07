"""
Simplified COPOM small-cap vs. large-cap backtest (no transaction costs).

Rule, decided after each COPOM decision k:
  Selic cut  -> long SMLL / short IBOV   (+1)
  Selic hike -> short SMLL / long IBOV   (-1)
  Hold       -> flat                     ( 0)

Timing: the decision is announced after the B3 close, so the position is
entered at the close of the first session after decision k and held until the
close of the first session after decision k+1, when the next signal takes over.

The position is a dollar-neutral spread rebalanced daily, so its daily return
is pos * (r_SMLL - r_IBOV). No costs, no financing, no borrow fees. Because the
spread is self-financing, Sharpe is computed on raw spread returns.

Benchmark: always long SMLL / short IBOV (+1 every day).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
copom = pd.read_csv(DATA / "copom_decisions.csv", parse_dates=["date"])
copom = copom[copom["date"] >= px.index[0]].reset_index(drop=True)

rets = px.pct_change().dropna()
spread = rets["SMLL"] - rets["IBOV"]


def next_session(d):
    after = px.index[px.index > d]
    return after[0] if len(after) else None


# --- daily position ----------------------------------------------------------
copom["signal"] = -np.sign(copom["delta_bp"]).astype(int)
copom["entry"] = copom["date"].map(next_session)
copom = copom.dropna(subset=["entry"]).reset_index(drop=True)

pos = pd.Series(np.nan, index=spread.index)
for _, r in copom.iterrows():
    # return on day d belongs to the position held from close d-1 -> close d
    pos.loc[pos.index > r["entry"]] = r["signal"]
pos = pos.dropna()
spread = spread.loc[pos.index]

strat = pos * spread
bench = spread
curves = pd.DataFrame({"Strategy": (1 + strat).cumprod(), "Always long SMLL / short IBOV": (1 + bench).cumprod()})


# --- stats -------------------------------------------------------------------
def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = eq / eq.cummax() - 1
    return {"CAGR": cagr, "Vol": vol, "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "Max DD": dd.min(), "Total": eq.iloc[-1] - 1}


tab = pd.DataFrame({"Strategy": stats(strat), "Benchmark": stats(bench)}).T
tab["Time in market"] = [(pos != 0).mean(), 1.0]

# per-trade (per COPOM window) results
trades = []
for i in range(len(copom) - 1):
    a, b, s = copom.loc[i, "entry"], copom.loc[i + 1, "entry"], copom.loc[i, "signal"]
    window = spread.loc[(spread.index > a) & (spread.index <= b)]
    if s != 0 and len(window):
        trades.append({"date": copom.loc[i, "date"], "signal": s,
                       "ret": (1 + s * window).prod() - 1})
trades = pd.DataFrame(trades)
trades.to_csv(DATA / "backtest_trades.csv", index=False)

pd.options.display.float_format = "{:.3f}".format
print(f"Period: {pos.index[0].date()} -> {pos.index[-1].date()}")
print(tab)
print(f"\nTrades: {len(trades)}  hit rate {(trades.ret > 0).mean():.1%}  "
      f"avg {trades.ret.mean():+.2%}  "
      f"long {(trades.signal == 1).sum()} (avg {trades[trades.signal==1].ret.mean():+.2%})  "
      f"short {(trades.signal == -1).sum()} (avg {trades[trades.signal==-1].ret.mean():+.2%})")
(pd.DataFrame({"position": pos, "spread_ret": spread, "strategy_ret": strat})
   .assign(equity=curves["Strategy"]).to_csv(DATA / "backtest_daily.csv"))

# --- plot --------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, GRAY = "#2a78d6", "#9b9a95"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
})
fig, (ax, axd) = plt.subplots(2, 1, figsize=(10, 7.4), dpi=200, sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
for a in (ax, axd):
    a.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

# shade position regimes
starts = pos.index[pos.ne(pos.shift())]
ends = list(starts[1:]) + [pos.index[-1]]
for s0, s1 in zip(starts, ends):
    p = pos.loc[s0]
    if p != 0:
        for a in (ax, axd):
            a.axvspan(s0, s1, color=BLUE if p > 0 else "#eb6834", alpha=0.07, lw=0)

cols = {"Strategy": BLUE, "Always long SMLL / short IBOV": GRAY}
for name, c in cols.items():
    ax.plot(curves.index, curves[name], color=c, lw=2, label=name)
    ax.annotate(f"{curves[name].iloc[-1]:.2f}×", (curves.index[-1], curves[name].iloc[-1]),
                xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK2, va="center")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_yticks([0.4, 0.5, 0.7, 1, 1.5, 2])
ax.axhline(1, color=MUTED, lw=1)
ax.set_ylabel("Growth of 1 (log scale)")
from matplotlib.patches import Patch
handles, labels = ax.get_legend_handles_labels()
handles += [Patch(color=BLUE, alpha=0.15), Patch(color="#eb6834", alpha=0.15)]
labels += ["Long SMLL / short IBOV (after cut)", "Short SMLL / long IBOV (after hike)"]
ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)

for name, c in cols.items():
    dd = curves[name] / curves[name].cummax() - 1
    axd.plot(dd.index, dd * 100, color=c, lw=1.5)
axd.set_ylabel("Drawdown (%)")
axd.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))

s, b = tab.loc["Strategy"], tab.loc["Benchmark"]
fig.suptitle("COPOM signal backtest: SMLL vs. IBOV spread", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"Cut → long small caps, hike → short small caps, hold → flat. Enter at close of session after "
             f"decision; no costs.\nStrategy: CAGR {s.CAGR:+.1%}, Sharpe {s.Sharpe:.2f}, max DD {s['Max DD']:.0%}, "
             f"in market {s['Time in market']:.0%}   ·   Benchmark: CAGR {b.CAGR:+.1%}, Sharpe {b.Sharpe:.2f}, "
             f"max DD {b['Max DD']:.0%}", loc="left", fontsize=9, color=INK2, pad=10)
fig.text(0.07, 0.01, "Daily-rebalanced dollar-neutral spread, return = position × (r_SMLL − r_IBOV). "
         "Sources: B3, Banco Central do Brasil.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.08, right=0.93, top=0.86, bottom=0.07)
out = HERE / "backtest_equity.png"
fig.savefig(out)
print("saved", out)
