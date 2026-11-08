"""
One-sided version of the long-short: long SMLL / short IBOV when the final core signal is
positive, flat otherwise (never short SMLL). Compared with the two-sided strategy.

  unlevered one-sided:     max(q_t, 0)
  vol-targeted one-sided:  max(L_t * q_t, 0), L_t = the two-sided strategy's leverage
                           (10% / EWMA 63d P&L vol, cap 5x), so each unit of signal is sized
                           exactly as in the two-sided version
Weekly rebalancing with the 0.02 no-trade band, 5 bp per unit traded on each leg, no borrow or
financing (the only short is IBOV, which can be done with Ibovespa futures).
Uses simulate() / stats() from ls_costs.py.
"""
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import ls_costs as lc

HERE, DATA = lc.HERE, lc.DATA
two_u, two_v = lc.TARGETS["Unlevered"], lc.TARGETS["Vol-targeted 10%"]
TG = {"Two-sided, unlevered": two_u, "Long-small only, unlevered": np.clip(two_u, 0, None),
      "Two-sided, vol-targeted": two_v, "Long-small only, vol-targeted": np.clip(two_v, 0, None)}
sims = {k: lc.simulate(t, "weekly", band=0.02) for k, t in TG.items()}
rows = []
for k, df in sims.items():
    s = lc.stats(df, "net")
    yrs = len(df) / 252
    tg = pd.Series(TG[k], index=df.index)
    rows.append({"strategy": k, **s, "Time long SMLL (target > 0)": (tg > 0).mean(),
                 "Avg position when long": tg[tg > 0].mean(),
                 "Worst year": ((1 + df["net"]).groupby(df.index.year).prod() - 1).min()})
tab = pd.DataFrame(rows).set_index("strategy")
tab.to_csv(DATA / "ls_long_small_only.csv", float_format="%.5f")
pd.options.display.width = 220
print(f"{lc.idx[0].date()} -> {lc.idx[-1].date()} (after 5 bp, weekly + band)")
print(tab.round(4).to_string())
yr = pd.DataFrame({k: (1 + df["net"]).groupby(df.index.year).prod() - 1 for k, df in sims.items()})
yr.to_csv(DATA / "ls_long_small_only_by_year.csv", float_format="%.4f")
print("\nby year (%):\n", (yr * 100).round(1).to_string())

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
AQUA, ORANGE = "#1baf7a", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axes = plt.subplots(3, 1, figsize=(11, 11), dpi=200, sharex=True,
                         gridspec_kw={"height_ratios": [1.6, 1.6, 0.9], "hspace": 0.2})
for a in axes:
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
for ax, lev in zip(axes[:2], ("unlevered", "vol-targeted")):
    for k, c, lw in ((f"Two-sided, {lev}", MUTED, 1.4), (f"Long-small only, {lev}", INK, 2.2)):
        eq = (1 + sims[k]["net"]).cumprod()
        t = tab.loc[k]
        ax.plot(eq.index, eq, color=c, lw=lw,
                label=f"{k.split(',')[0]}: CAGR {t.CAGR:+.1%}, vol {t.Vol:.1%}, Sharpe {t.Sharpe:.2f} "
                      f"(H1 {t['Sharpe H1']:.2f}, H2 {t['Sharpe H2']:.2f}), max DD {t.MaxDD:.0%}")
        ax.annotate(f"{eq.iloc[-1]:.2f}×", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=c, va="center")
    ax.axhline(1, color=MUTED, lw=1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title("Unlevered" if lev == "unlevered" else "Vol-targeted (same leverage as the two-sided 10% version)",
                 loc="left", fontsize=10, color=INK2)
    ax.set_ylabel("Growth of 1")
p2 = sims["Two-sided, unlevered"]["pos"]
ax = axes[2]
ax.fill_between(p2.index, 0, p2.clip(lower=0), color=AQUA, lw=0, step="post", label="kept: long SMLL / short IBOV")
ax.fill_between(p2.index, 0, p2.clip(upper=0), color=ORANGE, alpha=0.35, lw=0, step="post",
                label="dropped: short SMLL / long IBOV → flat")
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylabel("Unlevered position")
ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=2)
t = tab.loc["Long-small only, unlevered"]
ax.set_title(f"Position: long small caps {t['Time long SMLL (target > 0)']:.0%} of days, flat otherwise", loc="left", fontsize=10, color=INK2)
fig.suptitle("Long small / short Ibovespa only (flat when the signal is negative) vs. two-sided", x=0.07, ha="left",
             fontsize=13, fontweight="bold", y=0.985)
fig.text(0.07, 0.95, "After 5 bp trading costs, weekly rebalancing with a 0.02 no-trade band; no borrow or financing. "
         f"{lc.idx[0]:%b %Y} – {lc.idx[-1]:%b %Y}.", fontsize=9, color=INK2)
fig.subplots_adjust(left=0.08, right=0.93, top=0.92, bottom=0.04)
fig.savefig(HERE / "ls_long_small_only.png")
print("saved")
