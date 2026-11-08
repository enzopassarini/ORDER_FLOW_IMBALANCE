"""
Long-only SMLL / IBOV allocation with trading costs and realistic rebalancing.

Target SMLL weight w*_t = (q_t + 1) / 2, q_t = final core signal known before day t.
Holdings drift with returns between rebalances. On a rebalance day the portfolio is reset to
the target at the start of the day (i.e. traded at the previous close, when q_t is known).
Cost: 5 bp on traded notional; moving the SMLL weight by dw means selling dw of one index and
buying dw of the other, so cost = 0.0005 * 2 * |dw| of portfolio value.
Rebalance frequencies: daily (reference), weekly (first session of each week), monthly (first
session of each month). The 50/50 benchmark is rebalanced at the same frequency, same costs.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
COST = 0.0005
fc = pd.read_csv(DATA / "final_core_daily.csv", index_col=0, parse_dates=True)
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
rr = px.pct_change().loc[fc.index]
rS, rI = rr["SMLL"].to_numpy(), rr["IBOV"].to_numpy()
target = ((fc["filtered_position"] + 1) / 2).to_numpy()
dates = fc.index


def rebal_days(freq):
    if freq == "daily":
        return np.ones(len(dates), bool)
    key = dates.to_period("W-FRI" if freq == "weekly" else "M")
    return np.r_[True, key[1:] != key[:-1]]


def simulate(tgt, freq, cost=COST, band=0.0):
    """band: on a rebalance day, trade (back to the full target) only if |target - current| > band."""
    reb = rebal_days(freq)
    w = tgt[0]
    gross, net, wts, traded = [], [], [], []
    for t in range(len(dates)):
        tc = 0.0
        if reb[t] and abs(tgt[t] - w) > band:
            dw = abs(tgt[t] - w)
            tc = cost * 2 * dw
            traded.append(2 * dw)
            w = tgt[t]
        else:
            traded.append(0.0)
        wts.append(w)
        R = w * rS[t] + (1 - w) * rI[t]
        gross.append(R)
        net.append((1 - tc) * (1 + R) - 1)
        w = w * (1 + rS[t]) / (1 + R)  # drift
    return pd.DataFrame({"gross": gross, "net": net, "w": wts, "traded": traded}, index=dates)


res = {}
for f in ("daily", "weekly", "monthly"):
    res[(f, "Portfolio")] = simulate(target, f)
    res[(f, "Benchmark")] = simulate(np.full(len(dates), 0.5), f)


def summary(p, b, col):
    yrs = len(p) / 252
    a = p[col] - b[col]
    m = len(a) // 2
    ir = lambda x: x.mean() / x.std() * np.sqrt(252)
    cg = (1 + p[col]).prod() ** (1 / yrs) - 1
    cb = (1 + b[col]).prod() ** (1 / yrs) - 1
    return {"CAGR": cg, "Benchmark CAGR": cb, "Excess CAGR": cg - cb, "Tracking error": a.std() * np.sqrt(252),
            "IR": ir(a), "IR H1": ir(a.iloc[:m]), "IR H2": ir(a.iloc[m:]),
            "Turnover/yr (traded notional)": p["traded"].sum() / yrs,
            "Cost drag/yr": ((1 + p["gross"]).prod() ** (1 / yrs) - 1) - ((1 + p["net"]).prod() ** (1 / yrs) - 1),
            "MaxDD": ((1 + p[col]).cumprod() / (1 + p[col]).cumprod().cummax() - 1).min()}


rows = []
for f in ("daily", "weekly", "monthly"):
    for col in ("gross", "net"):
        rows.append({"rebalance": f, "costs": "none" if col == "gross" else "5 bp",
                     **summary(res[(f, "Portfolio")], res[(f, "Benchmark")], col)})
tab = pd.DataFrame(rows)
tab.to_csv(DATA / "long_only_costs.csv", index=False, float_format="%.5f")
pd.options.display.width = 220
print(f"{dates[0].date()} -> {dates[-1].date()}")
print(tab.round(4).to_string(index=False))
for f in ("daily", "weekly", "monthly"):
    p = res[(f, "Portfolio")]
    print(f"{f}: avg |w - target| {np.mean(np.abs(p['w'].to_numpy() - target)):.3f}, "
          f"rebalances/yr {(p['traded'] > 0).sum() / (len(p) / 252):.0f}")

# --- plot ------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COL = {"daily": "#9b9a95", "weekly": "#0b0b0b", "monthly": "#2a78d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=200, gridspec_kw={"width_ratios": [2.2, 1]})
for a in (ax1, ax2):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
for f, c in COL.items():
    p, b = res[(f, "Portfolio")], res[(f, "Benchmark")]
    rel = ((1 + p["net"]).cumprod() / (1 + b["net"]).cumprod() - 1) * 100
    s = tab[(tab.rebalance == f) & (tab.costs == "5 bp")].iloc[0]
    ax1.plot(rel.index, rel, color=c, lw=2 if f == "weekly" else 1.4,
             label=f"{f}: {s['Excess CAGR']:+.2%}/yr, IR {s.IR:.2f}, turnover {s['Turnover/yr (traded notional)']:.1f}×/yr")
    ax1.annotate(f"{rel.iloc[-1]:+.0f}%", (rel.index[-1], rel.iloc[-1]), xytext=(6, {"daily": -9, "weekly": 4, "monthly": 0}[f]), textcoords="offset points",
                 fontsize=9, color=c, va="center")
ax1.axhline(0, color=MUTED, lw=1)
ax1.set_ylabel("Relative to 50/50, after costs (%)")
ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
ax1.set_title("Cumulative alpha vs. 50/50 after 5 bp costs", loc="left", fontsize=10, color=INK2)
x = np.arange(3)
g = tab[tab.costs == "none"].set_index("rebalance").loc[list(COL)]
n = tab[tab.costs == "5 bp"].set_index("rebalance").loc[list(COL)]
ax2.bar(x - 0.18, g["Excess CAGR"] * 100, width=0.36, color="#c9c8c2", label="before costs", zorder=3)
ax2.bar(x + 0.18, n["Excess CAGR"] * 100, width=0.36, color=[COL[f] for f in COL], label="after 5 bp", zorder=3)
for i, f in enumerate(COL):
    ax2.text(i + 0.18, n.loc[f, "Excess CAGR"] * 100 + 0.03, f"{n.loc[f, 'Excess CAGR']:+.2%}", ha="center",
             fontsize=7.5)
    ax2.text(i - 0.18, g.loc[f, "Excess CAGR"] * 100 + 0.03, f"{g.loc[f, 'Excess CAGR']:+.2%}", ha="center",
             fontsize=7.5, color=INK2)
ax2.set_xticks(x)
ax2.set_xticklabels(list(COL))
ax2.set_ylabel("Excess return vs. 50/50 (%/yr)")
ax2.set_ylim(0, max(g["Excess CAGR"].max(), n["Excess CAGR"].max()) * 100 * 1.3)
from matplotlib.patches import Patch
ax2.legend([Patch(color="#c9c8c2"), Patch(color=INK)], ["before costs", "after 5 bp"], loc="upper right",
           frameon=False, fontsize=9, labelcolor=INK2)
ax2.set_title("Excess return by rebalance frequency", loc="left", fontsize=10, color=INK2)
fig.suptitle("Long-only SMLL / IBOV allocation: weekly vs. monthly rebalancing with 5 bp costs", x=0.05, ha="left",
             fontsize=13, fontweight="bold", y=0.98)
fig.text(0.05, 0.905, "SMLL weight target = (signal + 1) / 2; weights drift between rebalances; cost = 5 bp × traded "
         f"notional (2 × |Δw|); benchmark rebalanced the same way. {dates[0]:%b %Y} – {dates[-1]:%b %Y}.",
         fontsize=9, color=INK2)
fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08, wspace=0.25)
fig.savefig(HERE / "long_only_costs.png")
print("saved")
