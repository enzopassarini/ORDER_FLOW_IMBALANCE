"""
No-trade band for the weekly long-only SMLL / IBOV allocation (5 bp costs).

Each week (first session) the SMLL weight is reset to the target (signal + 1) / 2 only if it is
more than `band` away from it; otherwise the portfolio is left to drift. Bands of 0-10 pp
are compared; the 50/50 benchmark is rebalanced weekly with the same band and costs.
Uses simulate() from long_only_costs.py.
"""
import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import long_only_costs as lc  # runs the base comparison silently, reuses its data and simulate()

HERE, DATA = lc.HERE, lc.DATA
BANDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
rows, sims = [], {}
for b in BANDS:
    p = lc.simulate(lc.target, "weekly", band=b)
    bm = lc.simulate(np.full(len(lc.dates), 0.5), "weekly", band=b)
    sims[b] = (p, bm)
    s_net = lc.summary(p, bm, "net")
    s_gross = lc.summary(p, bm, "gross")
    yrs = len(p) / 252
    rows.append({"band (pp)": b * 100, "Excess before costs": s_gross["Excess CAGR"], **s_net,
                 "Trades/yr": (p["traded"] > 0).sum() / yrs,
                 "Avg |w - target|": np.mean(np.abs(p["w"].to_numpy() - lc.target))})
tab = pd.DataFrame(rows)
tab.to_csv(DATA / "long_only_band.csv", index=False, float_format="%.5f")
pd.options.display.width = 220
print(tab[["band (pp)", "Excess before costs", "Excess CAGR", "IR", "IR H1", "IR H2", "Turnover/yr (traded notional)",
           "Cost drag/yr", "Trades/yr", "Avg |w - target|"]].round(4).to_string(index=False))
mrow = lc.tab[(lc.tab.rebalance == "monthly") & (lc.tab.costs == "5 bp")].iloc[0]
print(f"monthly, no band: excess {mrow['Excess CAGR']:.4f}, IR {mrow.IR:.3f}, turnover {mrow['Turnover/yr (traded notional)']:.2f}")

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CH = 0.03
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=200, gridspec_kw={"width_ratios": [1.2, 1.6]})
for a in (ax1, ax2):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
bx = tab["band (pp)"]
ax1.plot(bx, tab["Excess CAGR"] * 100, marker="o", color=INK, lw=2, label="excess vs 50/50 after costs (%/yr)")
ax1.plot(bx, tab["Excess before costs"] * 100, marker="o", color=MUTED, lw=1.2, ls="--", label="before costs")
ax1.axhline(mrow["Excess CAGR"] * 100, color=BLUE, lw=1, ls=":")
ax1.text(bx.max(), mrow["Excess CAGR"] * 100, " monthly,\n no band", color=BLUE, fontsize=8, va="center")
ax1.set_xlabel("No-trade band (percentage points of SMLL weight)")
ax1.set_ylabel("Excess return vs. 50/50 (%/yr)")
axt = ax1.twinx()
axt.bar(bx, tab["Turnover/yr (traded notional)"], width=0.6, color=ORANGE, alpha=0.3, zorder=1)
axt.set_ylabel("Turnover per year (× portfolio)", color=ORANGE)
axt.tick_params(axis="y", colors=ORANGE)
axt.spines["top"].set_visible(False)
ax1.set_zorder(axt.get_zorder() + 1)
ax1.patch.set_visible(False)
ax1.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)
ax1.set_title("Weekly rebalancing: excess return and turnover by band", loc="left", fontsize=10, color=INK2)
for b, c, lw in ((0.0, MUTED, 1.4), (CH, INK, 2.1), (0.10, ORANGE, 1.2)):
    p, bm = sims[b]
    rel = ((1 + p["net"]).cumprod() / (1 + bm["net"]).cumprod() - 1) * 100
    r_ = tab[tab["band (pp)"] == b * 100].iloc[0]
    ax2.plot(rel.index, rel, color=c, lw=lw, label=f"band {b * 100:.0f} pp: {r_['Excess CAGR']:+.2%}/yr, IR {r_.IR:.2f}, "
                                                     f"turnover {r_['Turnover/yr (traded notional)']:.2f}×/yr, "
                                                     f"{r_['Trades/yr']:.0f} trades/yr")
ax2.axhline(0, color=MUTED, lw=1)
ax2.set_ylabel("Relative to 50/50, after costs (%)")
ax2.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
ax2.set_title("Cumulative alpha vs. 50/50 after 5 bp costs", loc="left", fontsize=10, color=INK2)
fig.suptitle("No-trade band on the weekly long-only allocation", x=0.05, ha="left", fontsize=13, fontweight="bold", y=0.98)
fig.text(0.05, 0.905, "Each week, trade back to the target SMLL weight (signal + 1) / 2 only if the current weight is "
         "more than the band away from it. 5 bp per unit traded; benchmark 50/50 treated the same way.",
         fontsize=9, color=INK2)
fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.1, wspace=0.32)
fig.savefig(HERE / "long_only_band.png")
print("saved")
