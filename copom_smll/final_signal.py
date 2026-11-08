"""
The final aggregate long-short signal: COPOM + GDP revisions (tranches) + real rate (tranches),
equal-risk weights, halved when its sign disagrees with the SMLL-IBOV spread's 63-session return.

Outputs:
  data/final_signal_fwd21.csv  weekly samples (signal as held from the Monday close) with the
                               forward 21-session SMLL-IBOV return, for focus_channel_plot.py
  final_signal_timeseries.png  the signal over time with its components
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
H = 21
agg = pd.read_csv(DATA / "aggregate3_daily.csv", index_col=0, parse_dates=True)
fc = pd.read_csv(DATA / "final_core_daily.csv", index_col=0, parse_dates=True)
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
q = fc["filtered_position"]

rows = []
for d in pd.date_range(q.index[0], q.index[-1], freq="W-FRI"):
    i0 = px.index.searchsorted(d, side="right")
    if i0 + H >= len(px) or px.index[i0 + 1] not in q.index:
        continue
    r = px.iloc[i0 + H] / px.iloc[i0] - 1
    rows.append({"date": d, "Final signal": q.loc[px.index[i0 + 1]], "fwd21": r["SMLL"] - r["IBOV"]})
pd.DataFrame(rows).set_index("date").to_csv(DATA / "final_signal_fwd21.csv")

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COL = {"COPOM": "#2a78d6", "GDP (tranches)": "#1baf7a", "Real rate (tranches)": "#eb6834"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), dpi=200, sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.15})
for a in (ax1, ax2):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    a.axhline(0, color=MUTED, lw=1)
    a.set_ylim(-1.05, 1.05)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
for k, c in COL.items():
    ax1.plot(agg.index, agg[k], color=c, lw=0.9, alpha=0.8, drawstyle="steps-post", label=k.replace(" (tranches)", ""))
ax1.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3)
ax1.set_ylabel("Component position")
ax1.set_title("Components (+ = long SMLL / short IBOV)", loc="left", fontsize=10, color=INK2)
ax2.fill_between(q.index, 0, q.clip(lower=0), color="#1baf7a", alpha=0.7, lw=0, step="post", label="long small caps")
ax2.fill_between(q.index, 0, q.clip(upper=0), color="#eb6834", alpha=0.7, lw=0, step="post", label="short small caps")
ax2.plot(agg.index, agg["Aggregate (3)"], color=INK, lw=0.8, alpha=0.6, drawstyle="steps-post",
         label="before trend filter")
ax2.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3)
ax2.set_ylabel("Final signal")
ax2.set_title(f"Final signal: equal-risk aggregate, halved against the 3-month trend. Long {(q > 0).mean():.0%} / "
              f"short {(q < 0).mean():.0%} of days, average |signal| {q.abs().mean():.2f}", loc="left", fontsize=10,
              color=INK2)
fig.suptitle("Aggregate long-short signal: SMLL vs. Ibovespa", x=0.06, ha="left", fontsize=13, fontweight="bold", y=0.975)
fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.05)
fig.savefig(HERE / "final_signal_timeseries.png")
print("saved")
