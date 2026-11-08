"""
Correlation matrices of the three signals as traded in aggregate3.py
(COPOM, GDP revisions in tranches, real rate in tranches): daily positions and daily
strategy returns, full sample and halves. Reads data/aggregate3_daily.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
S = ["COPOM", "GDP (tranches)", "Real rate (tranches)"]
LAB = ["COPOM", "GDP", "Real rate"]
d = pd.read_csv(DATA / "aggregate3_daily.csv", index_col=0, parse_dates=True)
mid = len(d) // 2
parts = {"Full": d, "First half": d.iloc[:mid], "Second half": d.iloc[mid:]}
out = []
for kind, cols in (("positions", S), ("returns", [c + " ret" for c in S])):
    for nm, p in parts.items():
        c = p[cols].corr()
        c.index = c.columns = LAB
        out.append(c.stack().rename(f"{kind} {nm}"))
tab = pd.concat(out, axis=1)
tab = tab[[a != b for a, b in tab.index]].iloc[[1, 2, 5]]  # COPOM-GDP, COPOM-RR, GDP-RR
tab.to_csv(DATA / "signal_corr_matrix.csv", float_format="%.3f")
print(f"{d.index[0].date()} -> {d.index[-1].date()}, split {d.index[mid].date()}")
print(tab.round(2).to_string())

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=200)
for i, (kind, cols) in enumerate((("Daily positions", S), ("Daily strategy returns", [c + " ret" for c in S]))):
    for j, (nm, p) in enumerate(parts.items()):
        c = p[cols].corr().values
        ax = axes[i, j]
        ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{c[a, b]:.2f}", ha="center", va="center", fontsize=12,
                        color="white" if abs(c[a, b]) > 0.6 else INK, fontweight="bold" if a != b else None)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LAB)
        ax.set_yticklabels(LAB if j == 0 else [])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{kind} — {nm}\n{p.index[0]:%b %Y} – {p.index[-1]:%b %Y}", loc="left", fontsize=9, color=INK2)
fig.suptitle("Correlation of the three signals: COPOM, GDP revisions (tranches), real rate (tranches)",
             x=0.03, ha="left", fontsize=13, fontweight="bold")
fig.text(0.03, 0.925, "Positions: + = long SMLL / short IBOV. Returns: position × (r_SMLL − r_IBOV), daily, no costs.",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(HERE / "signal_corr_matrix.png")
print("saved")
