"""
Decile analysis: bounded COPOM signal vs. forward 21-session SMLL - IBOV return.

The signal is discrete (9 values, many holds at 0), so decile edges fall inside
groups of tied values. Ties are broken by decision date (pd.qcut on a
first-occurrence rank); each decile is labelled with the signal range it holds.
Reads data/signal_fwd21.csv written by signal_vs_fwd21.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ev = pd.read_csv(HERE / "data" / "signal_fwd21.csv", parse_dates=["date"]).sort_values("date")

ev["decile"] = pd.qcut(ev["signal"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
dec = ev.groupby("decile").agg(n=("fwd21_rel", "size"), lo=("signal", "min"), hi=("signal", "max"),
                               sig=("signal", "mean"), mean=("fwd21_rel", "mean"),
                               std=("fwd21_rel", "std"), hit=("fwd21_rel", lambda s: (s > 0).mean()))
dec["mean"] *= 100
dec["se"] = dec["std"] * 100 / np.sqrt(dec["n"])
dec.drop(columns="std").round(3).to_csv(HERE / "data" / "signal_deciles.csv")
spread = dec.loc[10, "mean"] - dec.loc[1, "mean"]
print(dec.drop(columns="std").round(2))
print(f"D10 - D1 = {spread:+.2f} pp")

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b5b4ae"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
})
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
ax.axhline(0, color=MUTED, lw=1, zorder=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

colors = [BLUE if m > 0 else ORANGE if m < 0 else GRAY for m in dec["sig"]]
ax.bar(dec.index, dec["mean"], width=0.72, color=colors, zorder=3)
ax.errorbar(dec.index, dec["mean"], yerr=dec["se"], fmt="none", ecolor=INK2, elinewidth=1,
            capsize=3, zorder=4)
for d, r in dec.iterrows():
    off = r["se"] + 0.12
    ax.text(d, r["mean"] + (off if r["mean"] >= 0 else -off), f"{r['mean']:+.2f}",
            ha="center", va="bottom" if r["mean"] >= 0 else "top", fontsize=8.5, color=INK)

fmt = lambda v: f"{v:+.2f}" if v else "0"
ax.set_xticks(dec.index)
ax.set_xticklabels([f"D{d}\n{fmt(r.lo)}" + ("" if r.lo == r.hi else f"…{fmt(r.hi)}") + f"\nn={int(r.n)}"
                    for d, r in dec.iterrows()], fontsize=8)
lo_y = (dec["mean"] - dec["se"]).min() - 0.5
hi_y = (dec["mean"] + dec["se"]).max() + 0.5
ax.set_ylim(lo_y, hi_y)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}" if v else "0"))
ax.set_ylabel("Mean SMLL − IBOV fwd 21-session return (pp)")
ax.set_xlabel("Signal decile (signal range, count)")

from matplotlib.patches import Patch
ax.legend([Patch(color=ORANGE), Patch(color=GRAY), Patch(color=BLUE)],
          ["Hike (short small caps)", "Hold (signal 0)", "Cut (long small caps)"],
          loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)

fig.suptitle("COPOM signal deciles vs. next-month small-cap relative return", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"{len(ev)} decisions, {ev.date.min():%b %Y} – {ev.date.max():%b %Y}. Error bars ±1 standard error. "
             f"D10 − D1 = {spread:+.2f} pp.\nSignal has only 9 distinct values: ties at decile edges are split "
             f"by decision date, so D4–D6 are all holds and differ only by period.",
             loc="left", fontsize=8.5, color=INK2, pad=10)
fig.text(0.07, 0.01, "Signal = clip(−ΔSelic / 100 bp, −1, 1). Window: close of first session after decision → "
         "21 sessions later. Sources: B3, Banco Central do Brasil.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = HERE / "signal_deciles.png"
fig.savefig(out)
print("saved", out)
