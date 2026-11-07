"""
Bounded COPOM signal vs. forward 21-session SMLL - IBOV relative return.

Signal: -clip(delta_bp, -100, 100) / 100, in [-1, 1], known after the decision-day close.
Forward return: from the close of the first session after the decision
(the first tradable close) to the close 21 sessions later.
Relative return = SMLL return - IBOV return over that window.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from copom_signal import selic_signal

HERE = Path(__file__).parent
DATA = HERE / "data"
H = 21

px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
copom = pd.read_csv(DATA / "copom_decisions.csv", parse_dates=["date"])
copom = copom[copom["date"] >= px.index[0]].reset_index(drop=True)
copom["signal"] = selic_signal(copom["delta_bp"])

rows = []
for _, r in copom.iterrows():
    i0 = px.index.searchsorted(r["date"], side="right")  # first session after decision
    if i0 + H >= len(px):
        continue  # forward window not complete yet
    p0, p1 = px.iloc[i0], px.iloc[i0 + H]
    ret = p1 / p0 - 1
    rows.append({"date": r["date"], "delta_bp": r["delta_bp"], "signal": r["signal"],
                 "entry": px.index[i0], "exit": px.index[i0 + H],
                 "fwd21_rel": ret["SMLL"] - ret["IBOV"]})
ev = pd.DataFrame(rows)
ev.to_csv(DATA / "signal_fwd21.csv", index=False)

x, y = ev["signal"].to_numpy(), ev["fwd21_rel"].to_numpy() * 100
n = len(ev)
slope, intercept = np.polyfit(x, y, 1)
corr = np.corrcoef(x, y)[0, 1]
t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2))
hit = (np.sign(x[x != 0]) == np.sign(y[x != 0])).mean()
grp = ev.groupby("signal")["fwd21_rel"].agg(["count", "mean"])
grp["mean"] *= 100
print(grp.round(2))
print(f"n={n} slope={slope:.2f} pp per unit signal  corr={corr:.3f} t={t_stat:.2f}  hit(non-zero)={hit:.1%}")

# --- plot --------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE = "#2a78d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
})
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
ax.grid(True, color=GRID, lw=0.8, zorder=0)
ax.axhline(0, color=MUTED, lw=1, zorder=1)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

jit = np.random.default_rng(0).uniform(-0.03, 0.03, n)  # signal is discrete
ax.scatter(x + jit, y, s=36, color=BLUE, alpha=0.7, edgecolor=SURF, lw=1, zorder=3,
           label="One COPOM decision")
ax.scatter(grp.index, grp["mean"], s=90, marker="_", color=INK, lw=2.5, zorder=4,
           label="Mean at each signal level")
xs = np.linspace(-1, 1, 50)
ax.plot(xs, slope * xs + intercept, color=INK, lw=2, zorder=4,
        label=f"OLS fit: {slope:+.2f} pp per unit signal  (ρ = {corr:.2f}, t = {t_stat:.2f})")

for _, r in ev.loc[ev["fwd21_rel"].abs().nlargest(4).index].iterrows():
    ax.annotate(r["date"].strftime("%b %Y"), (r["signal"], r["fwd21_rel"] * 100),
                xytext=(7, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")

ax.set_xlim(-1.15, 1.15)
ax.set_xticks(np.arange(-1, 1.01, 0.25))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}" if v else "0"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
ax.set_xlabel("Signal = −clip(ΔSelic, ±100 bp) / 100 bp    ← hike = short small caps   ·   cut = long small caps →")
ax.set_ylabel("SMLL − IBOV forward 21-session return (pp)")
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)

fig.suptitle("COPOM signal vs. next-month small-cap relative return", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"{n} decisions, {ev['date'].min():%b %Y} – {ev['date'].max():%b %Y}. Window: close of first session "
             f"after decision → 21 sessions later.\nHit rate when signal ≠ 0: {hit:.0%}",
             loc="left", fontsize=9, color=INK2, pad=10)
fig.text(0.07, 0.01, "Sources: B3 (IBOV, SMLL daily closes); Banco Central do Brasil (COPOM history).",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = HERE / "signal_vs_fwd21.png"
fig.savefig(out)
print("saved", out)
