"""
Selic-target decision delta vs. SMLL-minus-IBOV return until the next COPOM.

Window: COPOM decisions are announced after the B3 close, so they are only
tradable on the next session. The window is shifted by one trading day: it runs
from the close of the first session after decision k to the close of the first
session after decision k+1.
Relative return = SMLL return - IBOV return over that window (simple returns).
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


def close_next_session(d):
    """Close of the first trading session strictly after date d."""
    return px.loc[px.index > d].iloc[0]


rows = []
for i in range(len(copom) - 1):
    d0, d1 = copom.loc[i, "date"], copom.loc[i + 1, "date"]
    if (px.index > d1).sum() == 0:
        continue  # next decision not yet followed by a session
    p0, p1 = close_next_session(d0), close_next_session(d1)
    r = p1 / p0 - 1
    rows.append({
        "meeting": copom.loc[i, "meeting"], "date": d0, "next_date": d1,
        "extraordinary": copom.loc[i, "extraordinary"],
        "selic_meta": copom.loc[i, "selic_meta"], "delta_bp": copom.loc[i, "delta_bp"],
        "ret_smll": r["SMLL"], "ret_ibov": r["IBOV"], "rel_ret": r["SMLL"] - r["IBOV"],
    })
ev = pd.DataFrame(rows)
ev.to_csv(DATA / "copom_events.csv", index=False)

# --- stats --------------------------------------------------------------
x, y = ev["delta_bp"].to_numpy(), ev["rel_ret"].to_numpy() * 100
slope, intercept = np.polyfit(x, y, 1)
corr = np.corrcoef(x, y)[0, 1]
n = len(ev)
t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2))
bucket = pd.cut(ev["delta_bp"], [-1e9, -0.5, 0.5, 1e9], labels=["Cut", "Hold", "Hike"])
summary = ev.groupby(bucket, observed=True)["rel_ret"].agg(["count", "mean", "median"])
summary[["mean", "median"]] *= 100
print(summary.round(2))
print(f"n={n}  slope={slope:.4f} pp per bp  corr={corr:.3f}  t={t_stat:.2f}")

# --- plot ---------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

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

rng = np.random.default_rng(0)
jit = rng.uniform(-4, 4, n)  # small horizontal jitter: deltas are discrete
reg = ~ev["extraordinary"].to_numpy()
ax.scatter(x[reg] + jit[reg], y[reg], s=36, color=BLUE, alpha=0.75,
           edgecolor=SURF, lw=1, zorder=3, label="Regular meeting")
if (~reg).any():
    ax.scatter(x[~reg] + jit[~reg], y[~reg], s=48, color=ORANGE, marker="D",
               edgecolor=SURF, lw=1, zorder=3, label="Extraordinary meeting")

xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, slope * xs + intercept, color=INK, lw=2, zorder=4,
        label=f"OLS fit: {slope*100:+.2f} pp per 100 bp  (ρ = {corr:.2f}, t = {t_stat:.2f})")

# label the extremes
for _, r in ev.loc[ev["rel_ret"].abs().nlargest(4).index].iterrows():
    ax.annotate(r["date"].strftime("%b %Y"), (r["delta_bp"], r["rel_ret"] * 100),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")

ax.set_xlabel("Selic target change at COPOM decision (bp)")
ax.set_ylabel("SMLL − IBOV return until next COPOM (pp)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))

txt = "   ".join(f"{k}: n={int(r['count'])}, mean {r['mean']:+.2f} pp" for k, r in summary.iterrows())
fig.suptitle("Selic decisions vs. small-cap relative performance", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"Each dot is one COPOM decision, {ev['date'].min():%b %Y} – {ev['date'].max():%b %Y}. "
             f"Window: close of session after decision → close of session after next decision.\n{txt}",
             loc="left", fontsize=9, color=INK2, pad=10)
ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
fig.text(0.07, 0.01, "Sources: B3 (IBOV, SMLL daily closes); Banco Central do Brasil (COPOM history).",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = HERE / "selic_delta_vs_smll_ibov.png"
fig.savefig(out)
print("saved", out)
