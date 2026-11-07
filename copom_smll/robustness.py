"""
Robustness of the COPOM signal at the 21-session horizon.

Subsets:
  Full sample
  Ex-outliers: drop the Sep 2008 (+75 bp) and Jun 2009 (-100 bp) decisions
  First half / second half: first 85 vs. last 85 decisions (split ~Jan 2016)

For each subset: signal vs. fwd-21 relative return stats, and the Sharpe of the
linear backtest (position = -signal, held decision to decision) over the same
decisions / period. Needs data/signal_fwd21.csv (run signal_vs_fwd.py 21) and
data/backtest_daily.csv (run backtest.py).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

ev = pd.read_csv(DATA / "signal_fwd21.csv", parse_dates=["date", "entry", "exit"]).sort_values("date")
ev = ev.reset_index(drop=True)
bt = pd.read_csv(DATA / "backtest_daily.csv", index_col=0, parse_dates=True)
copom = pd.read_csv(DATA / "copom_decisions.csv", parse_dates=["date"])

OUTLIERS = ev["date"].dt.strftime("%Y-%m").isin(["2008-09", "2009-06"])
half = len(ev) // 2
split_date = ev.loc[half, "date"]

subsets = {
    "Full sample": ev.index == ev.index,
    "Ex Sep 2008 & Jun 2009": ~OUTLIERS,
    f"First half ({ev.date.iloc[0]:%b %Y} – {ev.date.iloc[half-1]:%b %Y})": ev.index < half,
    f"Second half ({split_date:%b %Y} – {ev.date.iloc[-1]:%b %Y})": ev.index >= half,
}

# backtest days belonging to each excluded decision: from its entry until the next decision's entry
dec = copom[copom["date"] >= ev["date"].min()].sort_values("date").reset_index(drop=True)
def window_days(d):
    nxt = dec.loc[dec["date"] > d, "date"]
    a = bt.index[bt.index > d][0]                       # first session after decision = entry
    b = bt.index[bt.index > nxt.iloc[0]][0] if len(nxt) else bt.index[-1]
    return (bt.index > a) & (bt.index <= b)

out_days = np.zeros(len(bt), bool)
for d in ev.loc[OUTLIERS, "date"]:
    out_days |= window_days(d)
bt_split = bt.index[bt.index > split_date][0]
bt_masks = [np.ones(len(bt), bool), ~out_days, bt.index <= bt_split, bt.index > bt_split]


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


rows = []
for (name, m), bm in zip(subsets.items(), bt_masks):
    e = ev[m]
    x, y = e["signal"].to_numpy(), e["fwd_rel"].to_numpy() * 100
    n = len(e)
    slope = np.polyfit(x, y, 1)[0]
    c = np.corrcoef(x, y)[0, 1]
    nz = x != 0
    r = bt.loc[bm, "strategy_ret"]
    rows.append({"subset": name, "n": n, "slope_pp": slope, "corr": c,
                 "t": c * np.sqrt((n - 2) / (1 - c ** 2)),
                 "hit": (np.sign(x[nz]) == -np.sign(y[nz])).mean(),
                 "mean_cut": y[x < 0].mean(), "mean_hike": y[x > 0].mean(),
                 "bt_sharpe": sharpe(r), "bt_cagr": (1 + r).prod() ** (252 / len(r)) - 1})
tab = pd.DataFrame(rows).set_index("subset")
tab.round(3).to_csv(DATA / "robustness_fwd21.csv")
pd.options.display.width = 200
print(tab.round(3))

# --- plot: small multiples, shared axes -------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE = "#2a78d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
})
fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200, sharex=True, sharey=True)
rng = np.random.default_rng(0)
for ax, ((name, m), (_, st)) in zip(axes.flat, zip(subsets.items(), tab.iterrows())):
    e = ev[m]
    x, y = e["signal"].to_numpy(), e["fwd_rel"].to_numpy() * 100
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.scatter(x + rng.uniform(-0.03, 0.03, len(x)), y, s=22, color=BLUE, alpha=0.7,
               edgecolor=SURF, lw=0.8, zorder=3)
    g = e.groupby("signal")["fwd_rel"].mean() * 100
    ax.scatter(g.index, g.values, s=70, marker="_", color=INK, lw=2.2, zorder=4)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(-1, 1, 20)
    ax.plot(xs, b * xs + a, color=INK, lw=1.8, zorder=4)
    ax.set_title(f"{name}\nn={int(st.n)}  slope {st.slope_pp:+.2f} pp  ρ {st['corr']:+.2f}  t {st.t:+.2f}  "
                 f"hit {st.hit:.0%}  ·  backtest Sharpe {st.bt_sharpe:.2f}",
                 loc="left", fontsize=8.5, color=INK2)
for ax in axes[1]:
    ax.set_xlabel("Signal = clip(ΔSelic, ±100 bp) / 100 bp   ← cut · hike →")
for ax in axes[:, 0]:
    ax.set_ylabel("SMLL − IBOV fwd 21-session return (pp)")
axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
fig.suptitle("Robustness: COPOM signal vs. next 21-session small-cap relative return", x=0.06, ha="left",
             fontsize=13, fontweight="bold")
fig.text(0.06, 0.935, "Shared axes. Black dashes = mean at each signal level; line = OLS fit. "
         "Backtest Sharpe = linear strategy (position = −signal) over the same decisions, no costs.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = HERE / "robustness_fwd21.png"
fig.savefig(out)
print("saved", out)
