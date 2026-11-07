"""
Correlation over time between the Focus GDP-revision strategy (4 tranches x 21d)
and the COPOM linear strategy, both on the SMLL - IBOV spread.

Two correlations, rolling 2-year (504 sessions) windows on daily data:
  - positions: do the two signals point the same way?
  - daily strategy returns: what matters for diversification when combining them.
Equal-risk combined Sharpe for two strategies with Sharpes S1, S2 and return
correlation rho: (S1 + S2) / sqrt(2 * (1 + rho)).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
WIN = 504

f = pd.read_csv(DATA / "focus_tranches_daily.csv", index_col=0, parse_dates=True)
c = pd.read_csv(DATA / "backtest_daily.csv", index_col=0, parse_dates=True)
df = pd.DataFrame({"focus_pos": f["4 tranches x 21d"], "copom_pos": c["position"],
                   "focus_ret": f["4 tranches x 21d_ret"], "copom_ret": c["strategy_ret"]}).dropna()

roll_pos = df["focus_pos"].rolling(WIN).corr(df["copom_pos"])
roll_ret = df["focus_ret"].rolling(WIN).corr(df["copom_ret"])
mid = len(df) // 2
full = {"positions": df["focus_pos"].corr(df["copom_pos"]), "returns": df["focus_ret"].corr(df["copom_ret"])}
halves = {nm: {"positions": p["focus_pos"].corr(p["copom_pos"]), "returns": p["focus_ret"].corr(p["copom_ret"])}
          for nm, p in [("first half", df.iloc[:mid]), ("second half", df.iloc[mid:])]}
# agreement: share of days both are non-zero and on the same side
both = (df["focus_pos"] != 0) & (df["copom_pos"] != 0)
agree = (np.sign(df["focus_pos"]) == np.sign(df["copom_pos"]))[both].mean()


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


s1, s2 = sharpe(df["focus_ret"]), sharpe(df["copom_ret"])
comb = (s1 + s2) / np.sqrt(2 * (1 + full["returns"]))
out = pd.DataFrame({"rolling_corr_positions_2y": roll_pos, "rolling_corr_returns_2y": roll_ret})
out.to_csv(DATA / "focus_copom_rolling_corr.csv")
print(f"{df.index[0].date()} -> {df.index[-1].date()}")
print(f"Full sample: corr positions {full['positions']:+.3f}, corr daily returns {full['returns']:+.3f}")
for k, v in halves.items():
    print(f"  {k}: positions {v['positions']:+.3f}, returns {v['returns']:+.3f}")
print(f"Same side when both active: {agree:.0%} of {both.mean():.0%} of days with both positions on")
print(f"Rolling 2y return corr: min {roll_ret.min():+.2f}, median {roll_ret.median():+.2f}, max {roll_ret.max():+.2f}")
print(f"Rolling 2y position corr: min {roll_pos.min():+.2f}, median {roll_pos.median():+.2f}, max {roll_pos.max():+.2f}")
print(f"Sharpes: focus {s1:.2f}, copom {s2:.2f}; equal-risk combined at full-sample rho: {comb:.2f}")

# --- plot ---
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, AQUA, VIOLET = "#2a78d6", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (axp, axc) = plt.subplots(2, 1, figsize=(10, 7.4), dpi=200, sharex=True,
                               gridspec_kw={"height_ratios": [1.2, 1.6], "hspace": 0.12})
for a in (axp, axc):
    a.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
axp.plot(df.index, df["copom_pos"], color=BLUE, lw=1.3, drawstyle="steps-post", label="COPOM linear")
axp.plot(df.index, df["focus_pos"], color=AQUA, lw=1.3, drawstyle="steps-post", label="Focus GDP revisions (4 tranches)")
axp.axhline(0, color=MUTED, lw=1)
axp.set_ylim(-1.05, 1.05)
axp.set_ylabel("Position\n(+ = long SMLL / short IBOV)")
axp.legend(loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9, labelcolor=INK2, ncol=2)
axp.set_ylim(-1.05, 1.3)
axp.set_yticks([-1, -0.5, 0, 0.5, 1])

axc.plot(roll_pos.index, roll_pos, color=VIOLET, lw=2, label="Positions")
axc.plot(roll_ret.index, roll_ret, color=INK, lw=2, label="Daily strategy returns")
axc.axhline(0, color=MUTED, lw=1)
axc.axhline(full["returns"], color=INK, lw=1, ls=":")
axc.annotate(f"full-sample return corr {full['returns']:+.2f}", (pd.Timestamp("2018-01-01"), full["returns"]), xytext=(4, 4),
             textcoords="offset points", fontsize=8, color=INK2)
axc.set_ylim(-1, 1)
axc.set_ylabel("Rolling 2-year correlation")
axc.set_yticks(np.arange(-1, 1.01, 0.25))
axc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}" if abs(v) > 1e-9 else "0"))
axc.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2, ncol=2)

fig.suptitle("Focus GDP revisions vs. COPOM signal: correlation over time", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
axp.set_title(f"Full sample: position corr {full['positions']:+.2f}, daily return corr {full['returns']:+.2f} "
              f"(1st half {halves['first half']['returns']:+.2f}, 2nd half {halves['second half']['returns']:+.2f}). "
              f"Same side {agree:.0%} of the time both are active.\nEqual-risk combination at that correlation: "
              f"({s1:.2f} + {s2:.2f}) / √(2 × (1 + {full['returns']:.2f})) = Sharpe {comb:.2f}.",
              loc="left", fontsize=8.5, color=INK2, pad=10)
fig.text(0.07, 0.01, "Rolling windows of 504 sessions (~2 years), daily data. No costs. Sources: BCB Focus, BCB COPOM, B3.",
         fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.11, right=0.97, top=0.87, bottom=0.07)
fig.savefig(HERE / "focus_copom_corr.png")
print("saved")
