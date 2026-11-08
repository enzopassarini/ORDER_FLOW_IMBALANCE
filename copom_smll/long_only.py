"""
Long-only, always fully invested SMLL / IBOV portfolio driven by the final core signal.

Signal q_t in [-1, 1]: COPOM + GDP + real rate (equal risk), halved when against the SMLL-IBOV
spread's 3-month trend (data/final_core_daily.csv; known before day t).
Weight in SMLL:  w_t = (q_t + 1) / 2   (q = -1 -> 0% SMLL, q = 0 -> 50%, q = +1 -> 100%)
Weight in IBOV:  1 - w_t.  Benchmark: 50% SMLL / 50% IBOV.
Active return vs. benchmark = (w_t - 0.5) * (r_SMLL - r_IBOV) = q_t / 2 * spread_t,
i.e. half of the long-short strategy's return. Daily rebalancing, no costs.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
fc = pd.read_csv(DATA / "final_core_daily.csv", index_col=0, parse_dates=True)
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
rr = px.pct_change().loc[fc.index]
rS, rI = rr["SMLL"], rr["IBOV"]
w = (fc["filtered_position"] + 1) / 2
R = pd.DataFrame({"Portfolio": w * rS + (1 - w) * rI, "Benchmark 50/50": 0.5 * rS + 0.5 * rI,
                  "IBOV": rI, "SMLL": rS})
act = R["Portfolio"] - R["Benchmark 50/50"]
pd.DataFrame({"smll_weight": w, "portfolio": R["Portfolio"], "benchmark": R["Benchmark 50/50"], "active": act}).to_csv(
    DATA / "long_only_daily.csv")


def stats(r, b):
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    a = r - b
    m = len(r) // 2
    ir = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "Vol": r.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min(),
            "Excess CAGR": eq.iloc[-1] ** (1 / yrs) - (1 + b).prod() ** (1 / yrs), "Tracking error": a.std() * np.sqrt(252),
            "IR": ir(a), "IR H1": ir(a.iloc[:m]), "IR H2": ir(a.iloc[m:]),
            "Monthly hit": ((1 + a).resample("ME").prod() - 1).gt(0).mean() if a.std() > 0 else np.nan}


tab = pd.DataFrame({k: stats(R[k], R["Benchmark 50/50"]) for k in R}).T
tab.to_csv(DATA / "long_only_stats.csv", float_format="%.4f")
yearly = (1 + act).groupby(act.index.year).prod() - 1
yearly.to_csv(DATA / "long_only_alpha_by_year.csv", float_format="%.4f")
pd.options.display.width = 200
print(f"{R.index[0].date()} -> {R.index[-1].date()}, split {R.index[len(R) // 2].date()}")
print(tab.round(3).to_string())
print(f"SMLL weight: avg {w.mean():.1%}, range {w.min():.0%}-{w.max():.0%}; weekly |dw| {w.resample('W-FRI').last().diff().abs().mean():.3f}")
print("active by year (%):", (yearly * 100).round(1).to_dict())

# --- plot ------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
ORANGE, BLUE, AQUA = "#eb6834", "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), dpi=200, sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1.4, 1], "hspace": 0.18})
for a in (ax1, ax2, ax3):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
p_, b_ = tab.loc["Portfolio"], tab.loc["Benchmark 50/50"]
for k, c, lw, lab in (("Portfolio", INK, 2.2, f"Portfolio: {p_.CAGR:.1%}/yr, max DD {p_.MaxDD:.0%}"),
                      ("Benchmark 50/50", MUTED, 1.6, f"Benchmark 50/50: {b_.CAGR:.1%}/yr, max DD {b_.MaxDD:.0%}")):
    eq = (1 + R[k]).cumprod()
    ax1.plot(eq.index, eq, color=c, lw=lw, label=lab)
    ax1.annotate(f"{eq.iloc[-1]:.2f}×", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                 fontsize=9, color=c, va="center")
ax1.set_yscale("log")
ax1.set_yticks([0.5, 1, 2, 3, 4])
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax1.yaxis.set_minor_formatter(plt.NullFormatter())
ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
ax1.set_title("Growth of 1", loc="left", fontsize=10, color=INK2)
rel = ((1 + R["Portfolio"]).cumprod() / (1 + R["Benchmark 50/50"]).cumprod() - 1) * 100
ax2.fill_between(rel.index, 0, rel, where=rel >= 0, color=AQUA, alpha=0.25, lw=0)
ax2.fill_between(rel.index, 0, rel, where=rel < 0, color=ORANGE, alpha=0.25, lw=0)
ax2.plot(rel.index, rel, color=INK, lw=1.8)
ax2.axhline(0, color=MUTED, lw=1)
ax2.annotate(f"{rel.iloc[-1]:+.0f}%", (rel.index[-1], rel.iloc[-1]), xytext=(6, 0), textcoords="offset points",
             fontsize=9, va="center")
ax2.set_ylabel("Relative to 50/50 (%)")
ax2.set_title(f"Cumulative alpha vs. 50/50: {p_['Excess CAGR']:+.1%}/yr, tracking error {p_['Tracking error']:.1%}, "
              f"IR {p_.IR:.2f} (first half {p_['IR H1']:.2f}, second half {p_['IR H2']:.2f})",
              loc="left", fontsize=10, color=INK2)
ax3.fill_between(w.index, 0, w * 100, color=ORANGE, alpha=0.8, lw=0, step="post", label="SMLL")
ax3.fill_between(w.index, w * 100, 100, color=BLUE, alpha=0.55, lw=0, step="post", label="IBOV")
ax3.axhline(50, color=SURF, lw=1.2, ls="--")
ax3.set_ylim(0, 100)
ax3.set_ylabel("Weight (%)")
ax3.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2, ncol=2)
ax3.set_title(f"Allocation: SMLL weight = (signal + 1) / 2, range {w.min():.0%}–{w.max():.0%}", loc="left",
              fontsize=10, color=INK2)
fig.suptitle("Long-only SMLL / Ibovespa allocation vs. a 50/50 benchmark", x=0.07, ha="left", fontsize=14,
             fontweight="bold", y=0.975)
fig.text(0.07, 0.935, "Signal = COPOM + GDP revisions + real rate, halved when against the 3-month trend. Always 100% "
         f"invested, daily rebalancing, no costs. {R.index[0]:%b %Y} – {R.index[-1]:%b %Y}.", fontsize=9, color=INK2)
fig.subplots_adjust(left=0.08, right=0.93, top=0.9, bottom=0.05)
fig.savefig(HERE / "long_only.png")
print("saved")
