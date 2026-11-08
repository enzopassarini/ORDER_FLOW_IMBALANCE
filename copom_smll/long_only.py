"""
Long-only, always fully invested SMLL / IBOV portfolio driven by the unlevered final core
signal q_t in [-1, 1] (COPOM + GDP + real rate, halved when against the spread's 3-month
trend; data/final_core_daily.csv). q_t is already known before day t (one-day lag built in).

Weight in SMLL:  w_t = clip(w0 + k * q_t, 0, 1);  weight in IBOV: 1 - w_t.  No leverage, no shorts.
Daily return:    R_t = w_t * r_SMLL + (1 - w_t) * r_IBOV = r_IBOV + w_t * (r_SMLL - r_IBOV)
So the active return vs. IBOV is w_t * spread_t = w0 * spread_t (static small-cap tilt)
                                              + k * q_t * spread_t (signal timing).

Main spec: w0 = 50%, k = 0.5 -> q = +1 is 100% SMLL, q = -1 is 100% IBOV, q = 0 is 50/50.
Also: k = 1.0 (more aggressive, clipped), and a one-sided version with w0 = 0 (100% IBOV unless
the signal is positive, then w = q). Benchmarks: IBOV, SMLL and a static 50/50 mix.
Weights are reset daily (daily rebalancing); no costs.
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
q = fc["filtered_position"]

# rescale the [-1, 1] signal to a [0, 1] SMLL weight (all causal: only past values of q are used)
MAIN = "Linear: (q + 1) / 2"
ranks = q.expanding(min_periods=252).rank(pct=True)
qmin, qmax = q.expanding(min_periods=252).min(), q.expanding(min_periods=252).max()
W = pd.DataFrame({MAIN: (q + 1) / 2,
                  "Percentile rank of q (past)": ranks,
                  "Min–max of q (past range)": ((q - qmin) / (qmax - qmin)).clip(0, 1)}).dropna()
q = q.loc[W.index]
rS, rI = rS.loc[W.index], rI.loc[W.index]
W["Static 50/50"] = 0.5
W["SMLL"] = 1.0
W["IBOV"] = 0.0
R = W.mul(rS, axis=0) + (1 - W).mul(rI, axis=0)
R.to_csv(DATA / "long_only_daily_returns.csv")
W.to_csv(DATA / "long_only_weights.csv")


def stats(r, w):
    eq = (1 + r).cumprod()
    act = r - rI
    m = len(r) // 2
    yrs = len(r) / 252
    beta = np.cov(r, rI)[0, 1] / rI.var()
    te = act.std() * np.sqrt(252)
    ann_act = (1 + r).prod() ** (1 / yrs) - (1 + rI).prod() ** (1 / yrs)
    ir = lambda a: a.mean() / a.std() * np.sqrt(252) if a.std() > 0 else np.nan
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "Vol": r.std() * np.sqrt(252),
            "Return/vol": r.mean() / r.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min(),
            "Excess CAGR vs IBOV": ann_act, "Tracking error": te, "IR": ir(act), "IR H1": ir(act.iloc[:m]),
            "IR H2": ir(act.iloc[m:]), "Beta to IBOV": beta, "Avg SMLL weight": w.mean(),
            "SMLL weight range": f"{w.min():.0%}–{w.max():.0%}",
            "Hit rate (monthly active > 0)": ((1 + act).resample("ME").prod() - 1).gt(0).mean(),
            "Turnover (avg weekly |Δw|)": w.resample("W-FRI").last().diff().abs().mean()}


tab = pd.DataFrame({k: stats(R[k], W[k]) for k in R}).T
tab.to_csv(DATA / "long_only_stats.csv")
pd.options.display.width = 250
print(f"{R.index[0].date()} -> {R.index[-1].date()}")
print(tab.to_string())

# alpha decomposition: average SMLL weight (static tilt) vs. deviations from it (timing)
sp = rS - rI
wm = W[MAIN]
w0 = wm.mean()
decomp = pd.DataFrame({"Active vs IBOV": wm * sp, "Static 50% tilt": w0 * sp, "Timing (signal)": (wm - w0) * sp})
for k_ in W.columns[:3]:
    wk = W[k_]
    print(f"{k_}: avg weight {wk.mean():.1%}; static tilt {(wk.mean() * sp).sum() * 100:+.1f} pp, "
          f"timing {((wk - wk.mean()) * sp).sum() * 100:+.1f} pp")
cum = decomp.cumsum() * 100  # additive, in pp
yearly = decomp.groupby(decomp.index.year).sum() * 100
yearly.to_csv(DATA / "long_only_alpha_by_year.csv", float_format="%.2f")
print("\nAlpha decomposition, main spec (sum of daily active returns, pp):", cum.iloc[-1].round(1).to_dict())
print("\nBy year (pp):\n", yearly.round(1).to_string())

# --- plot ------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9b9a95"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig = plt.figure(figsize=(14, 13), dpi=200)
gs = fig.add_gridspec(4, 2, width_ratios=[2.1, 1], height_ratios=[1.8, 1.4, 1, 1.2], hspace=0.42, wspace=0.2)
ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])
ax4, ax5, ax6 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1:3, 1]), fig.add_subplot(gs[3, :])
for a in (ax1, ax2, ax3, ax4, ax5, ax6):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
COL = {MAIN: INK, "Percentile rank of q (past)": VIOLET,
       "Min–max of q (past range)": AQUA, "Static 50/50": GRAY, "SMLL": ORANGE, "IBOV": BLUE}
for kname, c in COL.items():
    eq = (1 + R[kname]).cumprod()
    ax1.plot(eq.index, eq, color=c, lw=2.3 if kname == MAIN else 1.2,
             ls="--" if kname == "Static 50/50" else "-",
             label=f"{kname}: CAGR {tab.loc[kname, 'CAGR']:.1%}, max DD {tab.loc[kname, 'MaxDD']:.0%}")
ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax1.yaxis.set_minor_formatter(plt.NullFormatter())
ax1.set_yticks([0.5, 0.75, 1, 1.5, 2, 3, 4])
ax1.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK2)
ax1.set_title("Growth of 1 (total-return indices, no costs)", loc="left", fontsize=9, color=INK2)
for kname, c in COL.items():
    if kname == "IBOV":
        continue
    act = ((1 + R[kname]).cumprod() / (1 + rI).cumprod() - 1) * 100
    ax2.plot(act.index, act, color=c, lw=2.3 if kname == MAIN else 1.2, ls="--" if kname == "Static 50/50" else "-",
             label=f"{kname}: {act.iloc[-1]:+.0f}%  (IR {tab.loc[kname, 'IR']:.2f})")
ax2.axhline(0, color=MUTED, lw=1)
ax2.set_ylabel("Cumulative relative return vs. IBOV (%)")
ax2.set_ylim(-45, 140)
ax2.legend(loc="upper left", frameon=False, fontsize=7.2, labelcolor=INK2, ncol=2)
ax2.set_title("Cumulative alpha vs. Ibovespa (portfolio wealth / IBOV wealth − 1)", loc="left", fontsize=9, color=INK2)
ax3.plot(cum.index, cum["Active vs IBOV"], color=INK, lw=2, label=f"total active: {cum['Active vs IBOV'].iloc[-1]:+.0f} pp")
ax3.plot(cum.index, cum["Static 50% tilt"], color=GRAY, lw=1.3, ls="--",
         label=f"static tilt (average weight {w0:.0%}): {cum['Static 50% tilt'].iloc[-1]:+.0f} pp")
ax3.plot(cum.index, cum["Timing (signal)"], color=AQUA, lw=1.8, label=f"signal timing: {cum['Timing (signal)'].iloc[-1]:+.0f} pp")
ax3.axhline(0, color=MUTED, lw=1)
ax3.set_ylabel("Sum of daily active returns (pp)")
ax3.set_ylim(-30, 45)
ax3.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK2, ncol=3)
ax3.set_title("Linear rescale: active return = static tilt at its average SMLL weight + timing (deviations from it)", loc="left", fontsize=9, color=INK2)
ax4.fill_between(wm.index, 0, wm * 100, color=ORANGE, alpha=0.75, lw=0, step="post", label="SMLL")
ax4.fill_between(wm.index, wm * 100, 100, color=BLUE, alpha=0.6, lw=0, step="post", label="IBOV")
ax4.axhline(50, color=SURF, lw=1, ls="--")
ax4.set_ylim(0, 100)
ax4.set_ylabel("Portfolio weight (%)")
ax4.legend(loc="lower left", frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2)
ax4.set_title(f"Linear rescale allocation: SMLL {wm.min():.0%}–{wm.max():.0%}, average {wm.mean():.0%}",
              loc="left", fontsize=9, color=INK2)
yb = yearly.index
ax5.barh(yb + 0.2, yearly["Timing (signal)"], height=0.4, color=AQUA, label="signal timing", zorder=3)
ax5.barh(yb - 0.2, yearly["Static 50% tilt"], height=0.4, color=GRAY, label="static tilt", zorder=3)
ax5.axvline(0, color=MUTED, lw=1)
ax5.invert_yaxis()
ax5.set_yticks(yb[::2])
ax5.set_xlabel("Active return vs. IBOV (pp, sum of daily)")
ax5.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=INK2)
pos_yrs = (yearly["Timing (signal)"] > 0).mean()
ax5.set_title(f"By year: timing positive in {pos_yrs:.0%} of years", loc="left", fontsize=9, color=INK2)
ax6.axis("off")
cols = ["CAGR", "Vol", "MaxDD", "Excess CAGR vs IBOV", "Tracking error", "IR", "IR H1", "IR H2", "Beta to IBOV",
        "Avg SMLL weight", "SMLL weight range", "Hit rate (monthly active > 0)", "Turnover (avg weekly |Δw|)"]
pctc = {"CAGR", "Vol", "MaxDD", "Excess CAGR vs IBOV", "Tracking error", "Avg SMLL weight", "Hit rate (monthly active > 0)"}
cell = []
for kname in COL:
    row = [kname]
    for c in cols:
        v = tab.loc[kname, c]
        row.append(v if isinstance(v, str) else ("–" if pd.isna(v) else (f"{v:.1%}" if c in pctc else f"{v:.2f}")))
    cell.append(row)
t = ax6.table(cellText=cell, colLabels=["portfolio", "CAGR", "vol", "max DD", "excess CAGR", "track. err.", "IR",
                                        "IR H1", "IR H2", "beta", "avg SMLL", "SMLL range", "monthly hit", "turnover/wk"],
              loc="center", cellLoc="center", colWidths=[0.2] + [0.0615] * 13)
t.auto_set_font_size(False)
t.set_fontsize(7.8)
t.scale(1, 1.4)
for (i, j), c_ in t.get_celld().items():
    c_.set_edgecolor(GRID)
    if i == 0:
        c_.set_text_props(color=INK2, fontweight="bold")
    if i == 1:
        c_.set_text_props(fontweight="bold")
ax6.set_title("IR = information ratio of daily active return vs. IBOV (annualised). Halves split "
              f"{R.index[len(R) // 2]:%b %Y}.", loc="left", fontsize=8.5, color=INK2)
fig.suptitle("Long-only SMLL / Ibovespa allocation driven by the final core signal", x=0.04, ha="left",
             fontsize=14, fontweight="bold")
fig.text(0.04, 0.955, "Weight in SMLL = the [−1, 1] signal rescaled to [0, 1], rest in IBOV; always 100% invested, no "
         f"leverage or shorts. Signal = COPOM + GDP + real rate, halved against the 3-month trend. {R.index[0]:%b %Y} – "
         f"{R.index[-1]:%b %Y}.", fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.02)
fig.savefig(HERE / "long_only.png")
print("saved")
