"""
Effect of scaling the final signal 1x-4x (clipped to [-1, 1]) on position sizes and performance.

  1x: q_t = final signal
  kx: clip(k * q_t, -1, 1), k = 2, 3, 4
Size metrics: average long size (mean q when q > 0), average short size (mean |q| when q < 0),
share of days long / short, share of days at the +-1 clip.
Performance after 5 bp costs, weekly rebalancing with no-trade band:
  long-short (unlevered, band 0.02) and long-only SMLL weight (q + 1) / 2 vs. 50/50 (band 1 pp).
"""
import contextlib
import io

import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import long_only_costs as lo
    import ls_costs as ls

MULTS = (1, 2, 3, 4)
q_full = pd.read_csv(ls.DATA / "final_core_daily.csv", index_col=0, parse_dates=True)["filtered_position"]
rows, keep = [], {}
for mult in MULTS:
    q = (mult * q_full).clip(-1, 1)
    size = {"Avg long size": q[q > 0].mean(), "Avg short size": -q[q < 0].mean(), "Avg |signal|": q.abs().mean(),
            "Days long": (q > 0).mean(), "Days short": (q < 0).mean(), "Days at ±1": (q.abs() >= 1).mean(),
            "Max long": q.max(), "Max short": -q.min()}
    d_ls = ls.simulate(q.loc[ls.idx].to_numpy(), "weekly", band=0.02)
    s_ls = ls.stats(d_ls, "net")
    tgt = ((q.loc[lo.dates] + 1) / 2).to_numpy()
    p = lo.simulate(tgt, "weekly", band=0.01)
    b = lo.simulate(np.full(len(lo.dates), 0.5), "weekly", band=0.01)
    s_lo = lo.summary(p, b, "net")
    keep[mult] = {"q": q, "ls": d_ls, "lo": p, "bm": b, "w": pd.Series(tgt, index=lo.dates)}
    rows.append({"signal": f"{mult}x", **size,
                 "L/S CAGR": s_ls["CAGR"], "L/S vol": s_ls["Vol"], "L/S Sharpe": s_ls["Sharpe"],
                 "L/S Sharpe H2": s_ls["Sharpe H2"], "L/S MaxDD": s_ls["MaxDD"], "L/S cost/yr": s_ls["Cost/yr"],
                 "LO CAGR": s_lo["CAGR"], "LO excess vs 50/50": s_lo["Excess CAGR"], "LO tracking error": s_lo["Tracking error"],
                 "LO IR": s_lo["IR"], "LO SMLL weight range": f"{tgt.min():.0%}–{tgt.max():.0%}"})
tab = pd.DataFrame(rows).set_index("signal").T
tab.to_csv(ls.DATA / "signal_scale_test.csv")
pd.options.display.width = 160
print(f"L/S sample {ls.idx[0].date()} -> {ls.idx[-1].date()}; long-only {lo.dates[0].date()} -> {lo.dates[-1].date()}")
print(tab.to_string())

# --- plots -----------------------------------------------------------------------------------
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COL = {1: "#9b9a95", 2: "#9085e9", 3: "#6250d6", 4: "#2c1f8f"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(a):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)


T = tab
fig = plt.figure(figsize=(14, 13), dpi=200)
gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], height_ratios=[1.15, 1.3, 1.3], hspace=0.38, wspace=0.18)
ax0, axh = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax1, ax2 = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
for a in (ax0, axh, ax1, ax2):
    style(a)
for m in MULTS[::-1]:
    qq = keep[m]["q"]
    ax0.plot(qq.index, qq, color=COL[m], lw=1.6 if m == 1 else 0.9, drawstyle="steps-post", label=f"{m}×",
             zorder=5 if m == 1 else 3)
ax0.axhline(0, color=MUTED, lw=1)
ax0.set_ylim(-1.05, 1.05)
ax0.set_ylabel("Signal")
h, l = ax0.get_legend_handles_labels()
ax0.legend(h[::-1], l[::-1], loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=4)
ax0.set_title("Signal at 1×–4×, clipped to ±1 (+ = long SMLL / short IBOV)", loc="left", fontsize=10, color=INK2)
x = np.arange(len(MULTS))
w_ = 0.27
for j, (row, lab) in enumerate((("Avg long size", "avg long"), ("Avg short size", "avg short"), ("Days at ±1", "days at cap"))):
    vals = [T.loc[row, f"{m}x"] for m in MULTS]
    axh.bar(x + (j - 1) * w_, vals, width=w_, color=["#1baf7a", "#eb6834", "#9b9a95"][j], label=lab, zorder=3)
    for i, v in enumerate(vals):
        axh.text(i + (j - 1) * w_, v + 0.012, f"{v:.2f}" if j < 2 else f"{v:.0%}", ha="center", fontsize=7)
axh.set_xticks(x)
axh.set_xticklabels([f"{m}×" for m in MULTS])
axh.set_ylim(0, 1)
axh.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
axh.set_title("Average long / short size and share of days at the cap", loc="left", fontsize=10, color=INK2)
for m in MULTS:
    d = keep[m]["ls"]
    eq = (1 + d["net"]).cumprod()
    k = f"{m}x"
    ax1.plot(eq.index, eq, color=COL[m], lw=2 if m in (1, 4) else 1.5,
             label=f"{m}×: CAGR {T.loc['L/S CAGR', k]:+.1%}, vol {T.loc['L/S vol', k]:.1%}, Sharpe {T.loc['L/S Sharpe', k]:.2f} "
                   f"(H2 {T.loc['L/S Sharpe H2', k]:.2f}), max DD {T.loc['L/S MaxDD', k]:.0%}")
    ax1.annotate(f"{eq.iloc[-1]:.2f}×", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                 fontsize=9, color=COL[m], va="center")
ax1.axhline(1, color=MUTED, lw=1)
ax1.set_yscale("log")
ax1.set_yticks([1, 1.5, 2, 3, 4])
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax1.yaxis.set_minor_formatter(plt.NullFormatter())
ax1.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
ax1.set_ylabel("Growth of 1 (log)")
ax1.set_title("Long-short SMLL − IBOV, unlevered (after 5 bp, weekly + band)", loc="left", fontsize=10, color=INK2)
for m in MULTS:
    p, b = keep[m]["lo"], keep[m]["bm"]
    rel = ((1 + p["net"]).cumprod() / (1 + b["net"]).cumprod() - 1) * 100
    k = f"{m}x"
    ax2.plot(rel.index, rel, color=COL[m], lw=2 if m in (1, 4) else 1.5,
             label=f"{m}×: CAGR {T.loc['LO CAGR', k]:.1%}, excess {T.loc['LO excess vs 50/50', k]:+.1%}/yr, "
                   f"TE {T.loc['LO tracking error', k]:.1%}, IR {T.loc['LO IR', k]:.2f}")
    ax2.annotate(f"{rel.iloc[-1]:+.0f}%", (rel.index[-1], rel.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                 fontsize=9, color=COL[m], va="center")
ax2.axhline(0, color=MUTED, lw=1)
ax2.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
ax2.set_ylabel("Relative to 50/50 (%)")
ax2.set_title("Long-only SMLL weight = (signal + 1) / 2: cumulative alpha vs. 50/50 (after 5 bp, weekly + 1 pp band)",
              loc="left", fontsize=10, color=INK2)
fig.suptitle("Final signal scaled 1× to 4× (clipped to ±1)", x=0.05, ha="left", fontsize=14, fontweight="bold", y=0.975)
fig.subplots_adjust(left=0.06, right=0.95, top=0.93, bottom=0.04)
fig.savefig(ls.HERE / "signal_scale_test.png")
print("saved")
