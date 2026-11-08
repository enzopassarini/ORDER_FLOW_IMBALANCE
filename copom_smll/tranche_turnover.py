"""
How 4 staggered 21-day tranches reduce turnover, using the real-rate signal.

Weekly signal s_t (real rate vs. rolling 5y mean, clip(x / 3 sigma, +-1), Friday survey).
  Weekly version:  position_t = s_t                       -> trade each week: |s_t - s_{t-1}|
  Tranche version: each week opens a tranche of s_t / 4, held 4 weeks, so
                   position_t = (s_t + s_{t-1} + s_{t-2} + s_{t-3}) / 4
                   -> trade each week: |s_t - s_{t-4}| / 4  (one tranche in, one out)
Both entered at the Monday close after the survey. Turnover in position units (x2 for the two
legs of the long-short). Backtest: unlevered long-short, 5 bp per unit traded on each leg,
legs drift between weekly rebalances (ls_costs.simulate, no band).
"""
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import ls_costs as ls

DATA, HERE = ls.DATA, ls.HERE
s = pd.read_csv(DATA / "focus_channels_signal.csv", index_col=0, parse_dates=True)["real_rate_r5"].dropna()
tr = s.rolling(4).mean()
w = pd.DataFrame({"weekly": s, "tranches": tr}).dropna()
dw = w.diff().abs().dropna()
yrs = len(dw) / 52.18
stats = pd.DataFrame({
    "Avg weekly |Δposition|": dw.mean(),
    "Median weekly |Δposition|": dw.median(),
    "95th pct weekly |Δposition|": dw.quantile(0.95),
    "Largest weekly |Δposition|": dw.max(),
    "Weeks with |Δ| > 0.05": (dw > 0.05).mean(),
    "Weeks with |Δ| > 0.10": (dw > 0.10).mean(),
    "Turnover per year (position units)": dw.sum() / yrs,
    "Turnover per year, both legs": 2 * dw.sum() / yrs,
    "Sign changes per year": (np.sign(w).diff().abs() > 0).sum() / yrs,
}).T
stats["reduction"] = 1 - stats["tranches"] / stats["weekly"]
print(f"weekly signals {w.index[0].date()} -> {w.index[-1].date()} ({len(dw)} weeks)")
print(stats.round(3).to_string())
stats.to_csv(DATA / "tranche_turnover_stats.csv", float_format="%.4f")

# biggest weekly jumps of the raw signal and what the tranche version did
big = dw["weekly"].nlargest(8).index.sort_values()
ex = pd.DataFrame({"signal before": w["weekly"].shift(1).loc[big], "signal after": w["weekly"].loc[big],
                   "weekly trade": dw["weekly"].loc[big], "tranche trade this week": dw["tranches"].loc[big],
                   "tranche trade, next 3 weeks": [dw["tranches"].loc[d:].iloc[1:4].sum() for d in big]})
print("\nLargest weekly signal jumps:\n", ex.round(3).to_string())
ex.to_csv(DATA / "tranche_turnover_examples.csv", float_format="%.4f")

# one worked example: the largest jump, week by week
d0 = dw["weekly"].idxmax()
i0 = w.index.get_loc(d0)
win = w.iloc[i0 - 2:i0 + 6].copy()
win["weekly trade"] = w["weekly"].diff().abs().iloc[i0 - 2:i0 + 6]
win["tranche trade"] = w["tranches"].diff().abs().iloc[i0 - 2:i0 + 6]
print(f"\nWorked example around {d0.date()}:\n", win.round(3).to_string())
win.to_csv(DATA / "tranche_turnover_worked_example.csv", float_format="%.4f")


# backtest with costs
def daily(sig):
    p = pd.Series(np.nan, index=ls.idx)
    for d, v in sig.dropna().items():
        j = ls.px.index.searchsorted(d, side="right")
        if j < len(ls.px):
            p.loc[p.index > ls.px.index[j]] = v
    return p.ffill().fillna(0).to_numpy()


bt = {}
for k in ("weekly", "tranches"):
    df = ls.simulate(daily(w[k]), "weekly")
    bt[k] = df
    for col in ("gross", "net"):
        st = ls.stats(df, col)
        print(f"{k:9s} {col:5s}: CAGR {st['CAGR']:.2%}, Sharpe {st['Sharpe']:.2f}, turnover {st['Turnover/yr']:.2f}/yr, "
              f"cost {st['Cost/yr']:.3%}/yr")

# --- plot ------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
ORANGE, VIOLET = "#eb6834", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig = plt.figure(figsize=(14, 12), dpi=200)
gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], height_ratios=[1.1, 1, 1.1], hspace=0.42, wspace=0.2)
ax1, ax2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0])
ax3, ax4, ax5 = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
for a in (ax1, ax2, ax3, ax4, ax5):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax1.plot(w.index, w["weekly"], color=ORANGE, lw=0.9, drawstyle="steps-post", label="weekly: position = this week's signal")
ax1.plot(w.index, w["tranches"], color=INK, lw=1.5, drawstyle="steps-post", label="4 tranches: average of last 4 signals")
ax1.axhline(0, color=MUTED, lw=1)
ax1.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
ax1.set_ylabel("Position")
ax1.set_title("Real-rate signal: weekly vs. 4-tranche position (they look almost the same – the signal is very persistent)",
              loc="left", fontsize=10, color=INK2)
# turnover over time (rolling 52-week sum)
rt = dw.rolling(52).sum()
ax2.plot(rt.index, rt["weekly"], color=ORANGE, lw=1.3, label="weekly")
ax2.plot(rt.index, rt["tranches"], color=INK, lw=1.5, label="4 tranches")
ax2.set_ylabel("Position traded, past 52 weeks")
ax2.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
ax2.set_title(f"Turnover per year: {stats.loc['Turnover per year (position units)', 'weekly']:.2f} → "
              f"{stats.loc['Turnover per year (position units)', 'tranches']:.2f} "
              f"({stats.loc['Turnover per year (position units)', 'reduction']:.0%} less)", loc="left", fontsize=10,
              color=INK2)
# distribution of weekly trade size
bins = np.linspace(0, dw.max().max(), 40)
ax3.hist(dw["weekly"], bins=bins, color=ORANGE, alpha=0.6, label="weekly", zorder=3)
ax3.hist(dw["tranches"], bins=bins, color=INK, alpha=0.55, label="4 tranches", zorder=3)
ax3.set_yscale("log")
ax3.set_xlabel("Weekly |Δ position|")
ax3.set_ylabel("Number of weeks (log)")
ax3.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
ax3.set_title(f"Largest weekly trade: {dw['weekly'].max():.2f} → {dw['tranches'].max():.2f}", loc="left",
              fontsize=10, color=INK2)
# worked example
x = np.arange(len(win))
ax4.step(x, win["weekly"], where="post", color=ORANGE, lw=2, label="weekly position")
ax4.step(x, win["tranches"], where="post", color=INK, lw=2, label="tranche position")
for i, (d, r_) in enumerate(win.iterrows()):
    ax4.text(i + 0.5, max(r_["weekly"], r_["tranches"]) + 0.03, f"Δ {r_['weekly trade']:.2f}\nΔ {r_['tranche trade']:.2f}",
             ha="center", fontsize=7.5, color=INK2)
ax4.set_xticks(x + 0.5)
ax4.set_xticklabels([d.strftime("%d %b\n%Y") for d in win.index], fontsize=7.5)
ax4.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
ax4.set_ylim(min(win[["weekly", "tranches"]].min()) - 0.1, max(win[["weekly", "tranches"]].max()) + 0.2)
ax4.set_title(f"Example: the biggest weekly jump ({d0:%d %b %Y}). Labels: trade size weekly (top) / tranches (bottom)",
              loc="left", fontsize=10, color=INK2)
# backtest
for k, c in (("weekly", ORANGE), ("tranches", INK)):
    df = bt[k]
    st = ls.stats(df, "net")
    eq = (1 + df["net"]).cumprod()
    ax5.plot(eq.index, eq, color=c, lw=1.6, label=f"{k}: Sharpe {st['Sharpe']:.2f}, cost {st['Cost/yr']:.2%}/yr")
ax5.axhline(1, color=MUTED, lw=1)
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax5.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
ax5.set_title("Long-short backtest after 5 bp (unlevered)", loc="left", fontsize=10, color=INK2)
fig.suptitle("How 4 staggered tranches reduce turnover: real-rate signal", x=0.04, ha="left", fontsize=14,
             fontweight="bold", y=0.975)
fig.text(0.04, 0.945, "Tranches: each week opens signal/4, held 4 weeks → position = average of the last 4 weekly "
         "signals, and each week's trade = |s(t) − s(t−4)| / 4 instead of |s(t) − s(t−1)|.", fontsize=9, color=INK2)
fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.05)
fig.savefig(HERE / "tranche_turnover.png")
print("saved")
