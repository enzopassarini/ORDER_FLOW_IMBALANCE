"""
Robustness of the trend filter on the core (COPOM + GDP + real rate) across lookbacks and
cut sizes, plus an illustration of how the filter works.

Filter, each day t (using closes up to t-1):
    trend_L  = SMLL-IBOV spread return over the last L sessions (sign only)
    if sign(core position) != sign(trend_L): position = cut x core position
    else:                                    position = core position
Lookbacks L = 10 ... 252 sessions; cut = 0 (flat), 0.25, 0.5, 0.75, 1 (= no filter).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
d = pd.read_csv(DATA / "aggregate3_daily.csv", index_col=0, parse_dates=True)
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
sp_all = (px["SMLL"].pct_change() - px["IBOV"].pct_change()).dropna()
idx = (1 + sp_all).cumprod()
p = d["Aggregate (3)"]
sp = sp_all.loc[p.index]
base = p * sp
LB = [10, 21, 42, 63, 84, 126, 189, 252]
CUTS = [0.0, 0.25, 0.5, 0.75, 1.0]
v21 = sp_all.rolling(21).std()
volcap = (v21.expanding(min_periods=252).median() / v21).clip(upper=1).shift(1).reindex(p.index).fillna(1)


def trend(L):
    return np.sign(idx / idx.shift(L) - 1).shift(1).reindex(p.index)


def filt(L, cut):
    return pd.Series(np.where(np.sign(p) * trend(L) < 0, cut, 1.0), index=p.index)


def stats(x, e):
    eq = (1 + x).cumprod()
    m = len(x) // 2
    sh = lambda y: y.mean() / y.std() * np.sqrt(252)
    roll = (1 + x).rolling(21).apply(np.prod, raw=True) - 1
    return {"Sharpe": sh(x), "Sharpe H1": sh(x.iloc[:m]), "Sharpe H2": sh(x.iloc[m:]),
            "MaxDD": (eq / eq.cummax() - 1).min(), "Worst 21d": roll.min(), "CAGR": eq.iloc[-1] ** (252 / len(x)) - 1,
            "Avg exposure": e.mean(), "Share of days cut": (e < 0.999).mean(),
            "Weekly turnover": (p * e).resample("W-FRI").last().diff().abs().mean()}


rows = []
for L in LB:
    for c in CUTS:
        s = filt(L, c)
        rows.append({"L": L, "cut": c, "vol cap": False, **stats(s * base, s)})
        s2 = s * volcap
        rows.append({"L": L, "cut": c, "vol cap": True, **stats(s2 * base, s2)})
res = pd.DataFrame(rows)
res.to_csv(DATA / "overlay_robustness.csv", index=False, float_format="%.4f")
core = res[(res.cut == 1) & (~res["vol cap"])].iloc[0]
pd.options.display.width = 200
h = res[(res.cut == 0.5) & (~res["vol cap"])].set_index("L")
print("Core:", core[["Sharpe", "Sharpe H2", "MaxDD", "Worst 21d", "CAGR"]].round(3).to_dict())
print("\nHalve on disagreement, by lookback:\n", h[["Sharpe", "Sharpe H1", "Sharpe H2", "MaxDD", "Worst 21d", "CAGR",
                                                    "Avg exposure", "Share of days cut", "Weekly turnover"]].round(3).to_string())
hv = res[(res.cut == 0.5) & (res["vol cap"])].set_index("L")
print("\n+ vol cap:\n", hv[["Sharpe", "Sharpe H2", "MaxDD", "Worst 21d", "Avg exposure"]].round(3).to_string())
print("\nAgreement between lookbacks (share of days with the same filter decision, cut 0.5):")
dec = pd.DataFrame({L: (np.sign(p) * trend(L) < 0) for L in LB})
print(pd.DataFrame({a: {b: (dec[a] == dec[b]).mean() for b in LB} for a in LB}).round(2).to_string())

# --- plots -----------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(a, yonly=False):
    a.grid(True, axis="y" if yonly else "both", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


fig = plt.figure(figsize=(14, 12.5), dpi=200)
gs = fig.add_gridspec(4, 4, height_ratios=[1.1, 1.1, 1.3, 1.4], hspace=0.6, wspace=0.35)
# heatmaps: lookback x cut, no vol cap
metrics = [("Sharpe", "Sharpe (full)", "RdYlGn", "{:.2f}"), ("Sharpe H2", "Sharpe, second half", "RdYlGn", "{:.2f}"),
           ("MaxDD", "Max drawdown", "RdYlGn", "{:.1%}"), ("Worst 21d", "Worst 21-day return", "RdYlGn", "{:.1%}")]
for j, (col, title, cmap, fmt) in enumerate(metrics):
    ax = fig.add_subplot(gs[0, j])
    m = res[~res["vol cap"]].pivot(index="L", columns="cut", values=col).loc[LB, CUTS]
    center = core[col]
    span = np.nanmax(np.abs(m.values - center))
    ax.imshow(m.values, cmap=cmap, vmin=center - span, vmax=center + span, aspect="auto")
    for i in range(m.shape[0]):
        for k in range(m.shape[1]):
            ax.text(k, i, fmt.format(m.values[i, k]), ha="center", va="center", fontsize=7,
                    fontweight="bold" if (LB[i] == 63 and CUTS[k] == 0.5) else None)
    ax.set_xticks(range(len(CUTS)))
    ax.set_xticklabels(["flat", "25%", "50%", "75%", "none"], fontsize=7.5)
    ax.set_yticks(range(len(LB)))
    ax.set_yticklabels([f"{L}d" for L in LB], fontsize=7.5)
    ax.set_xlabel("Position kept when trend disagrees", fontsize=7.5)
    if j == 0:
        ax.set_ylabel("Trend lookback (sessions)")
    ax.set_title(f"{title}\n(green = better than core {fmt.format(center)})", loc="left", fontsize=8.5, color=INK2)
# lines: metric vs lookback at 50% cut, with/without vol cap
for j, (col, title, fmt) in enumerate((("Sharpe", "Sharpe", "{:.2f}"), ("MaxDD", "Max drawdown", "{:.0%}"))):
    ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
    style(ax)
    ax.plot(range(len(LB)), h[col], marker="o", color=BLUE, lw=1.8, label="Halve on disagreement")
    ax.plot(range(len(LB)), hv[col], marker="s", color=AQUA, lw=1.8, label="Halve + vol cap")
    if col == "Sharpe":
        ax.plot(range(len(LB)), h["Sharpe H2"], marker="o", color=BLUE, lw=1, ls="--", label="Halve, second half")
        ax.axhline(core["Sharpe H2"], color=MUTED, lw=1, ls=":")
        ax.text(len(LB) - 0.5, core["Sharpe H2"], " core H2", fontsize=7, color=MUTED, va="center")
    ax.axhline(core[col], color=INK, lw=1.2, ls="--")
    ax.text(len(LB) - 0.5, core[col], " core", fontsize=7.5, color=INK, va="center")
    ax.set_xticks(range(len(LB)))
    ax.set_xticklabels([f"{L}d" for L in LB])
    ax.set_xlabel("Trend lookback (sessions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, f=fmt: f.format(v)))
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left" if col == "MaxDD" else "upper center", bbox_to_anchor=None if col == "MaxDD" else (0.5, -0.28), ncol=3)
    ax.set_title(f"{title} by lookback, keeping 50% when the trend disagrees", loc="left", fontsize=8.5, color=INK2)
# equity curves, all lookbacks at 50%
ax = fig.add_subplot(gs[2, :])
style(ax)
cm_ = plt.get_cmap("viridis")
for i, L in enumerate(LB):
    s = filt(L, 0.5)
    eq = (1 + s * base).cumprod()
    ax.plot(eq.index, eq, color=cm_(i / (len(LB) - 1)), lw=1, alpha=0.9, label=f"{L}d")
eq0 = (1 + base).cumprod()
ax.plot(eq0.index, eq0, color=INK, lw=2, label="core, no filter")
ax.axhline(1, color=MUTED, lw=1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK2, ncol=9)
ax.set_title("Equity curves for every lookback (keep 50% on disagreement): all trace the core, with shallower "
             "drawdowns in 2008, 2020 and 2022–25", loc="left", fontsize=8.5, color=INK2)
# illustration of the mechanics, 2021-2025
a0, a1 = "2021-01-01", "2025-12-31"
ax = fig.add_subplot(gs[3, :2])
style(ax)
t63 = (idx / idx.shift(63) - 1).loc[a0:a1] * 100
ax.fill_between(t63.index, 0, t63.clip(lower=0), color=AQUA, alpha=0.35, lw=0, label="spread 63d return > 0 (uptrend)")
ax.fill_between(t63.index, 0, t63.clip(upper=0), color=ORANGE, alpha=0.35, lw=0, label="spread 63d return < 0 (downtrend)")
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylabel("Spread 3-month return (%)")
ax.legend(loc="lower left", frameon=False, fontsize=7.5, labelcolor=INK2)
ax.set_title("How it works (2021–2025): 1) the trend = sign of the spread's last-63-session return …",
             loc="left", fontsize=8.5, color=INK2)
ax = fig.add_subplot(gs[3, 2:])
style(ax)
pf = (p * filt(63, 0.5)).loc[a0:a1]
pc = p.loc[a0:a1]
dis = (np.sign(pc) * trend(63).loc[a0:a1] < 0)
ax.fill_between(pc.index, -1, 1, where=dis, color=ORANGE, alpha=0.12, lw=0, step="post",
                label="core disagrees with trend → position halved")
ax.plot(pc.index, pc, color=MUTED, lw=1, drawstyle="steps-post", label="core position")
ax.plot(pf.index, pf, color=INK, lw=1.4, drawstyle="steps-post", label="filtered position")
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylim(-1.05, 1.0)
ax.set_ylabel("Position (+ = long SMLL)")
ax.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=INK2)
ax.set_title("… 2) where the core points the other way, hold half the position", loc="left", fontsize=8.5, color=INK2)
fig.suptitle("Trend filter on the core: robustness across lookbacks and cut sizes", x=0.04, ha="left",
             fontsize=14, fontweight="bold")
fig.text(0.04, 0.955, "Rule: if sign(core position) ≠ sign(SMLL − IBOV return over the last L sessions, up to yesterday's "
         "close), keep only a fraction of the core position; otherwise keep it all. Jan 2007 – Nov 2026, no costs.",
         fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.04)
fig.savefig(HERE / "overlay_robustness.png")
print("saved")
