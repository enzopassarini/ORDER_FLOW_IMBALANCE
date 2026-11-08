"""
Risk overlays on the core strategy (COPOM + GDP + real rate, equal risk) to limit large
drawdowns the slow macro signals do not react to. Overlays only scale the core position
down (never flip it or add leverage, except vol targeting which may go to 1.5x), and every
overlay uses information up to the previous close (applied with a one-day lag).

  A  Trend-disagree cut:  if sign(core position) != sign(spread's trailing L-day return),
                          scale to 50% (L = 21, 63) or to 0 (L = 63)
  B  Stop on strategy P&L: if the core's trailing 21-day return < -1.5 x its 21-day sigma
                           (sigma from past daily vol), scale to 50% / 0
  C  Drawdown control:    if core equity is more than 5% below its 63-day high, scale to 50%
  D  Vol cap:             scale = min(1, median past 21d vol / current 21d vol of the spread)
  E  Vol target:          scale = clip(target / current 21d vol, 0, 1.5), target = median
  F  A(63, 50%) + D

Reports risk (max drawdown, worst 21-day return, daily 5% CVaR), Sharpe overall and by half,
average exposure, and what each overlay did in the core's five worst drawdown episodes.
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
p = d["Aggregate (3)"]
sp = sp_all.loc[p.index]
base = p * sp
idx_all = (1 + sp_all).cumprod()


def lagged(s):
    return s.shift(1).reindex(p.index)


scales = {}
for L in (21, 63):
    trend = lagged(np.sign(idx_all / idx_all.shift(L) - 1))
    scales[f"A trend-disagree {L}d → 50%"] = np.where(np.sign(p) * trend < 0, 0.5, 1.0)
trend63 = lagged(np.sign(idx_all / idx_all.shift(63) - 1))
scales["A trend-disagree 63d → 0"] = np.where(np.sign(p) * trend63 < 0, 0.0, 1.0)
r21 = base.rolling(21).apply(lambda x: np.prod(1 + x) - 1, raw=True)
sig21 = base.expanding(min_periods=252).std() * np.sqrt(21)
stop = (r21 < -1.5 * sig21).shift(1).fillna(False)
scales["B stop on P&L → 50%"] = np.where(stop, 0.5, 1.0)
scales["B stop on P&L → 0"] = np.where(stop, 0.0, 1.0)
eq = (1 + base).cumprod()
ddc = (eq / eq.rolling(63, min_periods=1).max() - 1).shift(1)
scales["C drawdown > 5% → 50%"] = np.where(ddc < -0.05, 0.5, 1.0)
v21 = sp_all.rolling(21).std()
vmed = v21.expanding(min_periods=252).median()
ratio = lagged(vmed / v21)
scales["D vol cap"] = ratio.clip(upper=1.0).fillna(1.0).to_numpy()
scales["E vol target"] = ratio.clip(0, 1.5).fillna(1.0).to_numpy()
scales["F trend 63d 50% + vol cap"] = scales["A trend-disagree 63d → 50%"] * scales["D vol cap"]

ret = pd.DataFrame({"Core (no overlay)": base})
expo = pd.DataFrame({"Core (no overlay)": np.ones(len(p))}, index=p.index)
for k, s in scales.items():
    s = pd.Series(s, index=p.index).fillna(1.0)
    ret[k] = s * base
    expo[k] = s
ret.to_csv(DATA / "overlay_daily_returns.csv")


def stats(x, e):
    eqx = (1 + x).cumprod()
    roll21 = (1 + x).rolling(21).apply(np.prod, raw=True) - 1
    q = x.quantile(0.05)
    m = len(x) // 2
    sh = lambda y: y.mean() / y.std() * np.sqrt(252)
    return {"CAGR": eqx.iloc[-1] ** (252 / len(x)) - 1, "Vol": x.std() * np.sqrt(252), "Sharpe": sh(x),
            "Sharpe H1": sh(x.iloc[:m]), "Sharpe H2": sh(x.iloc[m:]), "MaxDD": (eqx / eqx.cummax() - 1).min(),
            "Worst 21d": roll21.min(), "CVaR 5% (daily)": x[x <= q].mean(), "Calmar": (eqx.iloc[-1] ** (252 / len(x)) - 1)
            / -(eqx / eqx.cummax() - 1).min(), "Avg exposure": e.mean(), "Time cut": (e < 0.999).mean()}


tab = pd.DataFrame({k: stats(ret[k], expo[k]) for k in ret}).T
tab.to_csv(DATA / "overlay_stats.csv", float_format="%.4f")
pd.options.display.width = 220
print(f"{ret.index[0].date()} -> {ret.index[-1].date()}")
print(tab.round(3).to_string())

# worst drawdown episodes of the core
eqb = (1 + base).cumprod()
dd = eqb / eqb.cummax() - 1
eps, used = [], pd.Series(False, index=dd.index)
for _ in range(5):
    trough = dd[~used].idxmin()
    peak = eqb.loc[:trough].idxmax()
    rec = eqb.loc[trough:][eqb.loc[trough:] >= eqb.loc[peak]]
    end = rec.index[0] if len(rec) else dd.index[-1]
    used.loc[peak:end] = True
    eps.append((peak, trough))
ep = pd.DataFrame({f"{a:%b %Y} → {b:%b %Y}": {k: (1 + ret[k].loc[a:b]).prod() - 1 for k in ret} for a, b in eps}).T
ep.to_csv(DATA / "overlay_episodes.csv", float_format="%.4f")
print("\nReturn over the core's 5 worst drawdowns (peak → trough):\n", (ep * 100).round(1).to_string())
# the big single-day spread crashes
big = sp.nsmallest(5).index.union(sp.nlargest(3).index)
print("\nLargest spread days, core vs overlays (%):\n", (ret.loc[big] * 100).round(2).T.to_string())

# --- plot ------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SHOW = {"Core (no overlay)": INK, "A trend-disagree 63d → 50%": "#2a78d6", "B stop on P&L → 50%": "#eb6834",
        "D vol cap": "#1baf7a", "F trend 63d 50% + vol cap": "#6250d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig = plt.figure(figsize=(14, 10.5), dpi=200)
gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1], height_ratios=[2.2, 1, 1.4], hspace=0.35, wspace=0.22)
ax, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0], sharex=None)
axt, axe = fig.add_subplot(gs[0:2, 1]), fig.add_subplot(gs[2, :])
for a in (ax, axd, axt, axe):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
for k, c in SHOW.items():
    e_ = (1 + ret[k]).cumprod()
    t = tab.loc[k]
    ax.plot(e_.index, e_, color=c, lw=2.2 if k.startswith("Core") else 1.3,
            label=f"{k}: Sharpe {t.Sharpe:.2f} (H2 {t['Sharpe H2']:.2f}), max DD {t.MaxDD:.1%}, worst 21d {t['Worst 21d']:.1%}")
    axd.plot(e_.index, (e_ / e_.cummax() - 1) * 100, color=c, lw=1.4 if k.startswith("Core") else 0.9)
ax.axhline(1, color=MUTED, lw=1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.legend(loc="upper left", frameon=False, fontsize=7.3, labelcolor=INK2)
ax.set_title("Core with and without overlays (no costs)", loc="left", fontsize=9, color=INK2)
axd.set_ylabel("Drawdown (%)")
# table-like dot plot: change in Sharpe and MaxDD vs core
ks = [k for k in tab.index if k != "Core (no overlay)"]
yb = np.arange(len(ks))[::-1]
b0 = tab.loc["Core (no overlay)"]
axt.scatter((tab.loc[ks, "MaxDD"] - b0.MaxDD) * 100, yb, color="#1baf7a", s=40, zorder=3, label="Δ max drawdown (pp, + = better)")
axt.scatter((tab.loc[ks, "Worst 21d"] - b0["Worst 21d"]) * 100, yb, color="#2a78d6", marker="s", s=30, zorder=3,
            label="Δ worst 21-day return (pp)")
axt.scatter((tab.loc[ks, "Sharpe"] - b0.Sharpe) * 10, yb, color="#eb6834", marker="D", s=30, zorder=3,
            label="Δ Sharpe × 10")
axt.axvline(0, color=MUTED, lw=1)
axt.set_yticks(yb)
axt.set_yticklabels([f"{k}\nexposure {tab.loc[k, 'Avg exposure']:.2f}" for k in ks], fontsize=7.5)
axt.legend(loc="upper center", bbox_to_anchor=(0.4, -0.04), frameon=False, fontsize=7.5, labelcolor=INK2)
axt.set_title(f"Change vs. core (core: Sharpe {b0.Sharpe:.2f}, max DD {b0.MaxDD:.1%}, worst 21d {b0['Worst 21d']:.1%})",
              loc="left", fontsize=9, color=INK2)
xs = np.arange(len(ep))
wbar = 0.8 / len(SHOW)
for j, (k, c) in enumerate(SHOW.items()):
    axe.bar(xs + (j - len(SHOW) / 2 + 0.5) * wbar, ep[k] * 100, width=wbar, color=c, zorder=3, label=k)
axe.axhline(0, color=MUTED, lw=1)
axe.set_xticks(xs)
axe.set_xticklabels(ep.index, fontsize=8)
axe.set_ylabel("Return over episode (%)")
axe.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2)
axe.set_title("The core's five worst drawdowns (peak → trough): what each overlay would have lost", loc="left",
              fontsize=9, color=INK2)
fig.suptitle("Downside protection overlays on the core signal", x=0.04, ha="left", fontsize=14, fontweight="bold")
fig.text(0.04, 0.945, "Overlays only scale the core position (vol target may lever to 1.5×), use data up to the previous "
         "close, and are applied daily. Spread = SMLL − IBOV, daily rebalanced.", fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.05, right=0.98, top=0.91, bottom=0.05)
fig.savefig(HERE / "overlay_analysis.png")
print("saved")
