"""
Volatility targeting of the final core strategy.

Core position q_t = Aggregate(COPOM, GDP tranches, real-rate tranches), halved when its sign
disagrees with the SMLL-IBOV spread's 63-session return (trend filter, as of the previous close).
Unlevered daily P&L: x_t = q_t * (r_SMLL - r_IBOV).

Vol targeting: leverage L_t = target / sigma_hat_t, capped at L_max, with sigma_hat_t the
annualised volatility of the strategy's own past daily P&L (up to t-1):
  rolling windows of 21 / 63 / 126 / 252 sessions, or EWMA with half-life 21 / 63 sessions.
Levered P&L = L_t * x_t. Variants: daily vs. weekly (Friday estimate, applied from Monday)
leverage updates, and caps of 3x / 5x / 8x. Benchmark: a constant leverage that hits 10%
over the full sample (uses hindsight).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
TARGET = 0.10
d = pd.read_csv(DATA / "aggregate3_daily.csv", index_col=0, parse_dates=True)
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
sp_all = (px["SMLL"].pct_change() - px["IBOV"].pct_change()).dropna()
idx = (1 + sp_all).cumprod()
p = d["Aggregate (3)"]
sp = sp_all.loc[p.index]
trend = np.sign(idx / idx.shift(63) - 1).shift(1).reindex(p.index)
q = p * np.where(np.sign(p) * trend < 0, 0.5, 1.0)
x = q * sp
pd.DataFrame({"core_position": p, "filtered_position": q, "spread": sp, "pnl": x}).to_csv(DATA / "final_core_daily.csv")
print(f"Final core (unlevered): vol {x.std() * np.sqrt(252):.2%}, Sharpe {x.mean() / x.std() * np.sqrt(252):.2f}")

EST = {"roll 21d": x.rolling(21, min_periods=21).std(), "roll 63d": x.rolling(63, min_periods=63).std(),
       "roll 126d": x.rolling(126, min_periods=126).std(), "roll 252d": x.rolling(252, min_periods=252).std(),
       "EWMA hl 21d": x.ewm(halflife=21, min_periods=63).std(), "EWMA hl 63d": x.ewm(halflife=63, min_periods=63).std()}


def run(sig, cap=5.0, weekly=False):
    s = (sig * np.sqrt(252)).shift(1)                    # known at t-1
    lev = (TARGET / s).clip(upper=cap)
    if weekly:
        fri = lev.resample("W-FRI").last()
        lev = fri.reindex(fri.index.union(lev.index)).ffill().reindex(lev.index).shift(1)
    lev = lev.dropna()
    return lev, (lev * x.loc[lev.index])


def stats(r, lev):
    eq = (1 + r).cumprod()
    yr = r.groupby(r.index.year).std() * np.sqrt(252)
    rv63 = r.rolling(63).std() * np.sqrt(252)
    m = len(r) // 2
    sh = lambda y: y.mean() / y.std() * np.sqrt(252)
    return {"Realised vol": r.std() * np.sqrt(252), "Sharpe": sh(r), "Sharpe H1": sh(r.iloc[:m]),
            "Sharpe H2": sh(r.iloc[m:]), "CAGR": eq.iloc[-1] ** (252 / len(r)) - 1,
            "MaxDD": (eq / eq.cummax() - 1).min(), "Worst 21d": ((1 + r).rolling(21).apply(np.prod, raw=True) - 1).min(),
            "Yearly vol min": yr.min(), "Yearly vol max": yr.max(),
            "63d vol within 7–13%": rv63.between(0.07, 0.13).mean(), "Avg leverage": lev.mean(),
            "Max leverage": lev.max(), "Days at cap": (lev >= lev.max() - 1e-9).mean(),
            "Weekly Δ leverage": lev.resample("W-FRI").last().diff().abs().mean()}


rows, keep = [], {}
start = None
for name, sig in EST.items():
    for cap in (3.0, 5.0, 8.0):
        for weekly in (False, True):
            lev, r = run(sig, cap, weekly)
            keep[(name, cap, weekly)] = (lev, r)
            start = max(start, r.index[0]) if start else r.index[0]
start = max(v[1].index[0] for v in keep.values())
for (name, cap, weekly), (lev, r) in keep.items():
    rows.append({"estimator": name, "cap": cap, "update": "weekly" if weekly else "daily",
                 **stats(r.loc[start:], lev.loc[start:])})
const_L = TARGET / (x.loc[start:].std() * np.sqrt(252))
rows.append({"estimator": "constant (hindsight)", "cap": np.nan, "update": "-",
             **stats(const_L * x.loc[start:], pd.Series(const_L, index=x.loc[start:].index))})
rows.append({"estimator": "unlevered", "cap": np.nan, "update": "-",
             **stats(x.loc[start:], pd.Series(1.0, index=x.loc[start:].index))})
tab = pd.DataFrame(rows)
tab.to_csv(DATA / "vol_target_grid.csv", index=False, float_format="%.4f")
pd.options.display.width = 250
cols = ["estimator", "cap", "update", "Realised vol", "Yearly vol min", "Yearly vol max", "63d vol within 7–13%", "Sharpe",
        "Sharpe H2", "MaxDD", "Worst 21d", "Avg leverage", "Max leverage", "Days at cap", "Weekly Δ leverage"]
print(f"\nCommon sample {start.date()} -> {x.index[-1].date()}")
print(tab[(tab.cap == 5.0) | tab.cap.isna()][cols].round(3).to_string(index=False))
print("\nCap sensitivity (EWMA hl 63d, weekly):")
print(tab[(tab.estimator == "EWMA hl 63d") & (tab["update"] == "weekly")][cols].round(3).to_string(index=False))

# --- chosen spec: EWMA hl 63d, cap 5x, weekly ------------------------------------------------
CH = ("EWMA hl 63d", 5.0, True)
lev, r = keep[CH]
lev, r = lev.loc[start:], r.loc[start:]
pd.DataFrame({"leverage": lev, "position": lev * q.loc[lev.index], "pnl": r}).to_csv(DATA / "vol_target_daily.csv")
alt = {"roll 21d": keep[("roll 21d", 5.0, True)][1].loc[start:], "roll 252d": keep[("roll 252d", 5.0, True)][1].loc[start:]}

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9b9a95"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig = plt.figure(figsize=(14, 12), dpi=200)
gs = fig.add_gridspec(4, 2, width_ratios=[2, 1], height_ratios=[2, 1, 1, 1.1], hspace=0.45, wspace=0.22)
axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
for a in axs[1:]:
    a.sharex(axs[0])
axy, axg, axc = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1:3, 1]), fig.add_subplot(gs[3, :])
for a in axs + [axy, axg, axc]:
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ch = tab[(tab.estimator == CH[0]) & (tab.cap == CH[1]) & (tab["update"] == "weekly")].iloc[0]
cst = tab[tab.estimator == "constant (hindsight)"].iloc[0]
ax = axs[0]
for lab, rr, c, lw in (("Vol-targeted 10% (EWMA hl 63d, cap 5×, weekly)", r, INK, 2.2),
                       ("Constant leverage hitting 10% (hindsight)", cst_r := const_L * x.loc[start:], GRAY, 1.3),
                       ("Vol-targeted, rolling 21d estimator", alt["roll 21d"], ORANGE, 1),
                       ("Vol-targeted, rolling 252d estimator", alt["roll 252d"], BLUE, 1)):
    eq = (1 + rr).cumprod()
    sh = rr.mean() / rr.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    ax.plot(eq.index, eq, color=c, lw=lw, label=f"{lab}: Sharpe {sh:.2f}, vol {rr.std() * np.sqrt(252):.1%}, max DD {dd:.0%}")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×" if v in (1.5, 3, 5, 7) else ""))
ax.axhline(1, color=MUTED, lw=1)
ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK2)
ax.set_title("Final core (macro signals + 3-month trend filter), levered to a 10% vol target (no costs, no financing)",
             loc="left", fontsize=9, color=INK2)
ax = axs[1]
rv = r.rolling(63).std() * np.sqrt(252) * 100
ax.plot(rv.index, rv, color=INK, lw=1.2, label="realised 63d vol, vol-targeted")
rvc = cst_r.rolling(63).std() * np.sqrt(252) * 100
ax.plot(rvc.index, rvc, color=GRAY, lw=1, label="realised 63d vol, constant leverage")
ax.axhspan(7, 13, color=AQUA, alpha=0.1, lw=0)
ax.axhline(10, color=AQUA, lw=1.2, ls="--")
ax.set_ylabel("Vol (%)")
ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2)
ax.set_title(f"Realised vol: within 7–13% {ch['63d vol within 7–13%']:.0%} of the time "
             f"(constant leverage {cst['63d vol within 7–13%']:.0%})", loc="left", fontsize=9, color=INK2)
ax = axs[2]
ax.plot(lev.index, lev, color=VIOLET, lw=1.1, label="leverage")
ax.plot(lev.index, (lev * q.loc[lev.index]), color=INK, lw=0.9, alpha=0.7, label="levered position (× SMLL−IBOV notional)")
ax.axhline(0, color=MUTED, lw=1)
ax.axhline(CH[1], color=MUTED, lw=1, ls=":")
ax.set_ylim(-2.5, 6.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2)
ax.set_title(f"Leverage on the unlevered strategy: average {ch['Avg leverage']:.1f}×, at the {CH[1]:.0f}× cap "
             f"{ch['Days at cap']:.0%} of days", loc="left", fontsize=9, color=INK2)
yv = r.groupby(r.index.year).std() * np.sqrt(252) * 100
yc = cst_r.groupby(cst_r.index.year).std() * np.sqrt(252) * 100
ax = axy
ax.barh(yv.index + 0.2, yv.values, height=0.4, color=INK, label="vol-targeted", zorder=3)
ax.barh(yc.index - 0.2, yc.values, height=0.4, color=GRAY, label="constant leverage", zorder=3)
ax.axvline(10, color=AQUA, lw=1.2, ls="--")
ax.invert_yaxis()
ax.set_yticks(yv.index[::2])
ax.set_yticklabels([str(y) for y in yv.index[::2]])
ax.set_xlabel("Realised vol by calendar year (%)")
ax.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=INK2)
ax.set_title("Vol by year", loc="left", fontsize=9, color=INK2)
ax = axg
g = tab[(tab.cap == 5.0) & (tab["update"] == "weekly")].set_index("estimator")
ests = list(EST)
yb = np.arange(len(ests))[::-1]
ax.scatter(g.loc[ests, "Realised vol"] * 100, yb + 0.15, color=INK, s=40, zorder=3, label="realised vol (%)")
ax.scatter(g.loc[ests, "Sharpe"] * 10, yb - 0.15, color=ORANGE, marker="D", s=30, zorder=3, label="Sharpe × 10")
ax.scatter(-g.loc[ests, "MaxDD"] * 100, yb, color=BLUE, marker="s", s=30, zorder=3, label="max drawdown (%)")
ax.axvline(10, color=AQUA, lw=1, ls="--")
ax.set_yticks(yb)
ax.set_yticklabels(ests)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=7.5, labelcolor=INK2, ncol=3)
ax.set_title("Estimator choice (cap 5×, weekly updates)", loc="left", fontsize=9, color=INK2)
ax = axc
ax.axis("off")
show = tab[((tab.cap == 5.0) & (tab["update"] == "weekly")) | tab.cap.isna()]
show = show[["estimator", "Realised vol", "Yearly vol min", "Yearly vol max", "Sharpe", "Sharpe H2", "CAGR", "MaxDD",
             "Worst 21d", "Avg leverage", "Days at cap"]]
cell = [[r_["estimator"]] + [f"{r_[c]:.1%}" if c in ("Realised vol", "Yearly vol min", "Yearly vol max", "CAGR", "MaxDD",
                                                    "Worst 21d", "Days at cap") else f"{r_[c]:.2f}" for c in show.columns[1:]]
        for _, r_ in show.iterrows()]
for row_ in cell:
    if row_[0] in ("constant (hindsight)", "unlevered"):
        row_[-1] = "–"
t = ax.table(cellText=cell, colLabels=["estimator", "vol", "yearly vol min", "yearly vol max", "Sharpe",
                                       "Sharpe H2", "CAGR", "max DD", "worst 21d", "avg lev.", "days at cap"],
             loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(8)
t.scale(1, 1.25)
ax.set_title("Cap 5×, weekly leverage updates (constant = one fixed leverage chosen with hindsight)", loc="left",
             fontsize=9, color=INK2)
for (i, j), c_ in t.get_celld().items():
    c_.set_edgecolor(GRID)
    if i == 0:
        c_.set_text_props(color=INK2, fontweight="bold")
fig.suptitle("Volatility targeting the final core at 10%, using the strategy's own past P&L volatility", x=0.04,
             ha="left", fontsize=14, fontweight="bold")
fig.text(0.04, 0.955, f"Leverage = 10% / predicted vol (past daily P&L, up to the previous close), capped; sample from "
         f"{start:%b %Y}. Returns exclude financing and trading costs.", fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.03)
fig.savefig(HERE / "vol_target.png")
print("saved")
