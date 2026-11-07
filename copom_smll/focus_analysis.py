"""
Focus GDP-revision signal vs. forward 21-session SMLL - IBOV return.
Non-overlapping windows, plain OLS, full sample and two halves. No backtest.

Signal construction (all steps printed by this script with a worked example):
  1. Weekly panel: for each Friday t and target year Y, M_t(Y) = the Focus median
     GDP-growth expectation for year Y on the last survey day on or before t
     (BCB Olinda, 'PIB Total', baseCalculo = 0).
  2. Target year: Y(t) = year(t) + 1.
  3. Revisions on the same target year (in pp of GDP growth):
        rev4(t) = M_t(Y) - M_{t-4w}(Y)
        rev8(t) = M_t(Y) - M_{t-8w}(Y)
  4. Raw feature: rev(t) = (rev4(t) + rev8(t)) / 2
  5. Scale: s(t) = 2 * std of rev over all weeks strictly before t (from 1999, min 52 weeks)
  6. Signal: x(t) = clip(rev(t) / s(t), -1, 1)      upgrade -> x > 0

Target: y(t) = SMLL return - IBOV return from the close of the first session after
Friday t (Monday, when the report is public) to the close 21 sessions later.

Non-overlapping sample: start at the first Friday; each next observation is the
first Friday whose entry session is on or after the previous observation's exit
session, so no two windows share a day of returns.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
H = 21

# --- 1-6: signal ---------------------------------------------------------------
w = pd.read_csv(DATA / "focus_gdp_weekly.csv", index_col=0, parse_dates=True)
w.columns = w.columns.astype(int)
f = pd.DataFrame(index=w.index)
f["Y"] = f.index.year + 1
f["M_t"] = [w.at[d, y] for d, y in zip(f.index, f["Y"])]
for k in (4, 8):
    lag_idx = np.arange(len(w)) - k
    f[f"M_t-{k}w"] = [w.iat[j, w.columns.get_loc(y)] if j >= 0 else np.nan for j, y in zip(lag_idx, f["Y"])]
    f[f"rev{k}"] = f["M_t"] - f[f"M_t-{k}w"]
f["rev"] = (f["rev4"] + f["rev8"]) / 2
f["scale"] = 2 * f["rev"].expanding(min_periods=52).std().shift(1)
f["raw_ratio"] = f["rev"] / f["scale"]
f["signal"] = f["raw_ratio"].clip(-1, 1)

# --- target and non-overlapping sample ----------------------------------------------
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
f = f[f.index >= px.index[0]].dropna(subset=["signal"])
rows, last_exit = [], None
for d in f.index:
    i0 = px.index.searchsorted(d, side="right")
    if i0 + H >= len(px):
        break
    if last_exit is not None and px.index[i0] < last_exit:
        continue
    r = px.iloc[i0 + H] / px.iloc[i0] - 1
    rows.append({"date": d, "entry": px.index[i0], "exit": px.index[i0 + H],
                 "SMLL_ret": r["SMLL"], "IBOV_ret": r["IBOV"], "fwd_rel": r["SMLL"] - r["IBOV"]})
    last_exit = px.index[i0 + H]
ev = f.join(pd.DataFrame(rows).set_index("date"), how="inner")
ev.to_csv(DATA / "focus_nonoverlap_fwd21.csv")

# --- print the transformation with a worked example -------------------------------------
pd.options.display.width = 220
pd.options.display.float_format = "{:.4f}".format
print("Signal steps for the last 6 observations in the non-overlapping sample:")
print(ev[["Y", "M_t", "M_t-4w", "rev4", "M_t-8w", "rev8", "rev", "scale", "raw_ratio", "signal",
          "entry", "exit", "fwd_rel"]].tail(6).to_string())
print(f"\nSignal distribution (n={len(ev)}): zero {(ev.signal == 0).mean():.0%}, "
      f"clipped at -1 {(ev.signal == -1).mean():.0%}, clipped at +1 {(ev.signal == 1).mean():.0%}, "
      f"mean {ev.signal.mean():+.3f}; scale s(t) ranged {ev.scale.min():.3f}-{ev.scale.max():.3f} pp")

# --- OLS by sample ---------------------------------------------------------------------
half = len(ev) // 2
samples = {
    "Full sample": ev,
    f"First half ({ev.index[0]:%b %Y} – {ev.index[half - 1]:%b %Y})": ev.iloc[:half],
    f"Second half ({ev.index[half]:%b %Y} – {ev.index[-1]:%b %Y})": ev.iloc[half:],
}
res = []
for name, e in samples.items():
    x, y = e["signal"], e["fwd_rel"] * 100
    m = sm.OLS(y, sm.add_constant(x)).fit()
    nz = x != 0
    res.append({"sample": name, "n": int(m.nobs), "intercept_pp": m.params["const"], "slope_pp": m.params["signal"],
                "se_slope": m.bse["signal"], "t": m.tvalues["signal"], "p": m.pvalues["signal"],
                "R2": m.rsquared, "corr": np.corrcoef(x, y)[0, 1],
                "hit": (np.sign(x[nz]) == np.sign(y[nz])).mean(),
                "mean_down": y[x < 0].mean(), "n_down": int((x < 0).sum()),
                "mean_zero": y[x == 0].mean(), "n_zero": int((x == 0).sum()),
                "mean_up": y[x > 0].mean(), "n_up": int((x > 0).sum())})
tab = pd.DataFrame(res).set_index("sample")
tab.to_csv(DATA / "focus_ols_halves.csv", float_format="%.4f")
pd.options.display.float_format = "{:.3f}".format
print("\nOLS: fwd 21-session SMLL-IBOV return (pp) on signal")
print(tab.to_string())

# --- plot: scatter row + quintile row -----------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
})
NQ = 5
fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), dpi=200, gridspec_kw={"height_ratios": [1.25, 1]})
qrows = []
for j, (name, e) in enumerate(samples.items()):
    st = tab.loc[name]
    x, y = e["signal"].to_numpy(), e["fwd_rel"].to_numpy() * 100
    ax = axes[0, j]
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.scatter(x, y, s=22, color=BLUE, alpha=0.65, edgecolor=SURF, lw=0.6, zorder=3)
    xs = np.linspace(-1, 1, 20)
    ax.plot(xs, st.intercept_pp + st.slope_pp * xs, color=INK, lw=2, zorder=4)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-16, 16)
    ax.set_title(f"{name}\nn={int(st.n)}  slope {st.slope_pp:+.2f} pp (SE {st.se_slope:.2f})  t {st.t:+.2f}  "
                 f"p {st.p:.3f}\nR² {st.R2:.3f}  ρ {st['corr']:+.2f}  hit {st.hit:.0%}",
                 loc="left", fontsize=8.5, color=INK2)
    ax.set_xlabel("Signal  ← downgrade · upgrade →")
    if j == 0:
        ax.set_ylabel("SMLL − IBOV fwd 21-session return (pp)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))

    # quintiles; ties (mostly zeros) split by date
    q = pd.qcut(e["signal"].rank(method="first"), NQ, labels=range(1, NQ + 1)).astype(int)
    g = e.assign(q=q).groupby("q").agg(n=("fwd_rel", "size"), lo=("signal", "min"), hi=("signal", "max"),
                                       zero=("signal", lambda v: (v == 0).mean()),
                                       sig=("signal", "mean"), mean=("fwd_rel", "mean"), sd=("fwd_rel", "std"))
    g["mean"] *= 100
    g["se"] = g["sd"] * 100 / np.sqrt(g["n"])
    qrows.append(g.assign(sample=name))
    ax = axes[1, j]
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cols = [GRAY if z > 0.5 else AQUA if s_ > 0 else ORANGE for s_, z in zip(g["sig"], g["zero"])]
    ax.bar(g.index, g["mean"], width=0.7, color=cols, zorder=3)
    ax.errorbar(g.index, g["mean"], yerr=g["se"], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    for qi, r in g.iterrows():
        off = r["se"] + 0.15
        ax.text(qi, r["mean"] + (off if r["mean"] >= 0 else -off), f"{r['mean']:+.2f}", ha="center",
                va="bottom" if r["mean"] >= 0 else "top", fontsize=8)
    fmt = lambda v: f"{v:+.2f}" if v != 0 else "0"
    ax.set_xticks(g.index)
    ax.set_xticklabels([f"Q{qi}\n{fmt(r.lo)}…{fmt(r.hi)}\nn={int(r.n)}" for qi, r in g.iterrows()], fontsize=7.5)
    ax.set_ylim(-4.5, 4.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    if j == 0:
        ax.set_ylabel("Mean fwd 21-session return (pp)")
    ax.set_title(f"Quintiles: Q5 − Q1 = {g.loc[NQ, 'mean'] - g.loc[1, 'mean']:+.2f} pp", loc="left",
                 fontsize=8.5, color=INK2)
pd.concat(qrows).drop(columns="sd").to_csv(DATA / "focus_quintiles_halves.csv", float_format="%.4f")

from matplotlib.patches import Patch
fig.legend([Patch(color=ORANGE), Patch(color=GRAY), Patch(color=AQUA)],
           ["Downgrade (signal < 0)", "Mostly no revision (0)", "Upgrade (signal > 0)"],
           loc="upper left", bbox_to_anchor=(0.045, 0.925), frameon=False, fontsize=8.5, ncol=3, labelcolor=INK2)
fig.suptitle("Focus GDP revisions vs. next 21-session small-cap relative return — non-overlapping, OLS",
             x=0.05, ha="left", fontsize=13, fontweight="bold", y=0.985)
fig.text(0.05, 0.935, "Signal = clip(((rev4 + rev8)/2) / (2 × past std), −1, 1), revisions to next-year GDP "
         "median on the same target year. One observation per 21-session window; standard OLS SEs. "
         "Error bars ±1 SE.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(HERE / "focus_ols_halves.png")
print("saved", HERE / "focus_ols_halves.png")
