"""
Compare three SMLL - IBOV signals and their equal-risk aggregate (no costs).

  COPOM      linear COPOM position, -clip(dSelic, +-100 bp)/100, held decision to decision
  GDP        Focus next-year GDP revisions, clip(rev, +-0.75 pp)/0.75, 4 staggered 21-day tranches
  Real rate  Focus ex-ante real rate vs. its rolling 5-year mean, clip(x / 3 sigma, +-1),
             4 staggered 21-day tranches: each Friday survey opens signal/4 at the Monday close,
             held 4 weeks, so position = mean of the last 4 weekly signals

Aggregate position = sum_i w_i,t * position_i,t with sum w = 1 (stays in [-1, 1]);
w_i proportional to 1 / sigma_i, sigma_i = vol of strategy i's daily returns on all days
before t (expanding, min 252 days; equal weights before that). Also reported: the
GDP + real-rate aggregate without COPOM.

Outputs: aggregate3_backtest.png, aggregate3_signal_fwd21.png and CSVs in data/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
H, NQ, NW_LAGS = 21, 10, 5

px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
spread = (px["SMLL"].pct_change() - px["IBOV"].pct_change()).dropna()
copom = pd.read_csv(DATA / "backtest_daily.csv", index_col=0, parse_dates=True)["position"]
gdp = pd.read_csv(DATA / "focus_tranches_daily.csv", index_col=0, parse_dates=True)["4 tranches x 21d"]
rr_w = pd.read_csv(DATA / "focus_channels_signal.csv", index_col=0, parse_dates=True)["real_rate_r5"].dropna()
rr_w = rr_w.rolling(4).mean().dropna()  # 4 tranches x 21 days
rr = pd.Series(np.nan, index=spread.index)
for d, v in rr_w.items():
    i0 = px.index.searchsorted(d, side="right")
    if i0 < len(px):
        rr.loc[rr.index > px.index[i0]] = v

pos = pd.DataFrame({"COPOM": copom, "GDP (tranches)": gdp, "Real rate (tranches)": rr}).dropna()
sp = spread.loc[pos.index]
ret = pos.mul(sp, axis=0)


def aggregate(cols):
    vol = ret[cols].expanding(min_periods=252).std().shift(1)
    inv = 1 / vol
    w = inv.div(inv.sum(axis=1), axis=0).fillna(1 / len(cols))
    return (w * pos[cols]).sum(axis=1), w


pos["Aggregate (3)"], w3 = aggregate(["COPOM", "GDP (tranches)", "Real rate (tranches)"])
pos["GDP + Real rate"], _ = aggregate(["GDP (tranches)", "Real rate (tranches)"])
ret = pos.mul(sp, axis=0)
pos.join(ret, rsuffix=" ret").join(w3.add_prefix("w ")).to_csv(DATA / "aggregate3_daily.csv")


def perf(r):
    eq = (1 + r).cumprod()
    return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "Vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min()}


mid = len(ret) // 2
STRATS = ["COPOM", "GDP (tranches)", "Real rate (tranches)", "GDP + Real rate", "Aggregate (3)"]
tab = pd.DataFrame({k: {**perf(ret[k]), "Sharpe H1": perf(ret[k].iloc[:mid])["Sharpe"],
                        "Sharpe H2": perf(ret[k].iloc[mid:])["Sharpe"], "avg |pos|": pos[k].abs().mean()}
                    for k in STRATS}).T
tab.to_csv(DATA / "aggregate3_stats.csv", float_format="%.4f")
cor = ret[["COPOM", "GDP (tranches)", "Real rate (tranches)"]].corr()
pcor = pos[["COPOM", "GDP (tranches)", "Real rate (tranches)"]].corr()
cor.to_csv(DATA / "aggregate3_return_corr.csv", float_format="%.3f")
pd.options.display.width = 200
print(f"{ret.index[0].date()} -> {ret.index[-1].date()}, split {ret.index[mid].date()}")
print(tab.round(3).to_string())
print("\nDaily return correlations:\n", cor.round(2).to_string())
print("\nPosition correlations:\n", pcor.round(2).to_string())
print("\nAverage weights (3-signal):", w3.mean().round(2).to_dict(), "last:", w3.iloc[-1].round(2).to_dict())

# --- weekly signal vs fwd 21-session return, all signals -------------------------------------
fridays = rr_w.index
rows = []
for d in fridays:
    i0 = px.index.searchsorted(d, side="right")
    if i0 + H >= len(px):
        continue
    day1 = px.index[i0 + 1]
    if day1 not in pos.index:
        continue
    r = px.iloc[i0 + H] / px.iloc[i0] - 1
    rows.append({"date": d, **pos.loc[day1, STRATS].to_dict(), "fwd21": r["SMLL"] - r["IBOV"]})
ev = pd.DataFrame(rows).set_index("date")
ev.to_csv(DATA / "aggregate3_fwd21.csv")
half = len(ev) // 2
samples = {"Full sample": ev, f"First half ({ev.index[0]:%b %Y} – {ev.index[half - 1]:%b %Y})": ev.iloc[:half],
           f"Second half ({ev.index[half]:%b %Y} – {ev.index[-1]:%b %Y})": ev.iloc[half:]}
st = []
for nm, e in samples.items():
    for k in STRATS:
        m = sm.OLS(e["fwd21"] * 100, sm.add_constant(e[k])).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
        st.append({"sample": nm, "signal": k, "n": len(e), "slope": m.params[k], "t_NW": m.tvalues[k],
                   "R2": m.rsquared})
st = pd.DataFrame(st)
st.to_csv(DATA / "aggregate3_fwd21_stats.csv", index=False, float_format="%.4f")
print("\nNW t-stats, fwd 21-session:\n", st.pivot(index="signal", columns="sample", values="t_NW")
      .reindex(STRATS)[list(samples)].round(2).to_string())

# --- plotting ---------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COL = {"COPOM": "#2a78d6", "GDP (tranches)": "#1baf7a", "Real rate (tranches)": "#eb6834", "GDP + Real rate": "#9b9a95",
       "Aggregate (3)": INK}
AQUA, ORANGE = "#1baf7a", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(a, yonly=False):
    a.grid(True, axis="y" if yonly else "both", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


eq = (1 + ret[STRATS]).cumprod()
fig, (ax, axp, axd) = plt.subplots(3, 1, figsize=(10.5, 10), dpi=200, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.7, 1.1], "hspace": 0.1})
for a in (ax, axp, axd):
    style(a)
    a.axvline(ret.index[mid], color=MUTED, lw=1, ls="--")
ends = []
for k in STRATS:
    t = tab.loc[k]
    ax.plot(eq.index, eq[k], color=COL[k], lw=2.4 if k == "Aggregate (3)" else 1.3,
            ls="--" if k == "GDP + Real rate" else "-",
            label=f"{k}: Sharpe {t.Sharpe:.2f} (H1 {t['Sharpe H1']:.2f}, H2 {t['Sharpe H2']:.2f}), max DD {t.MaxDD:.0%}")
    ends.append((eq[k].iloc[-1], k))
prev = None
for v, k in sorted(ends):
    yv = v if prev is None or v - prev > 0.04 else prev + 0.04
    ax.annotate(f"{v:.2f}×", (eq.index[-1], yv), xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                color=INK2, va="center")
    prev = yv
ax.axhline(1, color=MUTED, lw=1)
ax.set_ylabel("Growth of 1")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
ax.annotate("half split", (ret.index[mid], 1), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
for k in ["COPOM", "GDP (tranches)", "Real rate (tranches)"]:
    axp.plot(pos.index, pos[k], color=COL[k], lw=0.8, alpha=0.65, drawstyle="steps-post", label=k)
axp.fill_between(pos.index, 0, pos["Aggregate (3)"], color=INK, alpha=0.15, lw=0, step="post")
axp.plot(pos.index, pos["Aggregate (3)"], color=INK, lw=1.4, drawstyle="steps-post", label="Aggregate (3)")
axp.axhline(0, color=MUTED, lw=1)
axp.set_ylim(-1.05, 1.4)
axp.set_yticks([-1, -0.5, 0, 0.5, 1])
axp.set_ylabel("Position\n(+ = long SMLL / short IBOV)")
axp.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2, ncol=4)
dd = (eq / eq.cummax() - 1) * 100
for k in STRATS:
    axd.plot(dd.index, dd[k], color=COL[k], lw=1.4 if k == "Aggregate (3)" else 0.7,
             ls="--" if k == "GDP + Real rate" else "-")
axd.set_ylabel("Drawdown (%)")
fig.suptitle("COPOM, GDP and real rate (tranches) vs. aggregate: SMLL − IBOV, no costs",
             x=0.06, ha="left", fontsize=13, fontweight="bold", y=0.975)
c_ = cor
ax.set_title(f"Aggregate = Σ wᵢ · positionᵢ, wᵢ ∝ 1/past vol (avg weights: COPOM {w3['COPOM'].mean():.2f}, "
             f"GDP {w3['GDP (tranches)'].mean():.2f}, real rate {w3['Real rate (tranches)'].mean():.2f}).\nDaily return correlations: "
             f"COPOM–GDP {c_.loc['COPOM', 'GDP (tranches)']:+.2f}, COPOM–real rate {c_.loc['COPOM', 'Real rate (tranches)']:+.2f}, "
             f"GDP–real rate {c_.loc['GDP (tranches)', 'Real rate (tranches)']:+.2f}.", loc="left", fontsize=8.5, color=INK2, pad=10)
fig.text(0.06, 0.01, "Return = position × (r_SMLL − r_IBOV), daily-rebalanced dollar-neutral spread. "
         "Sources: BCB Focus, BCB COPOM, B3.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.1, right=0.93, top=0.89, bottom=0.06)
fig.savefig(HERE / "aggregate3_backtest.png")
plt.close(fig)

# signal plot for the 3-signal aggregate
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), dpi=200, gridspec_kw={"height_ratios": [1.25, 1]})
K = "Aggregate (3)"
for j, (nm, e) in enumerate(samples.items()):
    s = st[(st["sample"] == nm) & (st["signal"] == K)].iloc[0]
    comp = {k: st[(st["sample"] == nm) & (st["signal"] == k)].iloc[0].t_NW for k in ["COPOM", "GDP (tranches)", "Real rate (tranches)"]}
    x, y = e[K].to_numpy(), e["fwd21"].to_numpy() * 100
    ax = axes[0, j]
    style(ax)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.scatter(x, y, s=12, color="#6250d6", alpha=0.45, lw=0, zorder=3)
    b, a_ = np.polyfit(x, y, 1)
    xs = np.linspace(-1, 1, 20)
    ax.plot(xs, a_ + b * xs, color=INK, lw=2, zorder=4)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-16, 16)
    nz = x != 0
    hit = (np.sign(x[nz]) == np.sign(y[nz])).mean()
    ax.set_title(f"{nm}\nn={int(s.n)}  slope {s.slope:+.2f} pp  NW t {s.t_NW:+.2f}  R² {s.R2:.3f}  hit {hit:.0%}\n"
                 f"components NW t: COPOM {comp['COPOM']:+.2f}, GDP {comp['GDP (tranches)']:+.2f}, real rate "
                 f"{comp['Real rate (tranches)']:+.2f}", loc="left", fontsize=8.5, color=INK2)
    ax.set_xlabel("Aggregate signal  ← short small caps · long small caps →")
    if j == 0:
        ax.set_ylabel("SMLL − IBOV fwd 21-session return (pp)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    q = pd.qcut(e[K].rank(method="first"), NQ, labels=range(1, NQ + 1)).astype(int)
    g = e.assign(q=q).groupby("q").agg(n=("fwd21", "size"), lo=(K, "min"), hi=(K, "max"), sig=(K, "mean"),
                                       mean=("fwd21", "mean"), sd=("fwd21", "std"))
    g["mean"] *= 100
    g["se"] = g["sd"] * 100 / np.sqrt(g["n"]) * np.sqrt(H / 5)
    ax = axes[1, j]
    style(ax, True)
    ax.axhline(0, color=MUTED, lw=1)
    ax.bar(g.index, g["mean"], width=0.7, color=[AQUA if v > 0 else ORANGE for v in g["sig"]], zorder=3)
    ax.errorbar(g.index, g["mean"], yerr=g["se"], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
    for qi, r in g.iterrows():
        off = r["se"] + 0.15
        ax.text(qi, float(np.clip(r["mean"] + (off if r["mean"] >= 0 else -off), -4.6, 4.6)), f"{r['mean']:+.1f}",
                ha="center", va="bottom" if r["mean"] >= 0 else "top", fontsize=7)
    fmt = lambda v: f"{v:+.2f}" if abs(v) > 1e-9 else "0"
    ax.set_xticks(g.index)
    ax.set_xticklabels([f"D{qi}\n{fmt(r.lo)}\n{fmt(r.hi)}\nn={int(r.n)}" for qi, r in g.iterrows()], fontsize=6)
    ax.set_ylim(-5, 5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    if j == 0:
        ax.set_ylabel("Mean fwd 21-session return (pp)")
    ax.set_title(f"Deciles: D10 − D1 = {g.loc[NQ, 'mean'] - g.loc[1, 'mean']:+.2f} pp", loc="left",
                 fontsize=8.5, color=INK2)
fig.suptitle("Aggregate of COPOM + GDP revisions + real rate (tranches) vs. next 21-session small-cap relative return",
             x=0.05, ha="left", fontsize=13, fontweight="bold", y=0.985)
fig.text(0.05, 0.935, "Equal-risk weights from past volatility; weekly, as held from Monday close. Overlapping "
         "windows: Newey-West (5 lags); error bars ±1 SE × √(21/5).", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(HERE / "aggregate3_signal_fwd21.png")
print("saved plots")
