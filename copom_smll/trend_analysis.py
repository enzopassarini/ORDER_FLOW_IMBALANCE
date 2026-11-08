"""
Trend / momentum analysis of the SMLL - IBOV long-short return itself.

Spread index: cumulative product of (1 + r_SMLL - r_IBOV), daily (the same daily-rebalanced
dollar-neutral spread every backtest here trades).

Angles:
  1. Autocorrelation structure: variance ratios VR(q) = Var(q-day return) / (q * Var(1-day)),
     full sample and halves (VR > 1 = trending, < 1 = mean-reverting).
  2. Predictive grid: past L-session spread return (L = 5 ... 504) vs. forward H-session
     return (H = 5, 21, 63) from the next session; weekly samples, Newey-West t-stats
     (lags = H/5 + 1), full sample and halves.
  3. Other trend measures vs. fwd 21: price vs. 50/200-day moving average, 50/200 crossover,
     distance from the 1-year high, and a 1-week reversal.
  4. Trend-following backtests (weekly rebalance from Monday close, no costs): sign and
     linear time-series momentum at 3, 6 and 12 months, 200-day MA filter; correlation and
     combination with the core aggregate (COPOM + GDP + real rate).
  5. Conditioning: does the core signal work better when the trend agrees with it?
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
r = (px["SMLL"].pct_change() - px["IBOV"].pct_change()).dropna()
idx = (1 + r).cumprod()
lidx = np.log(idx)
mid_date = r.index[len(r) // 2]
print(f"spread {r.index[0].date()} -> {r.index[-1].date()}, split {mid_date.date()}; "
      f"total {idx.iloc[-1] - 1:+.1%}, ann. vol {r.std() * np.sqrt(252):.1%}")

# --- 1. variance ratios ---------------------------------------------------------------------
QS = [2, 5, 10, 21, 63, 126, 252]


def vr(x, q):
    lx = np.log1p(x)
    agg = lx.rolling(q).sum().dropna()
    return agg.var() / (q * lx.var())


vrt = pd.DataFrame({nm: [vr(p, q) for q in QS] for nm, p in
                    (("Full", r), ("First half", r[r.index < mid_date]), ("Second half", r[r.index >= mid_date]))},
                   index=QS)
vrt.index.name = "q (sessions)"
vrt.to_csv(DATA / "trend_variance_ratios.csv", float_format="%.3f")
print("\nVariance ratios:\n", vrt.round(2).to_string())
ac = pd.Series({k: np.log1p(r).autocorr(k) for k in (1, 2, 3, 4, 5)})
print("daily autocorrelations:", ac.round(3).to_dict())

# --- 2. predictive grid ----------------------------------------------------------------------
fridays = pd.date_range(r.index[0], r.index[-1], freq="W-FRI")
LB = [5, 10, 21, 42, 63, 126, 189, 252, 504]
HZ = [5, 21, 63]
rows = []
for d in fridays:
    i = idx.index.searchsorted(d, side="right") - 1          # Friday (or last session before)
    if i < 0 or i + 1 >= len(idx):
        continue
    row = {"date": d}
    for L in LB:
        row[f"past{L}"] = idx.iloc[i] / idx.iloc[i - L] - 1 if i - L >= 0 else np.nan
    e0 = i + 1                                                # Monday close = entry
    for H in HZ:
        row[f"fwd{H}"] = idx.iloc[e0 + H] / idx.iloc[e0] - 1 if e0 + H < len(idx) else np.nan
    row["ma50"] = lidx.iloc[i] - lidx.iloc[max(0, i - 49):i + 1].mean() if i >= 49 else np.nan
    row["ma200"] = lidx.iloc[i] - lidx.iloc[max(0, i - 199):i + 1].mean() if i >= 199 else np.nan
    row["cross"] = (lidx.iloc[i - 49:i + 1].mean() - lidx.iloc[i - 199:i + 1].mean()) if i >= 199 else np.nan
    row["from_high"] = idx.iloc[i] / idx.iloc[max(0, i - 251):i + 1].max() - 1 if i >= 251 else np.nan
    rows.append(row)
ev = pd.DataFrame(rows).set_index("date")
ev.to_csv(DATA / "trend_events.csv")
half = ev.index[len(ev) // 2]


def nwt(y, x, lags):
    d = pd.concat([y, x], axis=1).dropna()
    m = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.tvalues.iloc[1], np.corrcoef(d.iloc[:, 1], d.iloc[:, 0])[0, 1], len(d)


grid = []
for H in HZ:
    for L in LB:
        for nm, p in (("Full", ev), ("H1", ev[ev.index < half]), ("H2", ev[ev.index >= half])):
            t, c, n = nwt(p[f"fwd{H}"], p[f"past{L}"], H // 5 + 1)
            grid.append({"H": H, "L": L, "sample": nm, "t": t, "corr": c, "n": n})
grid = pd.DataFrame(grid)
grid.to_csv(DATA / "trend_predictive_grid.csv", index=False, float_format="%.3f")
print("\nNW t of past L-session return predicting fwd H (rows L, cols H x sample):")
print(grid.pivot_table(index="L", columns=["H", "sample"], values="t").round(2).to_string())

other = []
for col, lab in (("ma50", "price vs 50d MA"), ("ma200", "price vs 200d MA"), ("cross", "50d vs 200d MA"),
                 ("from_high", "distance from 1y high"), ("past5", "1-week return (reversal?)")):
    out = {"measure": lab}
    for nm, p in (("Full", ev), ("H1", ev[ev.index < half]), ("H2", ev[ev.index >= half])):
        out[f"t_{nm}"], out[f"corr_{nm}"], _ = nwt(p["fwd21"], p[col], 5)
    other.append(out)
other = pd.DataFrame(other).set_index("measure")
other.to_csv(DATA / "trend_other_measures.csv", float_format="%.3f")
print("\nOther trend measures vs fwd 21:\n", other.round(2).to_string())

# --- 4. backtests ----------------------------------------------------------------------------
core = pd.read_csv(DATA / "aggregate3_daily.csv", index_col=0, parse_dates=True)["Aggregate (3)"]


def weekly_pos(sig_fri):
    p = pd.Series(np.nan, index=r.index)
    for d, v in sig_fri.dropna().items():
        i0 = px.index.searchsorted(d, side="right")
        if i0 < len(px):
            p.loc[p.index > px.index[i0]] = v
    return p


strats = {}
for L, lab in ((63, "3m"), (126, "6m"), (252, "12m")):
    x = ev[f"past{L}"]
    strats[f"TSMOM {lab} sign"] = weekly_pos(np.sign(x))
    strats[f"TSMOM {lab} linear"] = weekly_pos((x / (2 * x.expanding(min_periods=52).std().shift(1))).clip(-1, 1))
strats["200d MA filter"] = weekly_pos(np.sign(ev["ma200"]))
strats["1-week reversal"] = weekly_pos((-ev["past5"] / (2 * ev["past5"].expanding(min_periods=52).std().shift(1))).clip(-1, 1))
pos = pd.DataFrame(strats)
pos["Core"] = core
pos = pos.dropna()
ret = pos.mul(r.loc[pos.index], axis=0)


def perf(x):
    eq = (1 + x).cumprod()
    return {"CAGR": eq.iloc[-1] ** (252 / len(x)) - 1, "Vol": x.std() * np.sqrt(252),
            "Sharpe": x.mean() / x.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min()}


m_ = len(ret) // 2
bt = pd.DataFrame({k: {**perf(ret[k]), "Sharpe H1": perf(ret[k].iloc[:m_])["Sharpe"],
                       "Sharpe H2": perf(ret[k].iloc[m_:])["Sharpe"],
                       "corr w/ core": ret[k].corr(ret["Core"]), "pos corr w/ core": pos[k].corr(pos["Core"])}
                   for k in ret}).T
# equal-risk mix of core + 12m TSMOM linear
best = "TSMOM 12m linear"
vol = ret[["Core", best]].expanding(min_periods=252).std().shift(1)
inv = 1 / vol
w = inv.div(inv.sum(axis=1), axis=0).fillna(0.5)
mix = (w * pos[["Core", best]]).sum(axis=1) * r.loc[pos.index]
bt.loc[f"Core + {best}"] = {**perf(mix), "Sharpe H1": perf(mix.iloc[:m_])["Sharpe"],
                            "Sharpe H2": perf(mix.iloc[m_:])["Sharpe"], "corr w/ core": mix.corr(ret["Core"]),
                            "pos corr w/ core": np.nan}
bt.to_csv(DATA / "trend_backtests.csv", float_format="%.4f")
pd.options.display.width = 200
print(f"\nBacktests {ret.index[0].date()} -> {ret.index[-1].date()}:\n", bt.round(2).to_string())

# --- 5. conditioning -------------------------------------------------------------------------
agg_ev = pd.read_csv(DATA / "aggregate3_fwd21.csv", index_col=0, parse_dates=True)
cj = agg_ev[["Aggregate (3)", "fwd21"]].join(ev[["past252", "past126"]], how="inner").dropna()
cj["agree"] = np.sign(cj["Aggregate (3)"]) == np.sign(cj["past126"])
cond = []
for nm, p in (("trend agrees (6m)", cj[cj["agree"]]), ("trend disagrees (6m)", cj[~cj["agree"]])):
    t, c, n = nwt(p["fwd21"] * 100, p["Aggregate (3)"], 5)
    cond.append({"subset": nm, "n": n, "t_core": t, "corr": c,
                 "mean core-signed fwd21 (pp)": (np.sign(p["Aggregate (3)"]) * p["fwd21"]).mean() * 100})
t, c, n = nwt(cj["fwd21"] * 100, pd.concat([cj["Aggregate (3)"], cj["past126"]], axis=1), 5) if False else (np.nan,) * 3
mj = sm.OLS(cj["fwd21"] * 100, sm.add_constant(cj[["Aggregate (3)", "past126"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
cond = pd.DataFrame(cond).set_index("subset")
cond.to_csv(DATA / "trend_conditioning.csv", float_format="%.3f")
print("\nCore signal conditional on 6m trend:\n", cond.round(2).to_string())
print("joint core + 6m trend NW t:", mj.tvalues.round(2).to_dict())

# --- plots -----------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9b9a95"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(a, yonly=False):
    a.grid(True, axis="y" if yonly else "both", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


fig = plt.figure(figsize=(14, 10), dpi=200)
gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1], hspace=0.45, wspace=0.28)
ax = fig.add_subplot(gs[0, :])
style(ax)
ax.plot(idx.index, idx, color=INK, lw=1.3, label="SMLL − IBOV spread index")
ax.plot(idx.index, np.exp(lidx.rolling(200).mean()), color=ORANGE, lw=1.2, label="200-day moving average")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_yticks([0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.5])
ax.axvline(mid_date, color=MUTED, lw=1, ls="--")
ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
ax.set_title(f"Spread index (log): total {idx.iloc[-1] - 1:+.0%} since 2005, vol {r.std() * np.sqrt(252):.1%}/yr. "
             "Long multi-year trends: 2005–07 up, 2008 crash, 2010–15 down, 2016–21 up, 2021–26 down.",
             loc="left", fontsize=8.5, color=INK2)
# variance ratios
ax = fig.add_subplot(gs[1, 0])
style(ax)
for nm, c in (("Full", INK), ("First half", BLUE), ("Second half", ORANGE)):
    ax.plot(vrt.index, vrt[nm], marker="o", color=c, lw=1.5, ms=4, label=nm)
ax.axhline(1, color=MUTED, lw=1)
ax.set_xscale("log")
ax.set_xticks(QS)
ax.set_xticklabels(QS)
ax.set_xlabel("Horizon q (sessions)")
ax.set_title("Variance ratio VR(q)  (>1 trending, <1 mean-reverting)", loc="left", fontsize=8.5, color=INK2)
ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
# predictive heatmaps
for j, (H, nm) in enumerate(((21, "Full"), (21, "H1"), (21, "H2"))):
    pass
ax = fig.add_subplot(gs[1, 1:])
style(ax)
ax.grid(False)
hm = grid.pivot_table(index="L", columns=["H", "sample"], values="t")
im = ax.imshow(hm.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ax.set_xticks(range(hm.shape[1]))
ax.set_xticklabels([f"fwd {h}\n{s}" for h, s in hm.columns], fontsize=7.5)
ax.set_yticks(range(len(hm)))
ax.set_yticklabels([f"past {L}" for L in hm.index], fontsize=7.5)
for i in range(hm.shape[0]):
    for jj in range(hm.shape[1]):
        v = hm.values[i, jj]
        ax.text(jj, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 2 else INK)
for x0 in (2.5, 5.5):
    ax.axvline(x0, color=SURF, lw=3)
ax.set_title("Newey-West t: past L-session spread return → forward H-session return (red = momentum, blue = reversal)",
             loc="left", fontsize=8.5, color=INK2)
# backtests
ax = fig.add_subplot(gs[2, :2])
style(ax)
show = {"TSMOM 6m linear": BLUE, "TSMOM 12m linear": VIOLET, "200d MA filter": ORANGE, "1-week reversal": GRAY,
        "Core": AQUA, f"Core + {best}": INK}
for k, c in show.items():
    x = mix if k.startswith("Core +") else ret[k]
    eq = (1 + x).cumprod()
    ax.plot(eq.index, eq, color=c, lw=2 if k.startswith("Core") else 1.1,
            label=f"{k}: Sharpe {bt.loc[k, 'Sharpe']:.2f} (H1 {bt.loc[k, 'Sharpe H1']:.2f}, H2 {bt.loc[k, 'Sharpe H2']:.2f})")
ax.axhline(1, color=MUTED, lw=1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.legend(loc="upper left", frameon=False, fontsize=7, labelcolor=INK2)
ax.set_title("Trend-following backtests on the spread (weekly, Monday close, no costs) vs. the core. "
             "Sign strategies run at ~13% vol, linear ~7%, core ~4%.",
             loc="left", fontsize=8.5, color=INK2)
ax = fig.add_subplot(gs[2, 2])
style(ax, True)
cc = cond["mean core-signed fwd21 (pp)"]
ax.bar([0, 1], cc.values, color=[AQUA, ORANGE], width=0.6, zorder=3)
for x0, (k, v) in enumerate(cc.items()):
    ax.text(x0, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f} pp\nn={int(cond.loc[k, 'n'])}, t {cond.loc[k, 't_core']:+.2f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
ax.set_ylim(0, cc.max() * 1.45)
ax.set_xticks([0, 1])
ax.set_xticklabels(["6m trend agrees\nwith core", "6m trend\ndisagrees"], fontsize=8)
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylabel("Mean fwd 21 return in the\ncore's direction (pp)")
ax.set_title("Core signal conditional on trend", loc="left", fontsize=8.5, color=INK2)
fig.suptitle("Trend analysis of the SMLL − IBOV long-short return", x=0.04, ha="left", fontsize=14, fontweight="bold")
fig.text(0.04, 0.945, "Daily-rebalanced spread, Sep 2005 – Nov 2026; weekly samples from the Monday close; halves split at "
         f"{mid_date:%b %Y}.", fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.05)
fig.savefig(HERE / "trend_analysis.png")
print("saved")
