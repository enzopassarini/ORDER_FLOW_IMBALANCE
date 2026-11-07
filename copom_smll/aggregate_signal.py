"""
Aggregate of the Focus GDP-revision signal (4 tranches x 21d) and the COPOM linear signal.

Aggregate position (daily):
    agg_t = w_F,t * focus_t + w_C,t * copom_t,   w_F + w_C = 1  ->  agg in [-1, 1]
Equal-risk weights: w_i proportional to 1 / sigma_i, where sigma_i is the volatility of
strategy i's daily returns over all days before t (expanding, min 252 days; 50/50 before
that). Causal, so no full-sample vol is used.

Outputs:
  aggregate_signal_fwd21.png  weekly aggregate signal (as held from Monday close) vs. next
                              21-session SMLL-IBOV return; full sample + halves, deciles,
                              Newey-West t-stats (windows overlap)
  aggregate_backtest.png      aggregate position over time and backtest vs. components
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
H, NQ, NW_LAGS = 21, 10, 5

f = pd.read_csv(DATA / "focus_tranches_daily.csv", index_col=0, parse_dates=True)
c = pd.read_csv(DATA / "backtest_daily.csv", index_col=0, parse_dates=True)
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
df = pd.DataFrame({"focus": f["4 tranches x 21d"], "copom": c["position"],
                   "focus_ret": f["4 tranches x 21d_ret"], "copom_ret": c["strategy_ret"],
                   "spread": c["spread_ret"]}).dropna()

vol = df[["focus_ret", "copom_ret"]].expanding(min_periods=252).std().shift(1)
inv = 1 / vol
df["w_focus"] = (inv["focus_ret"] / inv.sum(axis=1)).fillna(0.5)
df["w_copom"] = 1 - df["w_focus"]
df["agg"] = df["w_focus"] * df["focus"] + df["w_copom"] * df["copom"]
df["agg_ret"] = df["agg"] * df["spread"]
df.to_csv(DATA / "aggregate_daily.csv")


def perf(r):
    eq = (1 + r).cumprod()
    return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "Vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min()}


mid = len(df) // 2
names = {"agg_ret": "Aggregate (equal-risk)", "focus_ret": "Focus (4 tranches)", "copom_ret": "COPOM linear"}
tab = pd.DataFrame({lab: {**perf(df[k]), "Sharpe 1st half": perf(df[k].iloc[:mid])["Sharpe"],
                          "Sharpe 2nd half": perf(df[k].iloc[mid:])["Sharpe"]} for k, lab in names.items()}).T
tab["avg |pos|"] = [df["agg"].abs().mean(), df["focus"].abs().mean(), df["copom"].abs().mean()]
tab.round(3).to_csv(DATA / "aggregate_backtest_stats.csv")
pd.options.display.width = 200
print(f"Backtest {df.index[0].date()} -> {df.index[-1].date()}; split {df.index[mid].date()}")
print(tab.round(3))
print(f"weights: focus {df['w_focus'].iloc[300]:.2f} (2006-07) -> {df['w_focus'].iloc[-1]:.2f} (end); "
      f"mean {df['w_focus'].mean():.2f}")

# --- weekly samples: signal held from Monday close vs next 21 sessions ---------------------
fridays = pd.read_csv(DATA / "focus_signal_weekly.csv", index_col=0, parse_dates=True).index
rows = []
for d in fridays:
    i0 = px.index.searchsorted(d, side="right")
    if i0 + H >= len(px) or i0 + 1 >= len(px):
        continue
    day1 = px.index[i0 + 1]  # first day earning the position set at Monday close
    if day1 not in df.index:
        continue
    r = px.iloc[i0 + H] / px.iloc[i0] - 1
    rows.append({"date": d, "entry": px.index[i0], "agg": df.at[day1, "agg"],
                 "focus": df.at[day1, "focus"], "copom": df.at[day1, "copom"],
                 "fwd_rel": r["SMLL"] - r["IBOV"]})
ev = pd.DataFrame(rows).set_index("date")
ev.to_csv(DATA / "aggregate_fwd21.csv")
half = len(ev) // 2
samples = {"Full sample": ev,
           f"First half ({ev.index[0]:%b %Y} – {ev.index[half - 1]:%b %Y})": ev.iloc[:half],
           f"Second half ({ev.index[half]:%b %Y} – {ev.index[-1]:%b %Y})": ev.iloc[half:]}
res = []
for nm, e in samples.items():
    for col in ("agg", "focus", "copom"):
        x, y = e[col], e["fwd_rel"] * 100
        m = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
        nz = x != 0
        res.append({"sample": nm, "signal": col, "n": len(e), "slope_pp": m.params[col], "t_NW": m.tvalues[col],
                    "p_NW": m.pvalues[col], "R2": m.rsquared, "corr": np.corrcoef(x, y)[0, 1],
                    "hit": (np.sign(x[nz]) == np.sign(y[nz])).mean()})
st = pd.DataFrame(res)
st.round(4).to_csv(DATA / "aggregate_fwd21_stats.csv", index=False)
print("\nSignal vs fwd 21-session return (weekly, overlapping, Newey-West):")
print(st.round(3).to_string(index=False))

# --- plotting setup ------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, AQUA, ORANGE, VIOLET, GRAY = "#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#9b9a95"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(a, ygrid_only=False):
    a.grid(True, axis="y" if ygrid_only else "both", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


# --- figure 1: signal vs fwd 21 --------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), dpi=200, gridspec_kw={"height_ratios": [1.25, 1]})
for j, (nm, e) in enumerate(samples.items()):
    s = st[(st["sample"] == nm) & (st["signal"] == "agg")].iloc[0]
    x, y = e["agg"].to_numpy(), e["fwd_rel"].to_numpy() * 100
    ax = axes[0, j]
    style(ax)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.scatter(x, y, s=12, color=VIOLET, alpha=0.45, lw=0, zorder=3)
    b, a_ = np.polyfit(x, y, 1)
    xs = np.linspace(-1, 1, 20)
    ax.plot(xs, a_ + b * xs, color=INK, lw=2, zorder=4)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-16, 16)
    sf = st[(st["sample"] == nm) & (st["signal"] == "focus")].iloc[0]
    sc = st[(st["sample"] == nm) & (st["signal"] == "copom")].iloc[0]
    ax.set_title(f"{nm}\nn={s.n}  slope {s.slope_pp:+.2f} pp  NW t {s.t_NW:+.2f}  p {s.p_NW:.3f}  ρ {s['corr']:+.2f}  "
                 f"hit {s.hit:.0%}\ncomponents NW t: Focus {sf.t_NW:+.2f}, COPOM {sc.t_NW:+.2f}",
                 loc="left", fontsize=8.5, color=INK2)
    ax.set_xlabel("Aggregate signal  ← short small caps · long small caps →")
    if j == 0:
        ax.set_ylabel("SMLL − IBOV fwd 21-session return (pp)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))

    q = pd.qcut(e["agg"].rank(method="first"), NQ, labels=range(1, NQ + 1)).astype(int)
    g = e.assign(q=q).groupby("q").agg(n=("fwd_rel", "size"), lo=("agg", "min"), hi=("agg", "max"),
                                       sig=("agg", "mean"), zero=("agg", lambda v: (v.abs() < 1e-9).mean()),
                                       mean=("fwd_rel", "mean"), sd=("fwd_rel", "std"))
    g["mean"] *= 100
    g["se"] = g["sd"] * 100 / np.sqrt(g["n"]) * np.sqrt(H / 5)
    ax = axes[1, j]
    style(ax, True)
    ax.axhline(0, color=MUTED, lw=1)
    cols = [GRAY if z > 0.5 else AQUA if sg > 0 else ORANGE for sg, z in zip(g["sig"], g["zero"])]
    ax.bar(g.index, g["mean"], width=0.7, color=cols, zorder=3)
    ax.errorbar(g.index, g["mean"], yerr=g["se"], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
    for qi, r in g.iterrows():
        off = r["se"] + 0.15
        ax.text(qi, r["mean"] + (off if r["mean"] >= 0 else -off), f"{r['mean']:+.1f}", ha="center",
                va="bottom" if r["mean"] >= 0 else "top", fontsize=7)
    fmt = lambda v: f"{v:+.2f}" if abs(v) > 1e-9 else "0"
    ax.set_xticks(g.index)
    ax.set_xticklabels([f"D{qi}\n{fmt(r.lo)}\n{fmt(r.hi)}\nn={int(r.n)}" for qi, r in g.iterrows()], fontsize=6)
    ax.set_ylim(-5, 5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    if j == 0:
        ax.set_ylabel("Mean fwd 21-session return (pp)")
    ax.set_title(f"Deciles: D10 − D1 = {g.loc[NQ, 'mean'] - g.loc[1, 'mean']:+.2f} pp", loc="left",
                 fontsize=8.5, color=INK2)
from matplotlib.patches import Patch
fig.legend([Patch(color=ORANGE), Patch(color=GRAY), Patch(color=AQUA)],
           ["Short small caps (< 0)", "Mostly flat (0)", "Long small caps (> 0)"],
           loc="upper left", bbox_to_anchor=(0.045, 0.925), frameon=False, fontsize=8.5, ncol=3, labelcolor=INK2)
fig.suptitle("Aggregate signal (Focus + COPOM, equal-risk) vs. next 21-session small-cap relative return",
             x=0.05, ha="left", fontsize=13, fontweight="bold", y=0.985)
fig.text(0.05, 0.935, "Signal = w_F · Focus(4 tranches) + w_C · COPOM, weights ∝ 1/past vol of each strategy, "
         "sampled weekly as held from Monday close. Overlapping windows: Newey-West (5 lags) t-stats; "
         "error bars ±1 SE × √(21/5).", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(HERE / "aggregate_signal_fwd21.png")
plt.close(fig)

# --- figure 2: positions and backtest ---------------------------------------------------------
plt.rcParams.update({"font.size": 10})
eq = (1 + df[list(names)]).cumprod()
fig, (ax, axp, axd) = plt.subplots(3, 1, figsize=(10, 9), dpi=200, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.6, 1.1], "hspace": 0.1})
for a in (ax, axp, axd):
    style(a)
    a.axvline(df.index[mid], color=MUTED, lw=1, ls="--")
cols = {"agg_ret": INK, "focus_ret": AQUA, "copom_ret": BLUE}
ends = []
for k, lab in names.items():
    t = tab.loc[lab]
    ax.plot(eq.index, eq[k], color=cols[k], lw=2.3 if k == "agg_ret" else 1.3,
            label=f"{lab}  (Sharpe {t.Sharpe:.2f}; 1st half {t['Sharpe 1st half']:.2f}, 2nd {t['Sharpe 2nd half']:.2f})")
    ends.append((eq[k].iloc[-1], k))
for rank, (v, k) in enumerate(sorted(ends)):
    ax.annotate(f"{v:.2f}×", (eq.index[-1], v), xytext=(6, (rank - 1) * 7), textcoords="offset points",
                fontsize=9, color=INK2, va="center")
ax.axhline(1, color=MUTED, lw=1)
ax.set_ylabel("Growth of 1")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
ax.annotate("half split", (df.index[mid], 1), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
axp.plot(df.index, df["copom"], color=BLUE, lw=0.9, alpha=0.6, drawstyle="steps-post", label="COPOM")
axp.plot(df.index, df["focus"], color=AQUA, lw=0.9, alpha=0.7, label="Focus")
axp.fill_between(df.index, 0, df["agg"], color=INK, alpha=0.18, lw=0, step="post")
axp.plot(df.index, df["agg"], color=INK, lw=1.4, drawstyle="steps-post", label="Aggregate")
axp.axhline(0, color=MUTED, lw=1)
axp.set_ylim(-1.05, 1.35)
axp.set_yticks([-1, -0.5, 0, 0.5, 1])
axp.set_ylabel("Position\n(+ = long SMLL / short IBOV)")
axp.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2, ncol=3)
dd = (eq / eq.cummax() - 1) * 100
for k in names:
    axd.plot(dd.index, dd[k], color=cols[k], lw=1.3 if k == "agg_ret" else 0.8)
axd.set_ylabel("Drawdown (%)")
t = tab.loc["Aggregate (equal-risk)"]
fig.suptitle("Aggregate backtest: Focus GDP revisions + COPOM, SMLL vs. IBOV (no costs)", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"Aggregate = w_F · Focus + w_C · COPOM, weights ∝ 1/past vol (avg w_F {df['w_focus'].mean():.2f}). "
             f"CAGR {t.CAGR:+.1%}, vol {t.Vol:.1%}, Sharpe {t.Sharpe:.2f}, max DD {t.MaxDD:.0%}, "
             f"avg |position| {t['avg |pos|']:.2f}.", loc="left", fontsize=8.5, color=INK2, pad=10)
fig.text(0.07, 0.01, "Return = position × (r_SMLL − r_IBOV), daily-rebalanced dollar-neutral spread. "
         "Sources: BCB Focus, BCB COPOM, B3.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.1, right=0.93, top=0.89, bottom=0.06)
fig.savefig(HERE / "aggregate_backtest.png")
print("saved plots")
