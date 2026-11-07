"""
Fed cycle signal for SMLL vs. IBOV — same construction as the COPOM signal.

  signal   = clip(delta_bp, -100, 100) / 100   in [-1, 1]  (positive = Fed hike)
  position = -signal                           (Fed cut -> long SMLL / short IBOV)

Timing: the FOMC statement is released at 14:00 ET, before the B3 close, but to stay
conservative (and match the COPOM code) the position is entered at the close of the first
session AFTER the decision date and held until the close of the first session after the
next decision.
Analysis: one observation per decision, fwd 21-session SMLL - IBOV return, plain OLS,
full sample + halves; means by signal level (the signal only takes 8 values).
Backtest: daily position * (r_SMLL - r_IBOV), no costs; crisis windows; correlation with
the COPOM and Focus strategies.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from copom_signal import selic_signal

HERE = Path(__file__).parent
DATA = HERE / "data"
H = 21

px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
fed = pd.read_csv(DATA / "fed_decisions.csv", parse_dates=["date"])
fed = fed[fed["date"] >= px.index[0]].reset_index(drop=True)
fed["signal"] = selic_signal(fed["delta_bp"])  # same transform: clip(+-100 bp) / 100

# --- per-decision forward returns ----------------------------------------------------------
rows = []
for _, r in fed.iterrows():
    i0 = px.index.searchsorted(r["date"], side="right")
    if i0 + H >= len(px):
        continue
    ret = px.iloc[i0 + H] / px.iloc[i0] - 1
    rows.append({"date": r["date"], "delta_bp": r["delta_bp"], "signal": r["signal"],
                 "extraordinary": r["extraordinary"], "entry": px.index[i0], "exit": px.index[i0 + H],
                 "fwd_rel": ret["SMLL"] - ret["IBOV"]})
ev = pd.DataFrame(rows)
ev.to_csv(DATA / "fed_fwd21.csv", index=False)
half = len(ev) // 2
samples = {"Full sample": ev,
           f"First half ({ev.date.iloc[0]:%b %Y} – {ev.date.iloc[half - 1]:%b %Y})": ev.iloc[:half],
           f"Second half ({ev.date.iloc[half]:%b %Y} – {ev.date.iloc[-1]:%b %Y})": ev.iloc[half:]}
res = []
for nm, e in samples.items():
    x, y = e["signal"], e["fwd_rel"] * 100
    m = sm.OLS(y, sm.add_constant(x)).fit()
    nz = x != 0
    res.append({"sample": nm, "n": len(e), "n_moves": int(nz.sum()), "slope_pp": m.params["signal"],
                "se": m.bse["signal"], "t": m.tvalues["signal"], "p": m.pvalues["signal"], "R2": m.rsquared,
                "corr": np.corrcoef(x, y)[0, 1], "hit": (np.sign(x[nz]) == -np.sign(y[nz])).mean(),
                "mean_cut": y[x < 0].mean(), "mean_hold": y[x == 0].mean(), "mean_hike": y[x > 0].mean()})
st = pd.DataFrame(res).set_index("sample")
st.to_csv(DATA / "fed_ols_halves.csv", float_format="%.4f")
pd.options.display.width = 220
print(st.round(3).to_string())
lvl = ev.groupby("signal")["fwd_rel"].agg(["count", "mean"])
lvl["mean"] *= 100
print("\nby signal level:\n", lvl.round(2).to_string())
ex = ev[~ev["date"].between("2008-01-01", "2009-06-30")]
m = sm.OLS(ex["fwd_rel"] * 100, sm.add_constant(ex["signal"])).fit()
print(f"\nExcluding 2008-H1 2009 decisions: n={len(ex)} slope {m.params['signal']:+.2f} t {m.tvalues['signal']:+.2f}")

# --- backtest ------------------------------------------------------------------------------
spread = (px["SMLL"].pct_change() - px["IBOV"].pct_change()).dropna()
pos = pd.Series(np.nan, index=spread.index)
for _, r in fed.iterrows():
    i0 = px.index.searchsorted(r["date"], side="right")
    if i0 < len(px):
        pos.loc[pos.index > px.index[i0]] = -r["signal"]
pos = pos.dropna()
spread = spread.loc[pos.index]
strat = pos * spread
bt = pd.DataFrame({"position": pos, "spread_ret": spread, "strategy_ret": strat})
bt["equity"] = (1 + strat).cumprod()
bt.to_csv(DATA / "fed_backtest_daily.csv")


def perf(r):
    eq = (1 + r).cumprod()
    return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "Vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "MaxDD": (eq / eq.cummax() - 1).min(), "Total": eq.iloc[-1] - 1}


mid = len(strat) // 2
tab = pd.DataFrame({"Full": perf(strat), "First half": perf(strat.iloc[:mid]),
                    "Second half": perf(strat.iloc[mid:])}).T
tab["time in market"] = [(pos != 0).mean(), (pos.iloc[:mid] != 0).mean(), (pos.iloc[mid:] != 0).mean()]
tab.to_csv(DATA / "fed_backtest_stats.csv", float_format="%.4f")
print(f"\nBacktest {strat.index[0].date()} -> {strat.index[-1].date()}")
print(tab.round(3).to_string())

crises = {"GFC (Sep 2007 – Mar 2009)": ("2007-09-01", "2009-03-31"),
          "Euro crisis (Jul 2011 – Dec 2011)": ("2011-07-01", "2011-12-31"),
          "Covid (Feb 2020 – Jun 2020)": ("2020-02-01", "2020-06-30"),
          "Fed hikes (Mar 2022 – Jul 2023)": ("2022-03-01", "2023-07-31")}
cr = []
for nm, (a, b) in crises.items():
    s_, sp = strat.loc[a:b], spread.loc[a:b]
    cr.append({"period": nm, "strategy": (1 + s_).prod() - 1, "SMLL-IBOV spread": (1 + sp).prod() - 1,
               "avg position": pos.loc[a:b].mean()})
cr = pd.DataFrame(cr).set_index("period")
cr.to_csv(DATA / "fed_crisis_windows.csv", float_format="%.4f")
print("\n", cr.round(3).to_string())

others = pd.DataFrame({"fed": strat,
                       "copom": pd.read_csv(DATA / "backtest_daily.csv", index_col=0, parse_dates=True)["strategy_ret"],
                       "focus": pd.read_csv(DATA / "focus_tranches_daily.csv", index_col=0,
                                            parse_dates=True)["4 tranches x 21d_ret"]}).dropna()
cor = others.corr()
cor.to_csv(DATA / "fed_strategy_correlations.csv", float_format="%.3f")
print("\nDaily strategy-return correlations:\n", cor.round(2).to_string())

# --- plots ---------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(a, yonly=False):
    a.grid(True, axis="y" if yonly else "both", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), dpi=200, gridspec_kw={"height_ratios": [1.25, 1]})
levels = sorted(ev["signal"].unique())
rng = np.random.default_rng(0)
for j, (nm, e) in enumerate(samples.items()):
    s = st.loc[nm]
    x, y = e["signal"].to_numpy(), e["fwd_rel"].to_numpy() * 100
    ax = axes[0, j]
    style(ax)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.scatter(x + rng.uniform(-0.03, 0.03, len(x)), y, s=24, color=VIOLET, alpha=0.65, edgecolor=SURF, lw=0.6, zorder=3)
    g = e.groupby("signal")["fwd_rel"].mean() * 100
    ax.scatter(g.index, g.values, s=80, marker="_", color=INK, lw=2.2, zorder=4)
    xs = np.linspace(-1, 1, 20)
    b, a_ = np.polyfit(x, y, 1)
    ax.plot(xs, a_ + b * xs, color=INK, lw=2, zorder=4)
    for _, r in e.loc[e["fwd_rel"].abs().nlargest(2).index].iterrows():
        ax.annotate(r["date"].strftime("%b %Y"), (r["signal"], r["fwd_rel"] * 100), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-16, 16)
    ax.set_title(f"{nm}\nn={int(s.n)} ({int(s.n_moves)} moves)  slope {s.slope_pp:+.2f} pp  SE {s.se:.2f}  "
                 f"t {s.t:+.2f}  p {s.p:.3f}\nR² {s.R2:.3f}  ρ {s['corr']:+.2f}  hit (moves) {s.hit:.0%}",
                 loc="left", fontsize=8.5, color=INK2)
    ax.set_xlabel("Signal = clip(ΔFed target, ±100 bp) / 100   ← Fed cut · Fed hike →")
    if j == 0:
        ax.set_ylabel("SMLL − IBOV fwd 21-session return (pp)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    # bars by signal level
    ax = axes[1, j]
    style(ax, True)
    ax.axhline(0, color=MUTED, lw=1)
    gg = e.groupby("signal")["fwd_rel"].agg(["count", "mean", "std"]).reindex(levels)
    gg["mean"] *= 100
    gg["se"] = gg["std"] * 100 / np.sqrt(gg["count"])
    xpos = np.arange(len(levels))
    cols = [BLUE if l < 0 else ORANGE if l > 0 else GRAY for l in levels]
    ax.bar(xpos, gg["mean"].fillna(0), width=0.7, color=cols, zorder=3)
    ax.errorbar(xpos, gg["mean"], yerr=gg["se"], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    for xi, (l, r) in zip(xpos, gg.iterrows()):
        if pd.isna(r["mean"]):
            continue
        off = (r["se"] if not pd.isna(r["se"]) else 0) + 0.2
        ax.text(xi, float(np.clip(r["mean"] + (off if r["mean"] >= 0 else -off), -7.4, 7.4)), f"{r['mean']:+.1f}", ha="center",
                va="bottom" if r["mean"] >= 0 else "top", fontsize=7.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{l:+.2f}\nn={int(c) if not pd.isna(c) else 0}" if l else f"0\nn={int(c)}"
                        for l, c in zip(levels, gg["count"])], fontsize=7.5)
    ax.set_ylim(-8, 8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    if j == 0:
        ax.set_ylabel("Mean fwd 21-session return (pp)")
    ax.set_title("Mean by signal level (±1 SE)", loc="left", fontsize=8.5, color=INK2)
from matplotlib.patches import Patch
fig.legend([Patch(color=BLUE), Patch(color=GRAY), Patch(color=ORANGE)], ["Fed cut", "Hold", "Fed hike"],
           loc="upper left", bbox_to_anchor=(0.045, 0.925), frameon=False, fontsize=8.5, ncol=3, labelcolor=INK2)
fig.suptitle("Fed cycle signal vs. next 21-session small-cap relative return — one observation per FOMC decision",
             x=0.05, ha="left", fontsize=13, fontweight="bold", y=0.985)
fig.text(0.05, 0.935, "Target = FRED DFEDTAR / DFEDTARU; meetings from federalreserve.gov (+ intermeeting moves). "
         "Window: close of first B3 session after the decision → 21 sessions later. Plain OLS.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(HERE / "fed_signal_fwd21.png")
plt.close(fig)

plt.rcParams.update({"font.size": 10})
fig, (ax, axp, axd) = plt.subplots(3, 1, figsize=(10, 8.6), dpi=200, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2, 1.1], "hspace": 0.1})
for a in (ax, axp, axd):
    style(a)
    a.axvline(strat.index[mid], color=MUTED, lw=1, ls="--")
    for nm, (a0, b0) in crises.items():
        a.axvspan(pd.Timestamp(a0), pd.Timestamp(b0), color=GRAY, alpha=0.12, lw=0)
ax.plot(bt.index, bt["equity"], color=VIOLET, lw=2)
ax.axhline(1, color=MUTED, lw=1)
ax.annotate(f"{bt['equity'].iloc[-1]:.2f}×", (bt.index[-1], bt["equity"].iloc[-1]), xytext=(6, 0),
            textcoords="offset points", fontsize=9, color=INK2, va="center")
ax.set_ylabel("Growth of 1")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
axp.fill_between(pos.index, 0, pos.clip(lower=0), color=BLUE, lw=0, step="post", label="Long SMLL (after Fed cut)")
axp.fill_between(pos.index, 0, pos.clip(upper=0), color=ORANGE, lw=0, step="post", label="Short SMLL (after Fed hike)")
axp.axhline(0, color=MUTED, lw=1)
axp.set_ylim(-1.05, 1.4)
axp.set_yticks([-1, -0.5, 0, 0.5, 1])
axp.set_ylabel("Position")
axp.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
dd = (bt["equity"] / bt["equity"].cummax() - 1) * 100
axd.plot(dd.index, dd, color=VIOLET, lw=1.1)
axd.set_ylabel("Drawdown (%)")
f_, h1, h2 = tab.loc["Full"], tab.loc["First half"], tab.loc["Second half"]
gfc = cr.iloc[0]
fig.suptitle("Fed cycle signal backtest: SMLL vs. IBOV spread (no costs)", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"Position = −clip(ΔFed, ±100 bp)/100, held decision to decision.  CAGR {f_.CAGR:+.1%}, vol {f_.Vol:.1%}, "
             f"Sharpe {f_.Sharpe:.2f} (1st half {h1.Sharpe:.2f}, 2nd {h2.Sharpe:.2f}), max DD {f_.MaxDD:.0%}.\n"
             f"Grey bands = stress windows. GFC: strategy {gfc.strategy:+.1%} vs. SMLL−IBOV {gfc['SMLL-IBOV spread']:+.1%}. "
             f"Return corr: COPOM {cor.loc['fed', 'copom']:+.2f}, Focus {cor.loc['fed', 'focus']:+.2f}.",
             loc="left", fontsize=8.5, color=INK2, pad=10)
fig.text(0.07, 0.01, "Return = position × (r_SMLL − r_IBOV), daily-rebalanced dollar-neutral spread. "
         "Sources: FRED, federalreserve.gov, B3.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.09, right=0.93, top=0.87, bottom=0.07)
fig.savefig(HERE / "fed_backtest.png")
print("saved plots")
