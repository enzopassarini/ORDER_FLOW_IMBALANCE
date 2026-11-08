"""
EMBI+ Brazil spread-trend signal: what can be done with data that ends 30 Jul 2024.

Signal (weekly, Friday close): -(EMBI+ Brazil spread change over 63 sessions), scaled by
2 x its past std, clipped to [-1, 1] (macro_channels.py). Traded like GDP and the real rate:
4 staggered 21-day tranches from the Monday close (position = mean of last 4 weekly signals);
the weekly version is shown for comparison. Backtest ends 30 Jul 2024.

Also: predictive t-stat and Sharpe by 5-year block, and the combination with the current
core (GDP + real rate, equal risk) over the same dates.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
END = pd.Timestamp("2024-07-30")
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
spread = (px["SMLL"].pct_change() - px["IBOV"].pct_change()).dropna()
sig = pd.read_csv(DATA / "macro_channels_signal.csv", index_col=0, parse_dates=True)["embi_trend"].dropna()
sig = sig[sig.index <= END]
core = pd.read_csv(DATA / "aggregate3_daily.csv", index_col=0, parse_dates=True)
ev = pd.read_csv(DATA / "macro_channels_events.csv", index_col=0, parse_dates=True)


def daily(s):
    p = pd.Series(np.nan, index=spread.index)
    for d, v in s.dropna().items():
        i0 = px.index.searchsorted(d, side="right")
        if i0 < len(px):
            p.loc[p.index > px.index[i0]] = v
    return p


pos = pd.DataFrame({"EMBI weekly": daily(sig), "EMBI 4 tranches": daily(sig.rolling(4).mean()),
                    "Core (GDP + real rate)": core["GDP + Real rate"]})
pos = pos[(pos.index <= END + pd.Timedelta(days=1))].dropna(subset=["EMBI 4 tranches"])
sp = spread.loc[pos.index]
ret = pos.mul(sp, axis=0)
both = ret[["EMBI 4 tranches", "Core (GDP + real rate)"]].dropna()
vol = both.expanding(min_periods=252).std().shift(1)
inv = 1 / vol
w = inv.div(inv.sum(axis=1), axis=0).fillna(0.5)
pos.loc[both.index, "Core + EMBI"] = (w * pos.loc[both.index, both.columns]).sum(axis=1)
ret["Core + EMBI"] = pos["Core + EMBI"] * sp
pos.join(ret, rsuffix=" ret").to_csv(DATA / "embi_backtest_daily.csv")


def perf(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {"start": r.index[0].date(), "CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "Vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min()}


S = ["EMBI weekly", "EMBI 4 tranches", "Core (GDP + real rate)", "Core + EMBI"]
tab = pd.DataFrame({k: perf(ret[k]) for k in S}).T
common = ret[S].dropna()
tab["Sharpe (common 2007–24)"] = [perf(common[k])["Sharpe"] for k in S]
tab.to_csv(DATA / "embi_backtest_stats.csv")
pd.options.display.width = 200
print(tab.to_string())
print(f"\ncorr(EMBI tranches, core) daily returns: {common['EMBI 4 tranches'].corr(common['Core (GDP + real rate)']):.2f}")

blocks = [("2005", "2009"), ("2010", "2014"), ("2015", "2019"), ("2020", "2024")]
e = ev[["embi_trend", "fwd21"]].dropna()
br = []
for a, b in blocks:
    p = e.loc[a:b]
    m = sm.OLS(p["fwd21"] * 100, sm.add_constant(p["embi_trend"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    r_ = ret.loc[a:b]
    br.append({"period": f"{a}–{b}", "n_weeks": len(p), "slope": m.params["embi_trend"], "t_NW": m.tvalues["embi_trend"],
               "Sharpe EMBI (tranches)": perf(r_["EMBI 4 tranches"])["Sharpe"],
               "Sharpe core": perf(r_["Core (GDP + real rate)"])["Sharpe"] if r_["Core (GDP + real rate)"].notna().sum() > 60 else np.nan,
               "Sharpe core + EMBI": perf(r_["Core + EMBI"])["Sharpe"] if r_["Core + EMBI"].notna().sum() > 60 else np.nan})
br = pd.DataFrame(br).set_index("period")
br.to_csv(DATA / "embi_by_period.csv", float_format="%.3f")
print("\n", br.round(2).to_string())
ex = e[~e.index.to_series().between("2008-01-01", "2009-12-31")]
m = sm.OLS(ex["fwd21"] * 100, sm.add_constant(ex["embi_trend"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(f"excluding 2008-09: t {m.tvalues['embi_trend']:+.2f}")
r_ex = ret.drop(ret.loc["2008":"2009"].index)
print("Sharpe excluding 2008-09:", {k: round(perf(r_ex[k])["Sharpe"], 2) for k in S})

# --- plot ------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COL = {"EMBI weekly": "#9b9a95", "EMBI 4 tranches": "#6250d6", "Core (GDP + real rate)": "#1baf7a", "Core + EMBI": INK}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig = plt.figure(figsize=(12, 10), dpi=200)
gs = fig.add_gridspec(3, 2, width_ratios=[2.6, 1], height_ratios=[3, 1.3, 1.1], hspace=0.12, wspace=0.25)
ax, axp, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])
axb = fig.add_subplot(gs[:, 1])
for a in (ax, axp, axd, axb):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
for a in (ax, axp):
    a.tick_params(labelbottom=False)
for k in S:
    r = ret[k].dropna()
    eq = (1 + r).cumprod()
    t = tab.loc[k]
    ax.plot(eq.index, eq, color=COL[k], lw=2.3 if k == "Core + EMBI" else 1.3, ls="--" if k == "EMBI weekly" else "-",
            label=f"{k}: Sharpe {t.Sharpe:.2f}, max DD {t.MaxDD:.0%}")
    ax.annotate(f"{eq.iloc[-1]:.2f}×", (eq.index[-1], eq.iloc[-1]), xytext=(5, 0), textcoords="offset points",
                fontsize=8, color=INK2, va="center")
    axd.plot(eq.index, (eq / eq.cummax() - 1) * 100, color=COL[k], lw=1.2 if k == "Core + EMBI" else 0.8)
ax.axhline(1, color=MUTED, lw=1)
ax.set_ylabel("Growth of 1")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
p = pos["EMBI 4 tranches"]
axp.fill_between(p.index, 0, p.clip(lower=0), color="#1baf7a", lw=0, step="post", label="Long SMLL (spreads tightening)")
axp.fill_between(p.index, 0, p.clip(upper=0), color="#eb6834", lw=0, step="post", label="Short SMLL (spreads widening)")
axp.axhline(0, color=MUTED, lw=1)
axp.set_ylim(-1.05, 1.4)
axp.set_yticks([-1, -0.5, 0, 0.5, 1])
axp.set_ylabel("EMBI position\n(4 tranches)")
axp.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2)
axd.set_ylabel("Drawdown (%)")
for a in (ax, axp, axd):
    a.set_xlim(pos.index[0], pos.index[-1] + pd.Timedelta(days=200))
yb = np.arange(len(br))[::-1]
axb.barh(yb + 0.18, br["t_NW"], height=0.34, color="#6250d6", label="Predictive NW t (fwd 21)", zorder=3)
axb.barh(yb - 0.18, br["Sharpe EMBI (tranches)"] * 4, height=0.34, color="#b5b4ae", label="Backtest Sharpe × 4", zorder=3)
for y, (_, r) in zip(yb, br.iterrows()):
    axb.text(r["t_NW"] + (0.08 if r["t_NW"] >= 0 else -0.08), y + 0.18, f"t {r['t_NW']:+.2f}", va="center",
             ha="left" if r["t_NW"] >= 0 else "right", fontsize=8)
    axb.text(r["Sharpe EMBI (tranches)"] * 4 + (0.08 if r["Sharpe EMBI (tranches)"] >= 0 else -0.08), y - 0.18,
             f"SR {r['Sharpe EMBI (tranches)']:+.2f}", va="center",
             ha="left" if r["Sharpe EMBI (tranches)"] >= 0 else "right", fontsize=8)
axb.axvline(0, color=MUTED, lw=1)
axb.axvline(1.96, color=MUTED, lw=1, ls="--")
axb.set_yticks(yb)
axb.set_yticklabels(br.index)
axb.set_xlim(-2.5, 5.5)
axb.set_title("EMBI signal by 5-year block", loc="left", fontsize=9, color=INK2)
axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=8, labelcolor=INK2)
fig.suptitle("EMBI+ Brazil spread trend: backtest to Jul 2024 (data end) and combination with the core",
             x=0.05, ha="left", fontsize=13, fontweight="bold", y=0.975)
fig.text(0.05, 0.925, "Signal = −(3-month change of EMBI+ Brazil spread) / (2 × past std), clipped ±1; 4 staggered "
         "21-day tranches from Monday close.\nCore = GDP revisions + real rate (tranches), equal risk; Core + EMBI = "
         "equal-risk mix. No costs. Sharpes in the legend use each strategy's own start (EMBI 2005, core 2007).",
         fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.07)
fig.savefig(HERE / "embi_backtest.png")
print("saved")
