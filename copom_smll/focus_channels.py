"""
Test more Focus-survey channels as predictors of SMLL - IBOV relative returns.

Every feature is oriented so that POSITIVE = good for small caps (long SMLL / short IBOV).
Friday t, Y_h = year(t) + h. rev48(x) = average of the 4- and 8-week change of x for the
SAME target year (so the January roll never creates a jump).

  fiscal_primary  +mean_{h=1,2} rev48(median primary balance, % GDP)       improving -> long
  fiscal_debt     -mean_{h=1,2} rev48(median net public debt, % GDP)        rising debt -> short
  disp_gdp        -rev48(cross-sectional std of GDP forecasts, h=1)         rising dispersion -> short
  disp_ipca       -rev48(cross-sectional std of IPCA forecasts, h=1)        rising dispersion -> short
  deanchor_level  -(mean_{h=2,3} IPCA median - inflation target), demeaned   above target -> short
  deanchor_chg    -mean_{h=2,3} rev48(IPCA median)                          drifting up -> short
  selic_rev       -(4-week change of median year-end Selic, h=1)            higher path -> short
  real_rate       -(expected Selic 12m ahead - 12m expected IPCA), demeaned high real rate -> short
  gdp_rev (ref.)  +rev48(median GDP, h=1)                                   upgrade -> long

Level features are demeaned with an expanding mean of past weeks. Then every feature is
scaled: signal = clip(x / (2 * expanding std of past weeks), -1, 1) (min 52 weeks).

Tests (weekly signal as held from Monday close, the first session after the Friday survey):
  - fwd 21-session SMLL-IBOV return, overlapping weekly windows, Newey-West (5 lags):
    full sample and halves, decile spread
  - next-week return (Monday -> Monday), plain OLS (what a weekly strategy trades)
  - correlation with the COPOM position and the Focus GDP signal; t-stat of the feature in a
    joint regression with both (does it add information?)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
H, NW_LAGS = 21, 5
TARGET = {**{y: 4.5 for y in range(2005, 2019)}, 2019: 4.25, 2020: 4.0, 2021: 3.75, 2022: 3.5, 2023: 3.25,
          **{y: 3.0 for y in range(2024, 2035)}}


def panel(key, kind="median"):
    w = pd.read_csv(DATA / f"focus_weekly_{key}_{kind}.csv", index_col=0, parse_dates=True)
    w.columns = w.columns.astype(int)
    return w


def at(w, h, lag=0):
    """value for target year year(t)+h, observed `lag` weeks before t (same target year)."""
    vals = []
    for i, d in enumerate(w.index):
        y, j = d.year + h, i - lag
        vals.append(w.iat[j, w.columns.get_loc(y)] if j >= 0 and y in w.columns else np.nan)
    return pd.Series(vals, index=w.index)


def rev48(w, h):
    return ((at(w, h) - at(w, h, 4)) + (at(w, h) - at(w, h, 8))) / 2


prim, debt, ipca, selic, gdp = (panel(k) for k in ("primary", "netdebt", "ipca", "selic", "gdp"))
gdp_sd, ipca_sd = panel("gdp", "std"), panel("ipca", "std")
ipca12 = pd.read_csv(DATA / "focus_weekly_ipca12m.csv", index_col=0, parse_dates=True)["Mediana"]

idx = gdp.index
raw = pd.DataFrame(index=idx)
raw["fiscal_primary"] = (rev48(prim, 1) + rev48(prim, 2)).reindex(idx) / 2
raw["fiscal_debt"] = -(rev48(debt, 1) + rev48(debt, 2)).reindex(idx) / 2
raw["disp_gdp"] = -rev48(gdp_sd, 1).reindex(idx)
raw["disp_ipca"] = -rev48(ipca_sd, 1).reindex(idx)
gap = pd.concat([at(ipca, h) - pd.Series([TARGET.get(d.year + h, np.nan) for d in ipca.index], index=ipca.index)
                 for h in (2, 3)], axis=1).mean(axis=1).reindex(idx)
raw["deanchor_level"] = -gap
raw["deanchor_chg"] = -((rev48(ipca, 2) + rev48(ipca, 3)) / 2).reindex(idx)
raw["selic_rev"] = -(at(selic, 1) - at(selic, 1, 4)).reindex(idx)
frac = pd.Series((idx.dayofyear / 365.25).to_numpy(), index=idx)  # 12m ahead sits this far between end-Y0 and end-Y1
selic12 = (1 - frac) * at(selic, 0).reindex(idx) + frac * at(selic, 1).reindex(idx)
raw["real_rate"] = -(selic12 - ipca12.reindex(idx))
raw["real_rate_r5"] = raw["real_rate"]  # same feature, rolling 5-year baseline and 3-sigma scale (see below)
# decomposition of the real rate (same rolling 5y / 3-sigma construction), + = long small caps
raw["nominal_exp_r5"] = -selic12                                   # expected Selic 12m ahead (Focus)
cop = pd.read_csv(DATA / "copom_decisions.csv", parse_dates=["date"]).sort_values("date")
raw["selic_now_r5"] = -pd.Series([cop.loc[cop["date"] < d, "selic_meta"].iloc[-1] if (cop["date"] < d).any()
                                  else np.nan for d in idx], index=idx)  # current Selic target
raw["infl_exp_r5"] = -ipca12.reindex(idx)                          # 12m expected IPCA (Focus)
_ir = pd.read_csv(DATA / "macro_ipca12_realized.csv", parse_dates=["date", "avail"]).sort_values("avail")
_ir = pd.Series(_ir["value"].to_numpy(), index=_ir["avail"]).groupby(level=0).last()
raw["infl_real_r5"] = -_ir.reindex(_ir.index.union(idx)).ffill().reindex(idx)  # realized 12m IPCA, once released
raw["gdp_rev"] = rev48(gdp, 1).reindex(idx)
LEVEL = {"deanchor_level", "real_rate"}
R5 = {"real_rate_r5", "nominal_exp_r5", "selic_now_r5", "infl_exp_r5", "infl_real_r5"}
FEATS = [c for c in raw.columns]
LABEL = {"fiscal_primary": "1a Fiscal: primary balance revisions", "fiscal_debt": "1b Fiscal: net debt revisions",
         "disp_gdp": "2a Dispersion: GDP forecasts", "disp_ipca": "2b Dispersion: IPCA forecasts",
         "deanchor_level": "3a De-anchoring: IPCA 2–3y − target (level)",
         "deanchor_chg": "3b De-anchoring: IPCA 2–3y revisions",
         "selic_rev": "4 Selic expectation revisions (4w)", "real_rate": "5 Ex-ante real rate (level)",
         "real_rate_r5": "5' Real rate vs. rolling 5y mean (3σ)",
         "nominal_exp_r5": "6a Nominal: expected Selic 12m vs. 5y mean",
         "selic_now_r5": "6b Nominal: current Selic target vs. 5y mean",
         "infl_exp_r5": "7a Inflation: 12m expected IPCA vs. 5y mean",
         "infl_real_r5": "7b Inflation: realized 12m IPCA vs. 5y mean",
         "gdp_rev": "ref. GDP revisions"}

sig = pd.DataFrame(index=idx)
for c in FEATS:
    x = raw[c]
    if c in R5:
        # rolling 5-year baseline instead of the expanding one, which stayed anchored to the very high
        # real rates of the early 2000s and left the signal at +1 for 30% of weeks; 3-sigma scale
        w = 5 * 52
        x = x - x.rolling(w, min_periods=w // 2).mean().shift(1)
        sig[c] = (x / (3 * x.rolling(w, min_periods=w // 2).std().shift(1))).clip(-1, 1)
        continue
    if c in LEVEL:
        x = x - x.expanding(min_periods=52).mean().shift(1)
    sig[c] = (x / (2 * x.expanding(min_periods=52).std().shift(1))).clip(-1, 1)
raw.to_csv(DATA / "focus_channels_raw.csv")
sig.to_csv(DATA / "focus_channels_signal.csv")

# --- targets and comparison signals at each Friday -------------------------------------------
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
spread_d = px["SMLL"].pct_change() - px["IBOV"].pct_change()
copom_pos = pd.read_csv(DATA / "backtest_daily.csv", index_col=0, parse_dates=True)["position"]
rows = []
fr = [d for d in idx if d >= px.index[0]]
for k, d in enumerate(fr):
    i0 = px.index.searchsorted(d, side="right")
    if i0 + 1 >= len(px):
        break
    row = {"date": d, "copom": copom_pos.get(px.index[i0 + 1], np.nan)}
    if i0 + H < len(px):
        r = px.iloc[i0 + H] / px.iloc[i0] - 1
        row["fwd21"] = r["SMLL"] - r["IBOV"]
    if k + 1 < len(fr):
        i1 = px.index.searchsorted(fr[k + 1], side="right")
        if i1 < len(px):
            row["fwd_week"] = (1 + spread_d.iloc[i0 + 1:i1 + 1]).prod() - 1
    rows.append(row)
ev = pd.DataFrame(rows).set_index("date").join(sig)
ev.to_csv(DATA / "focus_channels_events.csv")


def nw_ols(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})


res, dec_tab = [], {}
for c in FEATS:
    e = ev[list(dict.fromkeys([c, "fwd21", "fwd_week", "copom", "gdp_rev"]))].dropna(subset=[c, "fwd21"])
    half = len(e) // 2
    out = {"feature": c, "label": LABEL[c], "n": len(e), "start": e.index[0].date()}
    for nm, p in (("full", e), ("H1", e.iloc[:half]), ("H2", e.iloc[half:])):
        m = nw_ols(p["fwd21"] * 100, p[c])
        out[f"slope_{nm}"], out[f"t_{nm}"] = m.params[c], m.tvalues[c]
    nz = e[c] != 0
    out["hit"] = (np.sign(e.loc[nz, c]) == np.sign(e.loc[nz, "fwd21"])).mean()
    q = pd.qcut(e[c].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    d10 = e.groupby(q)["fwd21"].mean() * 100
    dec_tab[c] = d10
    out["D10_D1"] = d10.loc[10] - d10.loc[1]
    w = e.dropna(subset=["fwd_week"])
    mw = sm.OLS(w["fwd_week"] * 100, sm.add_constant(w[c])).fit()
    out["t_nextweek"] = mw.tvalues[c]
    out["corr_copom"] = e[c].corr(e["copom"])
    out["corr_gdp"] = e[c].corr(e["gdp_rev"]) if c != "gdp_rev" else 1.0
    out["autocorr_4w"] = sig[c].autocorr(4)
    j = e.dropna(subset=["copom"])
    Xcols = [c, "copom"] + (["gdp_rev"] if c != "gdp_rev" else [])
    mj = nw_ols(j["fwd21"] * 100, j[Xcols])
    out["t_joint"] = mj.tvalues[c]
    out["t_copom_in_joint"] = mj.tvalues["copom"]
    res.append(out)
st = pd.DataFrame(res).set_index("feature")
st.to_csv(DATA / "focus_channels_stats.csv", float_format="%.4f")
pd.options.display.width = 250
pd.options.display.max_columns = 30
print(st.drop(columns="label").round(2).to_string())
print("\nSignal correlation matrix (weekly):")
cm = ev[FEATS + ["copom"]].corr()
cm.to_csv(DATA / "focus_channels_corr.csv", float_format="%.3f")
print(cm.round(2).to_string())

# real-rate decomposition: which component carries the information?
for cols in (["real_rate_r5"], ["nominal_exp_r5"], ["infl_exp_r5"], ["nominal_exp_r5", "infl_exp_r5"],
             ["real_rate_r5", "nominal_exp_r5"], ["real_rate_r5", "infl_exp_r5"], ["selic_now_r5", "infl_real_r5"],
             ["nominal_exp_r5", "copom"], ["real_rate_r5", "copom"]):
    d_ = ev[cols + ["fwd21"]].dropna()
    m = nw_ols(d_["fwd21"] * 100, d_[cols])
    print("decomp", cols, {k: round(v, 2) for k, v in m.tvalues.items() if k != "const"}, f"n={len(d_)}")

# selic_rev vs COPOM horse race
hr = ev[["selic_rev", "copom", "fwd21"]].dropna()
for cols in (["copom"], ["selic_rev"], ["selic_rev", "copom"]):
    m = nw_ols(hr["fwd21"] * 100, hr[cols])
    print("horse race", cols, {k: round(v, 2) for k, v in m.tvalues.items() if k != "const"},
          f"R2 {m.rsquared:.3f}")

# --- plots ----------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(a, yonly=False):
    a.grid(True, axis="y" if yonly else "both", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


# fig 1: t-stat summary
order = FEATS
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=200, gridspec_kw={"width_ratios": [1.5, 1]})
style(ax)
ypos = np.arange(len(order))[::-1]
for col, colr, mk, lab in (("t_full", INK, "o", "Full sample"), ("t_H1", BLUE, "s", "First half"),
                           ("t_H2", ORANGE, "D", "Second half"), ("t_joint", VIOLET, "^",
                           "Full, controlling for COPOM + GDP")):
    ax.scatter(st.loc[order, col], ypos, s=46 if col == "t_full" else 30, color=colr, marker=mk, zorder=3, label=lab)
for x0 in (-1.96, 1.96):
    ax.axvline(x0, color=MUTED, lw=1, ls="--")
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(ypos)
ax.set_yticklabels([LABEL[c] for c in order])
ax.set_xlabel("Newey-West t-stat, fwd 21-session SMLL − IBOV return on signal (+ = predicts as hypothesised)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=8, labelcolor=INK2, ncol=4)
ax.set_title("Predictive power (dashed = ±1.96)", loc="left", fontsize=9, color=INK2)
style(ax2)
ax2.grid(False)
cm2 = ev[order + ["copom"]].corr()
im = ax2.imshow(cm2.values, cmap="RdBu_r", vmin=-1, vmax=1)
short = {c: c.replace("fiscal_", "fisc ").replace("deanchor_", "deanch ").replace("_", " ") for c in order}
short["copom"] = "COPOM"
ax2.set_xticks(range(len(cm2)))
ax2.set_xticklabels([short[c] for c in cm2.columns], rotation=60, ha="right", fontsize=7.5)
ax2.set_yticks(range(len(cm2)))
ax2.set_yticklabels([short[c] for c in cm2.index], fontsize=7.5)
for i in range(len(cm2)):
    for j in range(len(cm2)):
        v = cm2.values[i, j]
        ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.55 else INK)
ax2.set_title("Signal correlations (weekly)", loc="left", fontsize=9, color=INK2)
fig.suptitle("Focus channels vs. next 21-session small-cap relative return", x=0.02, ha="left",
             fontsize=13, fontweight="bold")
fig.text(0.02, 0.92, "Each signal oriented so + = long small caps; clip(x / (2 × past std), ±1). Weekly, held from "
         "Monday close; overlapping windows, Newey-West (5 lags). Sep 2005 – Oct 2026.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(HERE / "focus_channels_summary.png")
plt.close(fig)

# fig 2: deciles per feature (full sample)
fig, axes = plt.subplots(3, 3, figsize=(13, 10), dpi=200, sharey=True)
for a, c in zip(axes.flat, order):
    style(a, True)
    a.axhline(0, color=MUTED, lw=1)
    d = dec_tab[c]
    s = st.loc[c]
    a.bar(d.index, d.values, width=0.7, color=[ORANGE if v < 0 else AQUA for v in d.values], zorder=3)
    a.set_xticks(range(1, 11))
    a.set_xticklabels([f"D{i}" for i in range(1, 11)], fontsize=7)
    a.set_title(f"{LABEL[c]}\nNW t: full {s.t_full:+.2f} · H1 {s.t_H1:+.2f} · H2 {s.t_H2:+.2f} · "
                f"D10−D1 {s.D10_D1:+.1f} pp", loc="left", fontsize=8, color=INK2)
    a.set_yticks([-2, -1, 0, 1, 2])
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
for a in axes[:, 0]:
    a.set_ylabel("Mean fwd 21-session\nSMLL − IBOV (pp)")
fig.suptitle("Focus channels: deciles of signal vs. next 21-session small-cap relative return (full sample)",
             x=0.02, ha="left", fontsize=13, fontweight="bold")
fig.text(0.02, 0.945, "D1 = most bearish for small caps, D10 = most bullish. Ties (zero revisions) split by date.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "focus_channels_deciles.png")
print("saved plots")
