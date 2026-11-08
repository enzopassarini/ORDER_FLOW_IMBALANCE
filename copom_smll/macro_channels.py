"""
Test macro channels as predictors of SMLL - IBOV relative returns (same battery as
focus_channels.py). Every feature is oriented so POSITIVE = long SMLL / short IBOV.

Weekly on Fridays; a monthly value is used only once its availability date has passed
(see fetch_macro.py), daily series use the Friday close. dX = change over X months.

  credit_default  -mean(d3m, d6m) of the business default rate (BCB 21083)    rising stress -> short
  hh_dsr          -mean(d3m, d6m) of the household debt-service ratio (29034)  rising burden -> short
  epu             -mean(d3m, d6m) of log Brazil EPU                            rising uncertainty -> short
  dxy_trend       -mean(6m, 12m log change) of the dollar index                stronger dollar -> short
  embi_trend      -(3-month change of EMBI+ Brazil spread)                     widening -> short
  embi_level      -(EMBI+ Brazil spread - rolling 5y mean)                     wide spread -> short
                  (EMBI+ Brazil ends Jul 2024, so both EMBI features stop there)

Changes are scaled by 2 x their expanding past std; the level by 3 x its rolling 5-year std
(as for the real rate); all clipped to [-1, 1].
Tests: fwd 21-session SMLL-IBOV return from the Monday close (overlapping weekly windows,
Newey-West 5 lags), full sample and halves; next-week return (plain OLS); correlations with
the existing signals (COPOM, GDP, real rate, Fed, Focus IPCA dispersion) and the t-stat after
controlling for the current core (GDP + real rate, both in tranches).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
H, NW_LAGS = 21, 5

px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
fridays = pd.date_range("2000-01-07", px.index[-1], freq="W-FRI")


def load(name):
    return pd.read_csv(DATA / f"macro_{name}.csv", parse_dates=["date", "avail"])


def known_monthly(name):
    """Monthly series as known on each Friday: latest reference month with avail <= Friday."""
    d = load(name).sort_values("avail")
    s = pd.Series(d["value"].to_numpy(), index=d["avail"])
    s = s[~s.index.duplicated(keep="last")]
    ref = pd.Series(d["date"].to_numpy(), index=d["avail"])
    ref = ref[~ref.index.duplicated(keep="last")]
    return s.reindex(s.index.union(fridays)).ffill().reindex(fridays), d.set_index("date")["value"]


def monthly_change(name, log=False):
    """mean of 3m and 6m changes, computed on the monthly series, then made known at avail date."""
    d = load(name).sort_values("date").set_index("date")
    v = np.log(d["value"]) if log else d["value"]
    ch = ((v - v.shift(3)) + (v - v.shift(6))) / 2
    s = pd.Series(ch.to_numpy(), index=d["avail"].to_numpy()).dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(fridays)).ffill().reindex(fridays)


def daily_at_fridays(name):
    d = load(name).set_index("date")["value"]
    return d.reindex(d.index.union(fridays)).ffill().reindex(fridays), d


raw = pd.DataFrame(index=fridays)
raw["credit_default"] = -monthly_change("credit_default")
raw["hh_dsr"] = -monthly_change("hh_dsr")
raw["epu"] = -monthly_change("epu", log=True)
dxy_f, dxy_d = daily_at_fridays("dxy")
ld = np.log(dxy_d)
dxy_tr = ((ld - ld.shift(126)) + (ld - ld.shift(252))) / 2
raw["dxy_trend"] = -dxy_tr.reindex(dxy_tr.index.union(fridays)).ffill().reindex(fridays)
embi_f, embi_d = daily_at_fridays("embi")
embi_f[embi_f.index > embi_d.index.max() + pd.Timedelta(days=7)] = np.nan  # series ends Jul 2024
emb_tr = embi_d - embi_d.shift(63)
raw["embi_trend"] = -emb_tr.reindex(emb_tr.index.union(fridays)).ffill().reindex(fridays)
raw.loc[raw.index > embi_d.index.max() + pd.Timedelta(days=7), "embi_trend"] = np.nan
raw["embi_level"] = -embi_f
FEATS = list(raw.columns)
LABEL = {"credit_default": "1 Credit stress: business default rate (Δ)",
         "hh_dsr": "2 Household debt-service ratio (Δ)",
         "epu": "3 Brazil policy uncertainty, EPU (Δ log)",
         "dxy_trend": "4 Dollar cycle: DXY 6–12m trend",
         "embi_trend": "5a EMBI+ Brazil, 3m change (to Jul 2024)",
         "embi_level": "5b EMBI+ Brazil vs. 5y mean (to Jul 2024)"}

sig = pd.DataFrame(index=fridays)
for c in FEATS:
    x = raw[c]
    if c == "embi_level":
        w = 5 * 52
        x = x - x.rolling(w, min_periods=w // 2).mean().shift(1)
        sig[c] = (x / (3 * x.rolling(w, min_periods=w // 2).std().shift(1))).clip(-1, 1)
    else:
        sig[c] = (x / (2 * x.expanding(min_periods=52).std().shift(1))).clip(-1, 1)
raw.to_csv(DATA / "macro_channels_raw.csv")
sig.to_csv(DATA / "macro_channels_signal.csv")

# --- events: fwd returns and existing signals at each Friday -----------------------------------
spread_d = px["SMLL"].pct_change() - px["IBOV"].pct_change()
agg = pd.read_csv(DATA / "aggregate3_daily.csv", index_col=0, parse_dates=True)
fed_pos = pd.read_csv(DATA / "fed_backtest_daily.csv", index_col=0, parse_dates=True)["position"]
foc = pd.read_csv(DATA / "focus_channels_signal.csv", index_col=0, parse_dates=True)["disp_ipca"]
rows = []
fr = [d for d in fridays if d >= px.index[0]]
for k, d in enumerate(fr):
    i0 = px.index.searchsorted(d, side="right")
    if i0 + 1 >= len(px):
        break
    day1 = px.index[i0 + 1]
    row = {"date": d, "COPOM": agg["COPOM"].get(day1, np.nan), "GDP": agg["GDP (tranches)"].get(day1, np.nan),
           "RealRate": agg["Real rate (tranches)"].get(day1, np.nan), "Core": agg["GDP + Real rate"].get(day1, np.nan),
           "Fed": fed_pos.get(day1, np.nan), "disp_ipca": foc.get(d, np.nan)}
    if i0 + H < len(px):
        r = px.iloc[i0 + H] / px.iloc[i0] - 1
        row["fwd21"] = r["SMLL"] - r["IBOV"]
    if k + 1 < len(fr):
        i1 = px.index.searchsorted(fr[k + 1], side="right")
        if i1 < len(px):
            row["fwd_week"] = (1 + spread_d.iloc[i0 + 1:i1 + 1]).prod() - 1
    rows.append(row)
ev = pd.DataFrame(rows).set_index("date").join(sig)
ev.to_csv(DATA / "macro_channels_events.csv")


def nw_ols(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})


res, dec_tab = [], {}
for c in FEATS:
    e = ev[[c, "fwd21", "fwd_week", "COPOM", "GDP", "RealRate", "Fed", "disp_ipca"]].dropna(subset=[c, "fwd21"])
    half = len(e) // 2
    out = {"feature": c, "n": len(e), "start": e.index[0].date(), "end": e.index[-1].date()}
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
    out["t_nextweek"] = sm.OLS(w["fwd_week"] * 100, sm.add_constant(w[c])).fit().tvalues[c]
    for o in ("COPOM", "GDP", "RealRate", "Fed", "disp_ipca"):
        out[f"corr_{o}"] = e[c].corr(e[o])
    out["autocorr_4w"] = sig[c].autocorr(4)
    j = e.dropna(subset=["GDP", "RealRate"])
    out["t_vs_core"] = nw_ols(j["fwd21"] * 100, j[[c, "GDP", "RealRate"]]).tvalues[c]
    res.append(out)
st = pd.DataFrame(res).set_index("feature")
st.to_csv(DATA / "macro_channels_stats.csv", float_format="%.4f")
pd.options.display.width = 260
pd.options.display.max_columns = 40
print(st.round(2).to_string())
corr_cols = FEATS + ["disp_ipca", "Fed", "COPOM", "GDP", "RealRate"]
cm = ev[corr_cols].corr()
cm.to_csv(DATA / "macro_channels_corr.csv", float_format="%.3f")
print("\nSignal correlations (weekly):\n", cm.round(2).to_string())
# EPU vs Focus dispersion head to head; EMBI vs Fed and DXY
for cols in (["epu"], ["disp_ipca"], ["epu", "disp_ipca"], ["embi_trend", "dxy_trend", "Fed"],
             ["embi_level", "dxy_trend", "Fed"]):
    e = ev[cols + ["fwd21"]].dropna()
    m = nw_ols(e["fwd21"] * 100, e[cols])
    print("joint", cols, {k: round(v, 2) for k, v in m.tvalues.items() if k != "const"}, f"n={len(e)}")

# --- plots -------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(a, yonly=False):
    a.grid(True, axis="y" if yonly else "both", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=200, gridspec_kw={"width_ratios": [1.4, 1]})
style(ax)
ypos = np.arange(len(FEATS))[::-1]
for col, colr, mk, lab in (("t_full", INK, "o", "Full sample"), ("t_H1", BLUE, "s", "First half"),
                           ("t_H2", ORANGE, "D", "Second half"),
                           ("t_vs_core", VIOLET, "^", "Full, controlling for GDP + real rate")):
    ax.scatter(st.loc[FEATS, col], ypos, s=46 if col == "t_full" else 30, color=colr, marker=mk, zorder=3, label=lab)
for x0 in (-1.96, 1.96):
    ax.axvline(x0, color=MUTED, lw=1, ls="--")
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(ypos)
ax.set_yticklabels([f"{LABEL[c]}\n{st.loc[c, 'start']:%Y}–{st.loc[c, 'end']:%Y}, n={st.loc[c, 'n']}" for c in FEATS],
                   fontsize=8)
ax.set_xlabel("Newey-West t-stat, fwd 21-session SMLL − IBOV on signal (+ = predicts as hypothesised)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8, labelcolor=INK2, ncol=4)
ax.set_title("Predictive power (dashed = ±1.96)", loc="left", fontsize=9, color=INK2)
ax2.grid(False)
for s in ax2.spines.values():
    s.set_visible(False)
short = {"credit_default": "credit", "hh_dsr": "hh DSR", "epu": "EPU", "dxy_trend": "DXY", "embi_trend": "EMBI Δ",
         "embi_level": "EMBI lvl", "disp_ipca": "IPCA disp", "Fed": "Fed", "COPOM": "COPOM", "GDP": "GDP",
         "RealRate": "Real rate"}
im = ax2.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax2.set_xticks(range(len(cm)))
ax2.set_xticklabels([short[c] for c in cm.columns], rotation=60, ha="right", fontsize=7.5)
ax2.set_yticks(range(len(cm)))
ax2.set_yticklabels([short[c] for c in cm.index], fontsize=7.5)
for i in range(len(cm)):
    for j in range(len(cm)):
        v = cm.values[i, j]
        ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.55 else INK)
ax2.set_title("Signal correlations (weekly, + = long small caps)", loc="left", fontsize=9, color=INK2)
fig.suptitle("Macro channels vs. next 21-session small-cap relative return", x=0.02, ha="left",
             fontsize=13, fontweight="bold")
fig.text(0.02, 0.92, "Each signal oriented so + = long small caps, scaled by past volatility and clipped to ±1; "
         "monthly data used only after its release date. Weekly from Monday close; Newey-West (5 lags).",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(HERE / "macro_channels_summary.png")
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), dpi=200, sharey=True)
for a, c in zip(axes.flat, FEATS):
    style(a, True)
    a.axhline(0, color=MUTED, lw=1)
    d = dec_tab[c]
    s = st.loc[c]
    a.bar(d.index, d.values, width=0.7, color=[ORANGE if v < 0 else AQUA for v in d.values], zorder=3)
    a.set_xticks(range(1, 11))
    a.set_xticklabels([f"D{i}" for i in range(1, 11)], fontsize=7)
    a.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    a.set_title(f"{LABEL[c]}\nNW t: full {s.t_full:+.2f} · H1 {s.t_H1:+.2f} · H2 {s.t_H2:+.2f} · "
                f"D10−D1 {s.D10_D1:+.1f} pp", loc="left", fontsize=8, color=INK2)
for a in axes[:, 0]:
    a.set_ylabel("Mean fwd 21-session\nSMLL − IBOV (pp)")
fig.suptitle("Macro channels: deciles of signal vs. next 21-session small-cap relative return (full sample)",
             x=0.02, ha="left", fontsize=13, fontweight="bold")
fig.text(0.02, 0.935, "D1 = most bearish for small caps, D10 = most bullish.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(HERE / "macro_channels_deciles.png")
print("saved plots")
