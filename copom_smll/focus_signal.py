"""
Focus GDP-revision feature for SMLL vs. IBOV.

Feature (weekly, Friday survey date t):
  y        = year(t) + 1                       (next calendar year)
  rev_k(t) = Median_t[GDP_y] - Median_{t-k weeks}[GDP_y]   (same target year y,
             so the January roll to a new target year never creates a jump)
  rev_4_8  = (rev_4 + rev_8) / 2               (the "4-8 week" revision)
Signal in [-1, 1]: clip(rev / (2 * expanding std of rev), -1, 1); std uses only
past weeks (min 52), so the scaling is causal. Upgrade -> positive -> long small caps.

Timing: the Focus report with Friday's data is published Monday morning, so the
position is entered at the close of the first session after Friday (Monday) and
held one week. Forward returns: close of first session after Friday -> 21 sessions later.
Weekly observations with 21-session windows overlap, so t-stats are Newey-West (5 lags).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
H = 21

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
})


def style(ax):
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --- feature -------------------------------------------------------------------
w = pd.read_csv(DATA / "focus_gdp_weekly.csv", index_col=0, parse_dates=True)
w.columns = w.columns.astype(int)


def revision(k):
    out = []
    for i, d in enumerate(w.index):
        y = d.year + 1
        out.append(w.iat[i, w.columns.get_loc(y)] - w.iat[i - k, w.columns.get_loc(y)] if i >= k else np.nan)
    return pd.Series(out, index=w.index)


feat = pd.DataFrame({"gdp_next": [w.at[d, d.year + 1] for d in w.index],
                     "rev_4": revision(4), "rev_8": revision(8)}, index=w.index)
feat["rev_4_8"] = feat[["rev_4", "rev_8"]].mean(axis=1)
for c in ["rev_4", "rev_8", "rev_4_8"]:
    scale = 2 * feat[c].expanding(min_periods=52).std().shift(1)
    feat[f"sig_{c}"] = (feat[c] / scale).clip(-1, 1)

# --- forward returns ---------------------------------------------------------------
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
feat = feat[feat.index >= px.index[0]]
rows = []
for d in feat.index:
    i0 = px.index.searchsorted(d, side="right")
    if i0 + H >= len(px):
        continue
    r = px.iloc[i0 + H] / px.iloc[i0] - 1
    rows.append({"date": d, "entry": px.index[i0], "fwd_rel": r["SMLL"] - r["IBOV"]})
fwd = pd.DataFrame(rows).set_index("date")
ev = feat.join(fwd, how="inner").dropna()
ev.to_csv(DATA / "focus_fwd21.csv")


def nw(x, y, lags=5):
    m = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.params[1], m.tvalues[1]


stats = []
for c in ["rev_4", "rev_8", "rev_4_8"]:
    for name, x in [("raw", ev[f"sig_{c}"]),
                    ("demeaned", ev[f"sig_{c}"] - ev[f"sig_{c}"].expanding().mean().shift(1).fillna(0))]:
        y = ev["fwd_rel"] * 100
        b, t = nw(x.to_numpy(), y.to_numpy())
        nz = x != 0
        nonov = ev.iloc[::4]  # every 4th week: non-overlapping windows
        xo = nonov[f"sig_{c}"]
        stats.append({"feature": c, "version": name, "n": len(x), "slope_pp": b, "corr": np.corrcoef(x, y)[0, 1],
                      "t_NW": t, "hit": (np.sign(x[nz]) == np.sign(y[nz])).mean(),
                      "corr_nonoverlap": np.corrcoef(xo, nonov["fwd_rel"])[0, 1],
                      "mean_signal": x.mean(), "share_zero": (x == 0).mean()})
st = pd.DataFrame(stats)
st.round(3).to_csv(DATA / "focus_stats.csv", index=False)
pd.options.display.width = 200
print(st.round(3).to_string(index=False))

# split halves for the main feature
half = len(ev) // 2
for nm, part in [("first half", ev.iloc[:half]), ("second half", ev.iloc[half:])]:
    b, t = nw(part["sig_rev_4_8"].to_numpy(), part["fwd_rel"].to_numpy() * 100)
    print(f"rev_4_8 {nm} {part.index[0].date()}..{part.index[-1].date()}: slope {b:+.2f} pp, t_NW {t:+.2f}, "
          f"corr {part['sig_rev_4_8'].corr(part['fwd_rel']):+.3f}")

# --- plot 1: scatter -----------------------------------------------------------------
MAIN = "sig_rev_4_8"
x, y = ev[MAIN].to_numpy(), ev["fwd_rel"].to_numpy() * 100
b, t = nw(x, y)
a = np.polyfit(x, y, 1)[1]
c = np.corrcoef(x, y)[0, 1]
nz = x != 0
hit = (np.sign(x[nz]) == np.sign(y[nz])).mean()
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
style(ax)
ax.axhline(0, color=MUTED, lw=1, zorder=1)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
ax.scatter(x, y, s=14, color=BLUE, alpha=0.45, lw=0, zorder=3, label="One weekly Focus survey")
xs = np.linspace(-1, 1, 20)
ax.plot(xs, b * xs + a, color=INK, lw=2, zorder=4,
        label=f"OLS fit: {b:+.2f} pp per unit signal  (ρ = {c:.2f}, Newey-West t = {t:.2f})")
ax.set_xlabel("Signal = clip(4–8 wk revision to next-year GDP median / (2 × past std), −1, 1)   "
              "← downgrade · upgrade →", fontsize=9)
ax.set_ylabel("SMLL − IBOV forward 21-session return (pp)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=INK2)
fig.suptitle("Focus GDP revisions vs. next 21-session small-cap relative return", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"{len(ev)} weekly surveys, {ev.index[0]:%b %Y} – {ev.index[-1]:%b %Y}. Window: close of first session "
             f"after Friday survey → 21 sessions later (overlapping).\n"
             f"Hit rate when signal ≠ 0: {hit:.0%}.  {(x == 0).mean():.0%} of weeks have no revision (signal 0).",
             loc="left", fontsize=9, color=INK2, pad=10)
fig.text(0.07, 0.01, "Sources: BCB Focus survey (median, 30-day base), B3. Revisions compare the same target year.",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(HERE / "focus_vs_fwd21.png")
plt.close(fig)

# --- plot 2: deciles -------------------------------------------------------------------
ev["decile"] = pd.qcut(ev[MAIN].rank(method="first"), 10, labels=range(1, 11)).astype(int)
dec = ev.groupby("decile").agg(n=("fwd_rel", "size"), lo=(MAIN, "min"), hi=(MAIN, "max"), sig=(MAIN, "mean"),
                               zero_share=(MAIN, lambda v: (v.abs() < 1e-9).mean()),
                               mean=("fwd_rel", "mean"), std=("fwd_rel", "std"))
dec["mean"] *= 100
# overlap inflates naive SE; scale by sqrt(H/5) ~ number of overlapping weeks per window
dec["se"] = dec["std"] * 100 / np.sqrt(dec["n"]) * np.sqrt(H / 5)
dec.drop(columns="std").round(3).to_csv(DATA / "focus_deciles_fwd21.csv")
print(dec.drop(columns="std").round(2))
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
style(ax)
ax.grid(False, axis="x")
ax.axhline(0, color=MUTED, lw=1, zorder=1)
colors = [GRAY if z > 0.5 else AQUA if s > 0 else ORANGE for s, z in zip(dec["sig"], dec["zero_share"])]
ax.bar(dec.index, dec["mean"], width=0.72, color=colors, zorder=3)
ax.errorbar(dec.index, dec["mean"], yerr=dec["se"], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
for d, r in dec.iterrows():
    off = r["se"] + 0.08
    ax.text(d, r["mean"] + (off if r["mean"] >= 0 else -off), f"{r['mean']:+.2f}", ha="center",
            va="bottom" if r["mean"] >= 0 else "top", fontsize=8.5)
fmt = lambda v: f"{v:+.2f}" if abs(v) > 1e-9 else "0"
ax.set_xticks(dec.index)
ax.set_xticklabels([f"D{d}\n{fmt(r.lo)}" + ("" if abs(r.hi - r.lo) < 1e-9 else f"…{fmt(r.hi)}") + f"\nn={int(r.n)}"
                    for d, r in dec.iterrows()], fontsize=8)
ax.set_ylim((dec["mean"] - dec["se"]).min() - 0.4, (dec["mean"] + dec["se"]).max() + 0.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}" if v else "0"))
ax.set_ylabel("Mean SMLL − IBOV fwd 21-session return (pp)")
ax.set_xlabel("Signal decile (signal range, count)    ← biggest downgrades · biggest upgrades →")
from matplotlib.patches import Patch
ax.legend([Patch(color=ORANGE), Patch(color=GRAY), Patch(color=AQUA)],
          ["Downgrade (signal < 0)", "Mostly no revision (0)", "Upgrade (signal > 0)"],
          loc="best", frameon=False, fontsize=9, labelcolor=INK2)
zero_dec = dec.index[dec["zero_share"] > 0.5]
fig.suptitle("Focus GDP-revision deciles vs. next 21-session small-cap relative return", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"D10 − D1 = {dec.loc[10, 'mean'] - dec.loc[1, 'mean']:+.2f} pp. Error bars ±1 SE, widened for "
             f"overlapping windows.\n" + (("D" + "–D".join(map(str, sorted({zero_dec.min(), zero_dec.max()}))) + " are mostly 'no revision' "
             "weeks; ties are split by date, so they differ only by period.") if len(zero_dec) else ""), loc="left", fontsize=9, color=INK2, pad=10)
fig.tight_layout()
fig.savefig(HERE / "focus_deciles_fwd21.png")
plt.close(fig)

# --- backtest: weekly position, plus combination with COPOM --------------------------------------
spread = (px.pct_change()["SMLL"] - px.pct_change()["IBOV"]).dropna()
pos = pd.Series(np.nan, index=spread.index)
for d, s in feat[MAIN].dropna().items():
    i0 = px.index.searchsorted(d, side="right")
    if i0 < len(px):
        pos.loc[pos.index > px.index[i0]] = s
copom_bt = pd.read_csv(DATA / "backtest_daily.csv", index_col=0, parse_dates=True)
df = pd.DataFrame({"spread": spread, "focus_pos": pos, "copom_pos": copom_bt["position"]}).dropna()
df["focus"] = df["focus_pos"] * df["spread"]
df["copom"] = df["copom_pos"] * df["spread"]
# equal-risk combination: scale each by its full-sample vol (in-sample scaling, Sharpe-neutral per leg)
vol = df[["focus", "copom"]].std()
df["combined"] = 0.5 * (df["focus"] / vol["focus"] + df["copom"] / vol["copom"]) * vol.mean()
# bias checks: the signal is net short on average (forecasters mostly downgrade) while SMLL lagged IBOV.
df["short_only"] = df["spread"] * df["focus_pos"].mean()  # constant position equal to the average signal
dm = feat[MAIN] - feat[MAIN].expanding(min_periods=52).mean().shift(1)  # causal demeaning
pos_dm = pd.Series(np.nan, index=spread.index)
for d, s_ in dm.dropna().items():
    i0 = px.index.searchsorted(d, side="right")
    if i0 < len(px):
        pos_dm.loc[pos_dm.index > px.index[i0]] = s_
df["focus_demeaned"] = pos_dm.reindex(df.index).fillna(0) * df["spread"]
df.to_csv(DATA / "focus_backtest_daily.csv")


def perf(r):
    eq = (1 + r).cumprod()
    return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "Vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min()}


perf_tab = pd.DataFrame({k: perf(df[k]) for k in ["focus", "focus_demeaned", "copom", "combined", "short_only"]}).T
corr_ret = df["focus"].corr(df["copom"])
corr_pos = df["focus_pos"].corr(df["copom_pos"])
perf_tab.round(3).to_csv(DATA / "focus_backtest_stats.csv")
print(f"\nBacktest {df.index[0].date()} -> {df.index[-1].date()}")
print(perf_tab.round(3))
print(f"corr(daily returns focus, copom) = {corr_ret:.3f}; corr(positions) = {corr_pos:.3f}; "
      f"avg focus position {df['focus_pos'].mean():+.3f}, avg |pos| {df['focus_pos'].abs().mean():.3f}")
for nm, part in [("first half", df.iloc[: len(df) // 2]), ("second half", df.iloc[len(df) // 2:])]:
    print(nm, {k: round(perf(part[k])["Sharpe"], 2) for k in ["focus", "focus_demeaned", "copom", "combined"]})

fig, (ax, axd) = plt.subplots(2, 1, figsize=(10, 7.4), dpi=200, sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
for a_ in (ax, axd):
    style(a_)
names = {"focus": ("Focus GDP revisions", AQUA), "copom": ("COPOM linear", BLUE),
         "combined": ("Equal-risk combination", INK)}
for k, (lab, col) in names.items():
    eq = (1 + df[k]).cumprod()
    ax.plot(eq.index, eq, color=col, lw=2 if k != "combined" else 2.2, label=f"{lab}  (Sharpe {perf_tab.loc[k, 'Sharpe']:.2f})")
    ax.annotate(f"{eq.iloc[-1]:.2f}×", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=INK2, va="center")
    axd.plot(eq.index, (eq / eq.cummax() - 1) * 100, color=col, lw=1.0)
ax.set_yscale("log")
ax.set_yticks([0.7, 1, 1.5, 2, 3])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.axhline(1, color=MUTED, lw=1)
ax.set_ylabel("Growth of 1 (log scale)")
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
axd.set_ylabel("Drawdown (%)")
fig.suptitle("Focus GDP-revision backtest vs. COPOM signal: SMLL vs. IBOV spread", x=0.07, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
ax.set_title(f"Focus: weekly position = signal, entered Monday close, no costs.  Return correlation with COPOM: "
             f"{corr_ret:+.2f}.\nCombination scales each leg to equal volatility (full-sample vol, so slightly "
             f"in-sample).", loc="left", fontsize=8.5, color=INK2, pad=10)
fig.text(0.07, 0.01, "Daily-rebalanced dollar-neutral spread, return = position × (r_SMLL − r_IBOV). "
         "Sources: BCB Focus, BCB COPOM, B3.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.08, right=0.93, top=0.86, bottom=0.07)
fig.savefig(HERE / "focus_backtest.png")
plt.close(fig)
print("saved plots")
