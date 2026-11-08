"""
Scatter + decile plots (full sample and halves) for one Focus channel signal vs. the
forward 21-session SMLL - IBOV return. Weekly signal as held from Monday close; windows
overlap, so t-stats are Newey-West (5 lags) and decile error bars are widened by sqrt(21/5).
Reads data/focus_channels_events.csv (written by focus_channels.py).

Usage: python focus_channel_plot.py real_rate disp_ipca
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE / "data"
H, NQ, NW_LAGS = 21, 10, 5
TITLE = {"real_rate": ("Ex-ante real rate (level)",
                       "x = −(expected Selic 12m ahead − 12m expected IPCA), demeaned vs. past weeks; "
                       "signal = clip(x / (2 × past std), ±1). High real rate → short small caps."),
         "real_rate_r5": ("Ex-ante real rate vs. rolling 5-year mean",
                          "x = −(expected Selic 12m ahead − 12m expected IPCA) minus its rolling 5-year mean (past weeks); "
                          "signal = clip(x / (3 × rolling 5y std), ±1). High real rate → short small caps."),
         "Real rate (tranches)": ("Ex-ante real rate, 4 tranches × 21 days",
                                  "Weekly signal = clip((x − rolling 5y mean) / (3 × rolling 5y std), ±1), "
                                  "x = −(expected Selic 12m − 12m expected IPCA); position = mean of the last 4 weekly signals."),
         "GDP + Real rate": ("GDP revisions + real rate (both in tranches), equal-risk aggregate",
                            "Position = w_GDP · GDP(tranches) + w_RR · real rate(tranches), weights ∝ 1/past vol of each "
                            "strategy (sum to 1)."),
         "embi_trend": ("EMBI+ Brazil spread, 3-month change (data end Jul 2024)",
                        "x = −(EMBI+ Brazil spread − its value 63 sessions earlier), bp; signal = clip(x / (2 × past std), ±1). "
                        "Widening spreads → short small caps."),
         "nominal_exp_r5": ("Nominal rate: expected Selic 12m ahead vs. rolling 5-year mean",
                            "x = −(Focus expected Selic 12 months ahead) minus its rolling 5-year mean; "
                            "signal = clip(x / (3 × rolling 5y std), ±1). High nominal rate → short small caps."),
         "infl_exp_r5": ("Inflation: 12m expected IPCA vs. rolling 5-year mean",
                         "x = −(Focus 12-month-ahead IPCA expectation) minus its rolling 5-year mean; "
                         "signal = clip(x / (3 × rolling 5y std), ±1). High expected inflation → short small caps."),
         "curve_5y1y_r5": ("Local curve slope (5y − 1y prefixed yield) vs. rolling 5-year mean",
                           "x = −(5y − 1y Tesouro prefixado yield) minus its rolling 5-year mean; signal = clip(x / (3 × rolling "
                           "5y std), ±1). Steep curve → short small caps. Sample from Nov 2010."),
         "curve_2ysel_r5": ("Local curve: 2y prefixed yield − Selic target, vs. rolling 5-year mean",
                            "x = −(2y Tesouro prefixado yield − Selic target) minus its rolling 5-year mean; signal = clip(x / "
                            "(3 × rolling 5y std), ±1). Market pricing hikes → short small caps. Sample from Dec 2009."),
         "curve_y2_r5": ("2-year prefixed (nominal) yield vs. rolling 5-year mean",
                         "x = −(2y Tesouro prefixado yield) minus its rolling 5-year mean; signal = clip(x / (3 × rolling 5y "
                         "std), ±1). High yields → short small caps. Sample from Dec 2009."),
         "disp_ipca": ("IPCA forecast dispersion (revisions)",
                       "x = −(4–8 week change in the cross-sectional std of next-year IPCA forecasts); "
                       "signal = clip(x / (2 × past std), ±1). Rising dispersion → short small caps.")}

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(a, yonly=False):
    a.grid(True, axis="y" if yonly else "both", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


ev_all = pd.read_csv(DATA / "focus_channels_events.csv", index_col=0, parse_dates=True)
# signals as traded in aggregate3.py (e.g. "Real rate (tranches)", "GDP (tranches)", "Aggregate (3)")
ev_agg = pd.read_csv(DATA / "aggregate3_fwd21.csv", index_col=0, parse_dates=True)
ev_mac = pd.read_csv(DATA / "macro_channels_events.csv", index_col=0, parse_dates=True)  # macro_channels.py
for feat in sys.argv[1:] or ["real_rate", "disp_ipca"]:
    src = ev_all if feat in ev_all.columns else ev_mac if feat in ev_mac.columns else ev_agg
    ev = src[[feat, "fwd21"]].dropna()
    half = len(ev) // 2
    samples = {"Full sample": ev,
               f"First half ({ev.index[0]:%b %Y} – {ev.index[half - 1]:%b %Y})": ev.iloc[:half],
               f"Second half ({ev.index[half]:%b %Y} – {ev.index[-1]:%b %Y})": ev.iloc[half:]}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), dpi=200, gridspec_kw={"height_ratios": [1.25, 1]})
    rows = []
    for j, (nm, e) in enumerate(samples.items()):
        x, y = e[feat].to_numpy(), e["fwd21"].to_numpy() * 100
        m = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
        nz = x != 0
        hit = (np.sign(x[nz]) == np.sign(y[nz])).mean()
        corr = np.corrcoef(x, y)[0, 1]
        ax = axes[0, j]
        style(ax)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(0, color=MUTED, lw=1)
        ax.scatter(x, y, s=12, color=BLUE, alpha=0.45, lw=0, zorder=3)
        xs = np.linspace(-1, 1, 20)
        ax.plot(xs, m.params[0] + m.params[1] * xs, color=INK, lw=2, zorder=4)
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-16, 16)
        ax.set_title(f"{nm}\nn={len(e)}  slope {m.params[1]:+.2f} pp  NW SE {m.bse[1]:.2f}  t {m.tvalues[1]:+.2f}  "
                     f"p {m.pvalues[1]:.3f}\nR² {m.rsquared:.3f}  ρ {corr:+.2f}  hit {hit:.0%}  "
                     f"clipped ±1: {(np.abs(x) >= 1).mean():.0%}", loc="left", fontsize=8.5, color=INK2)
        ax.set_xlabel("Signal  ← short small caps · long small caps →")
        if j == 0:
            ax.set_ylabel("SMLL − IBOV fwd 21-session return (pp)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))

        q = pd.qcut(e[feat].rank(method="first"), NQ, labels=range(1, NQ + 1)).astype(int)
        g = e.assign(q=q).groupby("q").agg(n=("fwd21", "size"), lo=(feat, "min"), hi=(feat, "max"),
                                           sig=(feat, "mean"), mean=("fwd21", "mean"), sd=("fwd21", "std"))
        g["mean"] *= 100
        g["se"] = g["sd"] * 100 / np.sqrt(g["n"]) * np.sqrt(H / 5)
        rows.append(g.assign(sample=nm, slope=m.params[1], t_NW=m.tvalues[1]))
        ax = axes[1, j]
        style(ax, True)
        ax.axhline(0, color=MUTED, lw=1)
        ax.bar(g.index, g["mean"], width=0.7, color=[AQUA if s > 0 else ORANGE for s in g["sig"]], zorder=3)
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
    slug = feat.lower().replace(" ", "_").replace("(", "").replace(")", "")
    pd.concat(rows).drop(columns="sd").to_csv(DATA / f"focus_channel_{slug}_deciles.csv", float_format="%.4f")
    from matplotlib.patches import Patch
    fig.legend([Patch(color=ORANGE), Patch(color=AQUA)], ["Short small caps (< 0)", "Long small caps (> 0)"],
               loc="upper left", bbox_to_anchor=(0.045, 0.905), frameon=False, fontsize=8.5, ncol=2, labelcolor=INK2)
    t, sub = TITLE.get(feat, (feat, ""))
    fig.suptitle(f"Focus: {t} vs. next 21-session small-cap relative return", x=0.05, ha="left",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.05, 0.925, sub + "\nWeekly from Monday close; overlapping windows, Newey-West (5 lags); "
             "error bars ±1 SE × √(21/5).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = HERE / f"focus_channel_{feat.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)
