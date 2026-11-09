"""
Nonlinear signal-to-position mappings for the final aggregate signal q in [-1, 1].

  Linear         f(q) = q
  Sine           f(q) = sin(pi/2 * q)                 saturating (concave): small signals boosted, ±1 fixed
  Tanh k         f(q) = tanh(k q) / tanh(k), k = 2, 4  S-curve that reaches ±1 faster
  Power p        f(q) = sign(q) |q|^p, p = 1.5, 2     convex: shrinks small signals, keeps the extremes
All mappings keep ±1 at ±1 and 0 at 0, and use only today's q (no fitting).

Tests (after 5 bp, weekly rebalancing + no-trade band):
  - predictive: fwd 21-session SMLL-IBOV on f(q), weekly, Newey-West t
  - long-short (unlevered) Sharpe full / halves; CAGR also shown at the linear version's vol
    (constant multiple, for comparability only)
  - long-only SMLL weight (f(q) + 1) / 2 vs. 50/50: excess and information ratio
"""
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

with contextlib.redirect_stdout(io.StringIO()):
    import long_only_costs as lo
    import ls_costs as ls

DATA, HERE = ls.DATA, ls.HERE
q_all = pd.read_csv(DATA / "final_core_daily.csv", index_col=0, parse_dates=True)["filtered_position"]
MAPS = {
    "Linear": lambda q: q,
    "Sine": lambda q: np.sin(np.pi / 2 * q),
    "Tanh k=2": lambda q: np.tanh(2 * q) / np.tanh(2),
    "Tanh k=4": lambda q: np.tanh(4 * q) / np.tanh(4),
    "Power 1.5 (convex)": lambda q: np.sign(q) * np.abs(q) ** 1.5,
    "Power 2 (convex)": lambda q: np.sign(q) * np.abs(q) ** 2,
}
ev = pd.read_csv(DATA / "final_signal_fwd21.csv", index_col=0, parse_dates=True)
half = len(ev) // 2

rows, curves = [], {}
for name, f in MAPS.items():
    pos = f(q_all)
    # predictive
    x = f(ev["Final signal"])
    y = ev["fwd21"] * 100
    nw = lambda a, b: sm.OLS(b, sm.add_constant(a)).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues.iloc[1]
    # long-short
    d = ls.simulate(pos.loc[ls.idx].to_numpy(), "weekly", band=0.02)
    s = ls.stats(d, "net")
    # long-only
    tgt = ((pos.loc[lo.dates] + 1) / 2).to_numpy()
    p = lo.simulate(tgt, "weekly", band=0.01)
    b = lo.simulate(np.full(len(lo.dates), 0.5), "weekly", band=0.01)
    so = lo.summary(p, b, "net")
    curves[name] = (d, p, b)
    rows.append({"mapping": name, "avg |pos|": pos.abs().mean(), "avg long": pos[pos > 0].mean(),
                 "avg short": -pos[pos < 0].mean(),
                 "NW t full": nw(x, y), "NW t H1": nw(x.iloc[:half], y.iloc[:half]),
                 "NW t H2": nw(x.iloc[half:], y.iloc[half:]),
                 "LS CAGR": s["CAGR"], "LS vol": s["Vol"], "LS Sharpe": s["Sharpe"], "LS Sharpe H1": s["Sharpe H1"],
                 "LS Sharpe H2": s["Sharpe H2"], "LS MaxDD": s["MaxDD"], "LS cost/yr": s["Cost/yr"],
                 "LO excess": so["Excess CAGR"], "LO TE": so["Tracking error"], "LO IR": so["IR"],
                 "LO IR H1": so["IR H1"], "LO IR H2": so["IR H2"]})
tab = pd.DataFrame(rows).set_index("mapping")
vol0 = tab.loc["Linear", "LS vol"]
for name in MAPS:
    r = curves[name][0]["net"] * vol0 / tab.loc[name, "LS vol"]
    tab.loc[name, "LS CAGR @ linear vol"] = (1 + r).prod() ** (252 / len(r)) - 1
    tab.loc[name, "LS MaxDD @ linear vol"] = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
tab.to_csv(DATA / "signal_mapping.csv", float_format="%.4f")
pd.options.display.width = 250
print(tab.round(3).to_string())

# --- plot ------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COL = {"Linear": INK, "Sine": "#2a78d6", "Tanh k=2": "#1baf7a", "Tanh k=4": "#0b6e4a",
       "Power 1.5 (convex)": "#eb6834", "Power 2 (convex)": "#a8381a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig = plt.figure(figsize=(14, 11), dpi=200)
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.2)
ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
for a in (ax1, ax2, ax3, ax4):
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
xs = np.linspace(-1, 1, 401)
ax1b = ax1.twinx()
ax1b.hist(q_all, bins=np.linspace(-1, 1, 81), color=MUTED, alpha=0.25, zorder=0)
ax1b.set_yticks([])
ax1b.spines[["top", "right"]].set_visible(False)
for n, f in MAPS.items():
    ax1.plot(xs, f(xs), color=COL[n], lw=2.2 if n == "Linear" else 1.6, label=n, zorder=3)
ax1.set_zorder(ax1b.get_zorder() + 1)
ax1.patch.set_visible(False)
ax1.axhline(0, color=MUTED, lw=1)
ax1.axvline(0, color=MUTED, lw=1)
ax1.set_xlabel("Signal q")
ax1.set_ylabel("Position f(q)")
ax1.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
ax1.set_title(f"Mappings (grey: distribution of q; 90% of days |q| < {q_all.abs().quantile(0.9):.2f})",
              loc="left", fontsize=10, color=INK2)
names = list(MAPS)
x = np.arange(len(names))
for j, (c, lab, col) in enumerate((("LS Sharpe", "full", INK), ("LS Sharpe H1", "first half", "#9b9a95"),
                                   ("LS Sharpe H2", "second half", "#6250d6"))):
    ax2.bar(x + (j - 1) * 0.26, tab.loc[names, c], width=0.26, color=col, label=lab, zorder=3)
for i, n in enumerate(names):
    ax2.text(i - 0.26, tab.loc[n, "LS Sharpe"] + 0.02, f"{tab.loc[n, 'LS Sharpe']:.2f}", ha="center", fontsize=7.5)
    ax2.text(i + 0.26, tab.loc[n, "LS Sharpe H2"] + 0.02, f"{tab.loc[n, 'LS Sharpe H2']:.2f}", ha="center", fontsize=7.5)
ax2.set_xticks(x)
ax2.set_xticklabels([n.replace(" (convex)", "\n(convex)") for n in names], fontsize=8.5)
ax2.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3)
ax2.set_ylim(0, 1.35)
ax2.set_title("Long-short Sharpe after 5 bp (unlevered)", loc="left", fontsize=10, color=INK2)
for n in names:
    d = curves[n][0]
    r = d["net"] * vol0 / tab.loc[n, "LS vol"]
    eq = (1 + r).cumprod()
    ax3.plot(eq.index, eq, color=COL[n], lw=2.2 if n == "Linear" else 1.2,
             label=f"{n}: CAGR {tab.loc[n, 'LS CAGR @ linear vol']:+.1%}, max DD {tab.loc[n, 'LS MaxDD @ linear vol']:.0%}")
ax3.axhline(1, color=MUTED, lw=1)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
ax3.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
ax3.set_title(f"Long-short, each scaled to the linear version's vol ({vol0:.1%})", loc="left", fontsize=10, color=INK2)
for n in names:
    d, p, b = curves[n]
    rel = ((1 + p["net"]).cumprod() / (1 + b["net"]).cumprod() - 1) * 100
    ax4.plot(rel.index, rel, color=COL[n], lw=2.2 if n == "Linear" else 1.2,
             label=f"{n}: {tab.loc[n, 'LO excess']:+.1%}/yr, TE {tab.loc[n, 'LO TE']:.1%}, IR {tab.loc[n, 'LO IR']:.2f}")
ax4.axhline(0, color=MUTED, lw=1)
ax4.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
ax4.set_ylabel("Relative to 50/50 (%)")
ax4.set_title("Long-only SMLL weight (f(q) + 1) / 2 vs. 50/50", loc="left", fontsize=10, color=INK2)
fig.suptitle("Signal-to-position mapping: linear vs. saturating (sine, tanh) vs. convex (power)", x=0.04, ha="left",
             fontsize=14, fontweight="bold", y=0.975)
fig.text(0.04, 0.945, "Final aggregate signal (COPOM + GDP + real rate, trend filter). After 5 bp, weekly rebalancing "
         "with no-trade band; 2007/2008 – Nov 2026.", fontsize=9, color=INK2)
fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.05)
fig.savefig(HERE / "signal_mapping.png")
print("saved")
