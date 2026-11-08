"""
Long-short SMLL / IBOV backtest of the final core with 5 bp trading costs (no borrow/financing).

Target position (fraction of capital, + = long SMLL / short IBOV):
  unlevered:     q_t = final core signal (COPOM + GDP + real rate, halved against the 3-month trend)
  vol-targeted:  L_t * q_t, L_t = 10% / EWMA(63d half-life) vol of past P&L, capped at 5x, weekly
Legs drift with their index between rebalances (long leg x (1 + r_SMLL), short leg x (1 + r_IBOV)).
On rebalance days both legs are reset to the target, as of the previous close.
Cost = 5 bp x (|change in long leg| + |change in short leg|), as a fraction of capital.
Rebalancing: daily (as in the earlier backtests), weekly, weekly with a no-trade band
(trade only if the position is more than 0.02 away from target, i.e. 1 pp of the long-only SMLL weight).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
COST = 0.0005
fc = pd.read_csv(DATA / "final_core_daily.csv", index_col=0, parse_dates=True)
vt = pd.read_csv(DATA / "vol_target_daily.csv", index_col=0, parse_dates=True)
px = pd.read_csv(DATA / "indices_daily.csv", index_col="date", parse_dates=True).dropna()
rr = px.pct_change()
idx = vt.index  # common sample (vol target needs history)
rS, rI = rr["SMLL"].loc[idx].to_numpy(), rr["IBOV"].loc[idx].to_numpy()
TARGETS = {"Unlevered": fc["filtered_position"].loc[idx].to_numpy(),
           "Vol-targeted 10%": vt["position"].to_numpy()}


def rebal_days(freq):
    if freq == "daily":
        return np.ones(len(idx), bool)
    key = idx.to_period("W-FRI")
    return np.r_[True, key[1:] != key[:-1]]


def simulate(tgt, freq, band=0.0, cost=COST):
    reb = rebal_days(freq)
    L = S = 0.0  # long and short leg as fraction of current capital
    gross, net, pos, traded = [], [], [], []
    for t in range(len(idx)):
        tc = 0.0
        cur = (L + S) / 2
        if reb[t] and (abs(tgt[t] - cur) > band or t == 0):
            nL, nS = tgt[t], tgt[t]
            tr = abs(nL - L) + abs(nS - S)
            tc, L, S = cost * tr, nL, nS
            traded.append(tr)
        else:
            traded.append(0.0)
        pos.append((L + S) / 2)
        g = L * rS[t] - S * rI[t]
        gross.append(g)
        net.append(g - tc)
        cap = 1 + g
        L, S = L * (1 + rS[t]) / cap, S * (1 + rI[t]) / cap  # drift, as fraction of new capital
    return pd.DataFrame({"gross": gross, "net": net, "pos": pos, "traded": traded}, index=idx)


def stats(df, col):
    r = df[col]
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    m = len(r) // 2
    sh = lambda x: x.mean() / x.std() * np.sqrt(252)
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "Vol": r.std() * np.sqrt(252), "Sharpe": sh(r),
            "Sharpe H1": sh(r.iloc[:m]), "Sharpe H2": sh(r.iloc[m:]), "MaxDD": (eq / eq.cummax() - 1).min(),
            "Turnover/yr": df["traded"].sum() / yrs, "Trades/yr": (df["traded"] > 0).sum() / yrs,
            "Cost/yr": (df["gross"] - df["net"]).sum() / yrs}


rows, sims = [], {}
for name, tgt in TARGETS.items():
    for label, freq, band in (("daily", "daily", 0.0), ("weekly", "weekly", 0.0), ("weekly + band", "weekly", 0.02)):
        df = simulate(tgt, freq, band)
        sims[(name, label)] = df
        for col in ("gross", "net"):
            rows.append({"strategy": name, "rebalance": label, "costs": "none" if col == "gross" else "5 bp",
                         **stats(df, col)})
tab = pd.DataFrame(rows)
tab.to_csv(DATA / "ls_costs.csv", index=False, float_format="%.5f")
pd.options.display.width = 220
print(f"{idx[0].date()} -> {idx[-1].date()}")
print(tab.round(4).to_string(index=False))

# --- plot ------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axes = plt.subplots(2, 1, figsize=(11, 9), dpi=200, sharex=True, gridspec_kw={"hspace": 0.22})
for ax, name in zip(axes, TARGETS):
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    g = sims[(name, "daily")]
    eqg = (1 + g["gross"]).cumprod()
    sg = tab[(tab.strategy == name) & (tab.rebalance == "daily") & (tab.costs == "none")].iloc[0]
    ax.plot(eqg.index, eqg, color=MUTED, lw=1.3, ls="--",
            label=f"before costs (daily): Sharpe {sg.Sharpe:.2f}, {sg.CAGR:+.1%}/yr")
    for label, c, lw in (("daily", "#2a78d6", 1.2), ("weekly + band", INK, 2.2)):
        df = sims[(name, label)]
        eq = (1 + df["net"]).cumprod()
        s = tab[(tab.strategy == name) & (tab.rebalance == label) & (tab.costs == "5 bp")].iloc[0]
        ax.plot(eq.index, eq, color=c, lw=lw,
                label=f"after 5 bp, {label}: Sharpe {s.Sharpe:.2f} (H1 {s['Sharpe H1']:.2f}, H2 {s['Sharpe H2']:.2f}), "
                      f"{s.CAGR:+.1%}/yr, max DD {s.MaxDD:.0%}, cost {s['Cost/yr']:.2%}/yr")
        ax.annotate(f"{eq.iloc[-1]:.2f}×", (eq.index[-1], eq.iloc[-1]), xytext=(6, 6 if label != "daily" else -6), textcoords="offset points",
                    fontsize=9, color=c, va="center")
    ax.axhline(1, color=MUTED, lw=1)
    if name != "Unlevered":
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_yticks([1, 1.5, 2, 3, 4])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title(name + (" (position = signal, ~3.8% vol)" if name == "Unlevered" else
                         " (leverage = 10% / EWMA 63d P&L vol, cap 5×)"), loc="left", fontsize=10, color=INK2)
    ax.set_ylabel("Growth of 1")
fig.suptitle("Long-short SMLL − IBOV backtest of the final core with 5 bp trading costs", x=0.07, ha="left",
             fontsize=13, fontweight="bold", y=0.985)
fig.text(0.07, 0.925, "Legs drift between rebalances; cost = 5 bp × (|Δ long leg| + |Δ short leg|). Weekly + band = "
         "rebalance weekly only if the position moved > 0.02 from target.\nNo borrow or financing costs. "
         f"{idx[0]:%b %Y} – {idx[-1]:%b %Y}.", fontsize=8.5, color=INK2, wrap=True)
fig.subplots_adjust(left=0.08, right=0.93, top=0.89, bottom=0.05)
fig.savefig(HERE / "ls_costs.png")
print("saved")
