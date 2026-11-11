"""Run the vault-martingale study: base, variants, cost sensitivity, charts."""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest import Params, run_portfolio, stats
from fetch_data import load

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 220)

data = load("1h")
base = Params()

variants = {
    "base: 2x lots, sell 95% at TP":      base,
    "sell 100% at TP (no runner)":        replace(base, tp_sell_frac=1.0),
    "sell 90% at TP":                     replace(base, tp_sell_frac=0.90),
    "1.5x lots":                          replace(base, size_mult=1.5),
    "1x lots (flat grid)":                replace(base, size_mult=1.0),
    "no time stop (runners held forever)": replace(base, time_stop_days=10_000),
    "no time stop, sell 100%":            replace(base, time_stop_days=10_000, tp_sell_frac=1.0),
    "first buy 1% of equity (2.75x size)": replace(base, first_buy_pct=0.01),
}
rows, curves = {}, {}
for name, p in variants.items():
    eq, tr = run_portfolio(data, p)
    rows[name] = stats(eq, tr)
    curves[name] = eq
    if name.startswith("base"):
        base_eq, base_tr = eq, tr

bh = pd.DataFrame({t: d["Close"] / d["Close"].iloc[0] / len(data) for t, d in data.items()})
bh = bh.sort_index().ffill().bfill().sum(axis=1)
rows["buy & hold (EW)"] = stats(bh)
summary = pd.DataFrame(rows).T
summary.to_csv(OUT / "summary.csv")
print("=== Strategy comparison (10 B3 large caps, 1h bars, shared equity) ===")
print(summary.round(5).to_string())

by = base_tr.groupby(["reason", "fills"]).agg(
    runner_pnl=("runner_pnl", "sum"),
    n=("pnl", "size"), win_rate=("pnl", lambda x: (x > 0).mean()),
    avg_ret_on_invested=("ret_on_invested", "mean"),
    pnl_pct_equity=("pnl", "sum"))
by.to_csv(OUT / "exit_breakdown.csv")
print("\n=== Base: by exit path and number of fills ===")
print(by.round(5).to_string())

per = base_tr.groupby("ticker").agg(n=("pnl", "size"), win_rate=("pnl", lambda x: (x > 0).mean()),
                                    pnl_pct_equity=("pnl", "sum"), worst=("pnl", "min"),
                                    full_ladders=("fills", lambda x: (x == 6).sum()))
per.to_csv(OUT / "per_ticker.csv")
print("\n=== Base: per ticker ===")
print(per.round(5).to_string())
base_tr.to_csv(OUT / "trades_base.csv", index=False)

cost = pd.DataFrame({f"{b} bps": stats(*run_portfolio(data, replace(base, cost_bps=b)))
                     for b in [0, 2.5, 5, 10]}).T
cost.to_csv(OUT / "costs.csv")
print("\n=== Base vs. costs ===")
print(cost[["total_ret", "sharpe", "max_dd", "win_rate"]].round(5).to_string())

fig, axes = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw=dict(height_ratios=[2, 1, 1.3]))
ax = axes[0]
for name in ["base: 2x lots, sell 95% at TP", "sell 100% at TP (no runner)", "1x lots (flat grid)",
             "no time stop (runners held forever)", "no time stop, sell 100%"]:
    s = curves[name]
    ax.plot(s.index, s / s.iloc[0] - 1, lw=1.2, label=name)
ax.axhline(0, color="grey", lw=0.6)
ax.set_title("Vault martingale - cumulative return, 10 B3 large caps, 1h bars, Dec-2023 to Nov-2026")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
dd = base_eq / base_eq.cummax() - 1
ax.fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4, label="drawdown")
ax2 = ax.twinx()
ax2.plot(base_eq.index, base_eq.attrs["exposure"], color="tab:blue", lw=0.6, alpha=0.6)
ax2.set_ylabel("gross exposure", color="tab:blue")
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Base: drawdown (red) and gross exposure (blue)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

ax = axes[2]
r = base_tr.pnl
ax.hist(r[r > 0], bins=30, color="tab:green", alpha=0.7, label="winners")
ax.hist(r[r <= 0], bins=30, color="tab:red", alpha=0.7, label="losers")
ax.set_title("Base: per-cycle P&L as % of total equity")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "martingale_results.png", dpi=130)
