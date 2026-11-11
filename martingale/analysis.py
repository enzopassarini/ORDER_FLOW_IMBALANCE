"""Run the martingale study: baseline, benchmarks, sensitivity grid, charts."""
from dataclasses import replace
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest import Params, run_portfolio, stats
from fetch_data import load

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 200)

data = load("1h")
base = Params()

# --- 1. Baseline + variants that isolate the martingale effect ------------
variants = {
    "martingale 2x (base)": base,
    "flat grid 1x":         replace(base, multiplier=1.0),
    "single entry":         replace(base, max_levels=1),
    "martingale, no time stop": replace(base, time_stop_days=10_000),
    "martingale 2x, 6 levels":  replace(base, max_levels=6),
}
rows, curves = {}, {}
for name, p in variants.items():
    port, _, tr = run_portfolio(data, p)
    rows[name] = stats(port, tr)
    curves[name] = port
    if name.startswith("martingale 2x (base)"):
        base_port, base_by_tkr, base_tr = port, _, tr

bh = pd.DataFrame({t: d["Close"] / d["Close"].iloc[0] / len(data) for t, d in data.items()})
bh = bh.sort_index().ffill().bfill().sum(axis=1)
rows["buy & hold (EW)"] = stats(bh)
curves["buy & hold (EW)"] = bh
summary = pd.DataFrame(rows).T
summary.to_csv(OUT / "summary.csv")
print("\n=== Strategy comparison (portfolio of 10, 1h bars) ===")
print(summary.round(4).to_string())

# --- 2. Per-ticker breakdown of the base strategy ------------------------
per = []
for t in data:
    tr = base_tr[base_tr.ticker == t]
    eq = base_by_tkr[t]
    per.append(dict(ticker=t, n_trades=len(tr), win_rate=(tr.pnl > 0).mean(),
                    ret_on_sleeve=eq.iloc[-1] / eq.iloc[0] - 1,
                    max_dd=(eq / eq.cummax() - 1).min(),
                    worst_trade=tr.ret_on_sleeve.min(),
                    time_stops=(tr.reason == "time_stop").sum(),
                    full_ladders=(tr.levels == base.max_levels).sum(),
                    buy_hold=data[t]["Close"].iloc[-1] / data[t]["Close"].iloc[0] - 1))
per = pd.DataFrame(per).set_index("ticker")
per.to_csv(OUT / "per_ticker.csv")
print("\n=== Per ticker (base) ===")
print(per.round(4).to_string())

# --- 3. Exit reasons / ladder depth ---------------------------------------
by = base_tr.groupby(["reason", "levels"]).agg(
    n=("pnl", "size"), win_rate=("pnl", lambda x: (x > 0).mean()),
    avg_ret_on_invested=("ret_on_invested", "mean"),
    total_pnl_pct_of_capital=("pnl", "sum"))
by.to_csv(OUT / "exit_breakdown.csv")
print("\n=== Base: by exit reason and ladder depth ===")
print(by.round(4).to_string())
base_tr.to_csv(OUT / "trades_base.csv", index=False)

# --- 4. Sensitivity grid ---------------------------------------------------
grid = []
for step, lv, tp, ez in product([0.01, 0.02, 0.03], [3, 4, 5], [0.005, 0.01, 0.02], [1.5, 2.0, 2.5]):
    p = replace(base, step_pct=step, max_levels=lv, tp_pct=tp, entry_z=ez)
    port, _, tr = run_portfolio(data, p)
    grid.append(dict(step=step, levels=lv, tp=tp, entry_z=ez) | stats(port, tr))
grid = pd.DataFrame(grid)
grid.to_csv(OUT / "grid.csv", index=False)
print(f"\n=== Grid: {len(grid)} combos; {(grid.total_ret > 0).mean():.0%} profitable ===")
cols = ["step", "levels", "tp", "entry_z", "total_ret", "sharpe", "max_dd",
        "win_rate", "pct_time_stop", "avg_exposure"]
print("Top 5 by Sharpe:\n", grid.sort_values("sharpe", ascending=False)[cols].head().round(4).to_string())
print("Bottom 5 by Sharpe:\n", grid.sort_values("sharpe")[cols].head().round(4).to_string())

# --- 5. Cost sensitivity ----------------------------------------------------
cost_rows = {}
for bps in [0, 2.5, 5, 10]:
    port, _, tr = run_portfolio(data, replace(base, cost_bps=bps))
    cost_rows[f"{bps} bps"] = stats(port, tr)
cost = pd.DataFrame(cost_rows).T
cost.to_csv(OUT / "costs.csv")
print("\n=== Base vs. costs ===")
print(cost[["total_ret", "sharpe", "max_dd"]].round(4).to_string())

# --- 6. Charts --------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw=dict(height_ratios=[2, 1, 1.3]))
ax = axes[0]
for name in ["martingale 2x (base)", "flat grid 1x", "single entry", "martingale, no time stop"]:
    s = curves[name]
    ax.plot(s.index, s / s.iloc[0] - 1, lw=1.2, label=name)
ax.axhline(0, color="grey", lw=0.6)
ax.set_title("Cumulative return - 10 B3 large caps, 1h bars, Dec-2023 to Nov-2026")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(loc="lower left", fontsize=8)

ax = axes[1]
dd = base_port / base_port.cummax() - 1
ax.fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4)
ax.set_title("Martingale (base) drawdown")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

ax = axes[2]
r = base_tr.ret_on_sleeve
ax.hist(r[r > 0], bins=40, color="tab:green", alpha=0.7, label="winners")
ax.hist(r[r <= 0], bins=40, color="tab:red", alpha=0.7, label="losers")
ax.set_title("Per-trade P&L as % of ticker sleeve (many small wins, few large losses)")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "martingale_results.png", dpi=130)
print(f"\nSaved results to {OUT}")
