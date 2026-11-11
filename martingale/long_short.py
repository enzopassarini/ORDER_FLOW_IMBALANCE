"""Long/short grid study: vault rules applied long, short, and both, on stocks and on stock-vs-basket."""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest import Params, run_portfolio, stats
from fetch_data import load, relative_to_basket

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 250)

data = load("1h")
rv = relative_to_basket(data)
b = Params()
rvb = replace(b, cost_bps=10.0, borrow_on_all=True)   # two legs: double costs, always a short leg

variants = {
    # outright stocks, entry on z-score signal
    "stocks | signal | long":        (data, b),
    "stocks | signal | short":       (data, replace(b, sides=(-1,))),
    "stocks | signal | long+short":  (data, replace(b, sides=(1, -1))),
    # outright stocks, always-on grid (new cycle whenever flat)
    "stocks | always | long":        (data, replace(b, mode="always")),
    "stocks | always | short":       (data, replace(b, mode="always", sides=(-1,))),
    "stocks | always | long+short":  (data, replace(b, mode="always", sides=(1, -1))),
    # market-neutral: stock / equal-weight basket ratio
    "vs basket | signal | long":       (rv, rvb),
    "vs basket | signal | short":      (rv, replace(rvb, sides=(-1,))),
    "vs basket | signal | long+short": (rv, replace(rvb, sides=(1, -1))),
    "vs basket | always | long+short": (rv, replace(rvb, mode="always", sides=(1, -1))),
}
rows, curves, trades = {}, {}, {}
for name, (d, p) in variants.items():
    eq, tr = run_portfolio(d, p)
    rows[name], curves[name], trades[name] = stats(eq, tr), eq, tr

summary = pd.DataFrame(rows).T
summary.to_csv(OUT / "long_short_summary.csv")
cols = ["total_ret", "cagr", "sharpe", "max_dd", "avg_exposure", "max_exposure", "avg_net_exp",
        "n_trades", "win_rate", "avg_win", "avg_loss", "worst", "tp_rate"]
print("=== Long/short grids (vault rules, 5-day time stop, 1h bars) ===")
print(summary[cols].round(5).to_string())

# per-side P&L inside the two-sided books
print("\n=== P&L by side (% of equity, cycles summed) ===")
side_rows = {}
for name in ["stocks | signal | long+short", "stocks | always | long+short",
             "vs basket | signal | long+short", "vs basket | always | long+short"]:
    tr = trades[name]
    side_rows[name] = tr.groupby(["side", "reason"]).pnl.sum().unstack().fillna(0).stack()
side = pd.DataFrame(side_rows).T
side.to_csv(OUT / "long_short_by_side.csv")
print(side.round(4).to_string())

# robustness of the two-sided signal book
ls = replace(b, sides=(1, -1))
sens = {
    "L+S base":                 (data, ls),
    "L+S no costs, no borrow":  (data, replace(ls, cost_bps=0, borrow_fee=0)),
    "L+S sell 100% at TP":      (data, replace(ls, tp_sell_frac=1.0)),
    "L+S entry z 1.5":          (data, replace(ls, entry_z=1.5)),
    "L+S entry z 2.5":          (data, replace(ls, entry_z=2.5)),
    "L+S no time stop":         (data, replace(ls, time_stop_days=10_000, tp_sell_frac=1.0)),
    "L+S 1x lots":              (data, replace(ls, size_mult=1.0)),
    "vs basket L+S no costs":   (rv, replace(rvb, sides=(1, -1), cost_bps=0, borrow_fee=0)),
    "vs basket L+S no time stop": (rv, replace(rvb, sides=(1, -1), time_stop_days=10_000, tp_sell_frac=1.0)),
}
sens_rows = {k: stats(*run_portfolio(d, p)) for k, (d, p) in sens.items()}
sens_df = pd.DataFrame(sens_rows).T
sens_df.to_csv(OUT / "long_short_sensitivity.csv")
print("\n=== Sensitivity ===")
print(sens_df[["total_ret", "sharpe", "max_dd", "max_exposure", "n_trades", "win_rate", "worst"]].round(5).to_string())

# chart
fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
for ax, group in zip(axes, ["stocks", "vs basket"]):
    for name, s in curves.items():
        if name.startswith(group + " "):
            ax.plot(s.index, s / s.iloc[0] - 1, lw=1.1, label=name.split("| ", 1)[1])
    ax.axhline(0, color="grey", lw=0.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="lower left")
axes[0].set_title("Vault grid on outright stocks (long, short, both) - 10 B3 large caps, 1h bars")
axes[1].set_title("Vault grid on stock / equal-weight basket (market-neutral)")
fig.tight_layout()
fig.savefig(OUT / "long_short_results.png", dpi=130)
