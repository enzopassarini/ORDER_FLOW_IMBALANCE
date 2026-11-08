"""
Effect of doubling the final signal (clipped to [-1, 1]) on position sizes and performance.

  1x: q_t = final signal
  2x: clip(2 * q_t, -1, 1)
Size metrics: average long size (mean q when q > 0), average short size (mean |q| when q < 0),
share of days long / short, share of days at the +-1 clip.
Performance after 5 bp costs, weekly rebalancing with no-trade band:
  long-short (unlevered, band 0.02) and long-only SMLL weight (q + 1) / 2 vs. 50/50 (band 1 pp).
"""
import contextlib
import io

import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import long_only_costs as lo
    import ls_costs as ls

q_full = pd.read_csv(ls.DATA / "final_core_daily.csv", index_col=0, parse_dates=True)["filtered_position"]
rows = []
for mult in (1, 2):
    q = (mult * q_full).clip(-1, 1)
    size = {"Avg long size": q[q > 0].mean(), "Avg short size": -q[q < 0].mean(), "Avg |signal|": q.abs().mean(),
            "Days long": (q > 0).mean(), "Days short": (q < 0).mean(), "Days at ±1": (q.abs() >= 1).mean(),
            "Max long": q.max(), "Max short": -q.min()}
    d_ls = ls.simulate(q.loc[ls.idx].to_numpy(), "weekly", band=0.02)
    s_ls = ls.stats(d_ls, "net")
    tgt = ((q.loc[lo.dates] + 1) / 2).to_numpy()
    p = lo.simulate(tgt, "weekly", band=0.01)
    b = lo.simulate(np.full(len(lo.dates), 0.5), "weekly", band=0.01)
    s_lo = lo.summary(p, b, "net")
    rows.append({"signal": f"{mult}x", **size,
                 "L/S CAGR": s_ls["CAGR"], "L/S vol": s_ls["Vol"], "L/S Sharpe": s_ls["Sharpe"],
                 "L/S Sharpe H2": s_ls["Sharpe H2"], "L/S MaxDD": s_ls["MaxDD"], "L/S cost/yr": s_ls["Cost/yr"],
                 "LO CAGR": s_lo["CAGR"], "LO excess vs 50/50": s_lo["Excess CAGR"], "LO tracking error": s_lo["Tracking error"],
                 "LO IR": s_lo["IR"], "LO SMLL weight range": f"{tgt.min():.0%}–{tgt.max():.0%}"})
tab = pd.DataFrame(rows).set_index("signal").T
tab.to_csv(ls.DATA / "signal_scale_test.csv")
pd.options.display.width = 160
print(f"L/S sample {ls.idx[0].date()} -> {ls.idx[-1].date()}; long-only {lo.dates[0].date()} -> {lo.dates[-1].date()}")
print(tab.to_string())
