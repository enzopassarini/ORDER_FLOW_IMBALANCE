# Martingale with a 5-day time stop — 10 B3 large caps (1h bars)

**Data:** Yahoo Finance hourly bars, Dec 2023 – Nov 2026 (~730 sessions, ~4,600 bars per ticker) for
PETR4, VALE3, ITUB4, BBDC4, BBAS3, ABEV3, WEGE3, B3SA3, ITSA4, BPAC11 (`data/`).
Yahoo keeps only ~60 days of 5m/15m data, so 1h is the finest bar with enough history for multi-day trades.

**Strategy (long only, one cycle per ticker at a time, `backtest.py`):**
- Entry: close is more than 2σ below its 35-bar (~5 sessions) SMA → buy 1 unit
- Martingale: every further −2% from the last fill → buy 2× the previous lot (1, 2, 4, 8; max 4 levels)
- Take profit: close ≥ average cost +1%
- **Time stop: exit 5 trading sessions after the first fill**
- Signals on bar close, fills at the next bar's open; 5 bps per side costs
- Each ticker gets 10% of capital; a full ladder (15 units) uses 100% of that sleeve (no leverage)

Run: `python fetch_data.py && python analysis.py` (full output in `results/analysis_output.txt`).

## Results (portfolio of 10)

| Strategy | Total ret | Sharpe | Max DD | Win rate | Trades | Avg exposure |
|---|---|---|---|---|---|---|
| **Martingale 2x, 5-day stop (base)** | **−1.3%** | **−0.32** | −2.7% | 74% | 591 | 2.7% |
| Flat grid 1x, 5-day stop | −2.0% | −0.24 | −4.7% | 71% | 588 | 7.5% |
| Single entry, 5-day stop | −0.2% | 0.02 | −7.6% | 66% | 587 | 21% |
| Martingale 2x, no time stop | +13.6% | 0.91 | −4.9% | 98% | 392 | 23% |
| Buy & hold (equal weight) | +19.4% | 0.41 | −26.3% | – | – | 100% |

## Takeaways
- **The 5-day time stop is what makes it lose.** Take-profit exits (427 trades) earned +6.9% of capital;
  time-stop exits (164 trades, 28%) lost −8.2%. Every 3- or 4-level ladder that hit the time stop was a loss —
  the stop crystallises losses exactly when the position is largest.
- Classic martingale profile: 74% winners averaging +0.16% of the sleeve vs. losers averaging −0.55%, worst −5.7%.
- Without the time stop the same rules made +13.6% — but only because B3 large caps rose over the period
  and positions could wait for recovery. That is the unbounded-risk version of martingale; treat it as
  regime-dependent, not as edge.
- Sensitivity: only 7 of 81 parameter combos (step 1–3%, 3–5 levels, TP 0.5–2%, entry z 1.5–2.5) were profitable,
  all near zero (best +0.2%). Tight steps (1%) with small TPs were the worst.
- Costs matter: from −0.15% at 0 bps to −2.0% at 10 bps per side.
- Low exposure (2.7% avg) because the doubling ladder leaves most capital reserved for levels rarely reached.

Caveats: survivorship (today's large caps), Yahoo intraday quality, hourly-bar granularity, no dividends adjustment intraday.
