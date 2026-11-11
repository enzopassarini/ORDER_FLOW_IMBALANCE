# Vault martingale with a 5-day time stop — 10 B3 large caps (1h bars)

**Data:** Yahoo Finance hourly bars, Dec 2023 – Nov 2026 (~730 sessions, ~4,600 bars per ticker) for
PETR4, VALE3, ITUB4, BBDC4, BBAS3, ABEV3, WEGE3, B3SA3, ITSA4, BPAC11 (`data/`).
Yahoo keeps only ~60 days of 5m/15m data, so 1h is the finest bar with enough history for multi-day trades.

## Rules (`backtest.py`, `Params`)
One shared-equity portfolio; each ticker runs one cycle at a time.

| Rule | Setting |
|---|---|
| Entry | close more than 2σ below its 35-bar SMA → first buy at the next bar's open |
| First buy | **0.364% of current total equity** |
| Adds | resting limit buys at **−2.25%, −3.54%, −5.04%, −7.00%, −9.85% below the first fill** |
| Add size | each lot 2× the previous (0.364 → 0.73 → 1.46 → 2.91 → 5.82 → 11.65%); full ladder = 22.9% of equity per ticker *(assumed — not specified)* |
| Take profit | resting limit sell at **average entry +0.56%**, sells **95%** ("almost everything") *(95% assumed)* |
| Runner | the leftover 5% is held aside, does not block a new cycle, and is sold at its own time stop |
| Time stop | **5 trading sessions** after the first fill → sell everything at the next open |
| Fills | limits fill when the bar's Low/High touches the level (at the open on a gap). If an add fills, TP waits for the next bar. |
| Costs | 5 bps per side; average entry includes buy costs |
| Leverage | not capped (no financing cost modelled); gross exposure is reported |

Run: `python fetch_data.py && python analysis.py` (full output in `results/analysis_output.txt`).

## Results

| Variant | Total ret | CAGR | Sharpe | Max DD | Win rate | Cycles | Avg / max exposure |
|---|---|---|---|---|---|---|---|
| **Base (2x lots, sell 95%, 5-day stop)** | **+1.5%** | **+0.5%** | **0.25** | **−2.4%** | **92%** | 815 | 1.7% / 77% |
| Sell 100% at TP | +1.7% | +0.6% | 0.29 | −2.4% | 93% | 815 | 1.6% / 77% |
| Sell 90% at TP | +1.4% | +0.5% | 0.22 | −2.4% | 86% | 815 | 1.8% / 78% |
| 1.5x lots | 0.0% | 0.0% | 0.00 | −1.2% | 91% | 805 | 1.2% / 30% |
| 1x lots (flat grid) | −0.7% | −0.2% | −0.51 | −1.0% | 87% | 794 | 0.9% / 12% |
| No time stop, sell 100% | +8.6% | +2.9% | 0.84 | −3.3% | 100% | 762 | 5.5% / 111% |
| No time stop, runners held forever | +14.8% | +4.9% | 0.57 | −13.4% | – | 762 | 30% / 160% |
| First buy 1% (2.75x size) | +4.0% | +1.4% | 0.25 | −6.5% | 92% | 815 | 4.6% / 219% |
| Buy & hold (equal weight) | +19.4% | +6.3% | 0.41 | −26.3% | – | – | 100% |

Costs (base): +2.3% at 0 bps, +1.9% at 2.5 bps, +1.5% at 5 bps, +0.9% at 10 bps per side.

## Takeaways
- **Barely positive:** +1.5% over ~3 years (+0.5%/yr), well below the Brazilian risk-free rate.
  Capital is idle most of the time (1.7% average exposure), so the 0.364% first buy is the main reason returns are tiny.
- **Typical martingale payoff:** 92% of cycles win, averaging +0.009% of equity; losers average −0.08% (9× larger).
  Take-profit cycles made +6.6% of equity and time-stop cycles lost −5.1%; the 4 full-ladder time stops alone cost −2.5%.
- **The time stop remains the main drag**, as in the previous version. Without it, the same rules made +8.6%, but only because B3 rose over the
  period, and exposure reached 111% of equity.
- **Doubling the lots is what makes it profitable:** 1.5x lots break even and flat lots lose money, because the late, large
  lots pull the average entry close enough to price for the +0.56% target to be hit.
- **The 5% runner adds nothing:** selling 100% did slightly better (+1.7% vs. +1.5%).
- **Tail risk is concentrated:** the whole max drawdown came from one market-wide selloff (Aug 2026), when several tickers
  hit full ladders together and gross exposure jumped from ~2% to 77%. Scaling the first buy to 1% would have taken exposure
  to 219% of equity.

Caveats: survivorship (today's large caps), Yahoo intraday quality, hourly bars (intrabar order of fills unknown),
no margin financing cost.
