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

---

# Long/short grids (`long_short.py`)

The same vault rules, mirrored for shorts. A short cycle opens with a short sale, adds at **+2.25 … +9.85% above** the first fill,
covers 95% at **average entry −0.56%**, and has the same 5-day time stop. Shorts pay a 1%/yr borrow fee.
Each (ticker, side) runs its own cycle, so a stock can hold a long and a short ladder at the same time.

Entry modes:
- **signal**: long when z < −2, short when z > +2 (35-bar SMA)
- **always**: open a new cycle as soon as the book is flat (classic always-on grid)

Underlyings:
- **stocks**: the 10 stocks outright
- **vs basket**: each stock divided by an equal-weight, every-bar-rebalanced basket of the 10. This is market-neutral
  (long stock/short basket or the reverse). It is charged 10 bps per side (two legs) and borrow on the full notional.

| Variant | Total ret | Sharpe | Max DD | Max gross exp. | Cycles | Win rate |
|---|---|---|---|---|---|---|
| stocks, signal, long (previous base) | +1.5% | 0.25 | −2.4% | 77% | 815 | 92% |
| stocks, signal, short | −11.2% | −1.11 | −14.4% | 159% | 854 | 88% |
| stocks, signal, long+short | −9.8% | −0.82 | −15.0% | 159% | 1,669 | 90% |
| stocks, always, long | +1.8% | 0.17 | −5.2% | 128% | 5,801 | 93% |
| stocks, always, short | −14.0% | −0.68 | −26.5% | 200% | 5,562 | 93% |
| stocks, always, long+short | −12.2% | −0.51 | −23.6% | 203% | 11,363 | 93% |
| vs basket, signal, long | +1.0% | 0.24 | −1.3% | 34% | 728 | 86% |
| vs basket, signal, short | −3.2% | −0.85 | −3.8% | 35% | 761 | 85% |
| vs basket, signal, long+short | −2.3% | −0.37 | −4.1% | 37% | 1,489 | 85% |
| vs basket, always, long+short | −11.1% | −0.77 | −15.5% | 211% | 8,254 | 86% |

P&L by side in the two-sided books (sum over cycles, % of equity):

| Book | Long TP | Long time stop | Short TP | Short time stop |
|---|---|---|---|---|
| stocks, signal | +6.1% | −4.7% | +5.2% | **−16.3%** |
| stocks, always | +30.1% | −28.7% | +35.9% | **−49.0%** |
| vs basket, signal | +4.6% | −2.8% | +3.2% | −6.3% |
| vs basket, always | +20.3% | −24.2% | +20.7% | −27.3% |

Sensitivity, signal long+short on stocks: no costs/borrow −7.6%, sell 100% −9.6%, entry z 1.5 −10.3%, z 2.5 −4.3%,
1x lots −3.4%. Without a time stop it made +19.6%, but only by holding losing shorts until they recovered (max gross 162%);
its 100% win rate counts closed cycles only. Vs basket, signal long+short: +1.6% with zero costs, +15.1% with no time stop.

## Takeaways
- **Adding shorts hurts.** Short cycles win 88–93% of the time, but the time stop on short ladders lost 16% of equity
  (signal) and 49% (always-on). The tail is in sharp rallies (Aug 2024, Jan 2026), when several short ladders filled
  completely and gross exposure went to 160–200%.
- **Two-sided books don't hedge:** long and short ladders open on different stocks at different times, so net exposure
  stays near zero on average while gross exposure and tail risk add up.
- **Always-on grids churn.** About 7× more cycles, with costs and time-stop losses outweighing the TP harvest. Only the long
  side roughly breaks even (+1.8%), helped by the market drift.
- **Market-neutral (vs basket) is smaller but still not an edge:** −2.3% for long+short, +1.6% only at zero costs.
  The only positive basket variant is long-only (+1.0%).
- **Nothing beats the long-only signal version on risk-adjusted terms**, and that version is still well below the risk-free rate.
  The edge, such as it is, is long-side mean reversion in a rising market, not grid mechanics.
- Removing the time stop makes every variant look good. That comes from holding losers through recoveries in this
  sample (unbounded tail risk), not from a robust result.

Caveats as above, plus: borrow fees vary by stock and over time, and shorting in B3 needs an available borrow. The basket ratio's
intrabar high/low are estimated conservatively, and an earlier, naive basket built from the previous close showed a fake +43%
for the always-on basket grid, so be wary of grids on synthetic spreads.
