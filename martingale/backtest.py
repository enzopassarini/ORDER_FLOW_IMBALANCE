"""
Long-only martingale (doubling-down) mean-reversion backtest on 1h bars,
with a 5-trading-day time stop.

Rules (per ticker, one cycle at a time):
  Entry     : close z-score vs. a rolling SMA < -ENTRY_Z  -> buy 1 unit
  Martingale: each further drop of STEP_PCT below the last fill price
              -> buy 2x the previous lot (1, 2, 4, 8, ...), up to MAX_LEVELS fills
  Exit      : close >= average cost * (1 + TP_PCT)          -> take profit
              5 trading sessions after the first fill        -> time stop
              (optional) close <= avg cost * (1 - HARD_STOP) -> stop loss
  Execution : signals on bar close, fills at the NEXT bar's open (no look-ahead).
  Costs     : COST_BPS per side on traded notional.
  Sizing    : each ticker owns 1/N of capital; the unit is sized so a full
              ladder (1+2+4+...) uses exactly 100% of that sleeve (no leverage).
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Params:
    sma_len: int = 35          # ~5 sessions of 7 hourly bars
    entry_z: float = 2.0
    step_pct: float = 0.02     # add every -2% from last fill
    max_levels: int = 4        # 1+2+4+8 = 15 units max
    multiplier: float = 2.0    # 2.0 = classic martingale, 1.0 = flat grid
    tp_pct: float = 0.01       # +1% over average cost
    time_stop_days: int = 5
    hard_stop: float | None = None
    cost_bps: float = 5.0      # per side (fees + slippage)


def run_ticker(df: pd.DataFrame, p: Params, capital: float = 1.0):
    o, c = df["Open"].to_numpy(), df["Close"].to_numpy()
    sma = df["Close"].rolling(p.sma_len).mean()
    sd = df["Close"].rolling(p.sma_len).std()
    z = ((df["Close"] - sma) / sd).to_numpy()
    session = pd.factorize(df.index.normalize())[0]   # trading-session index
    n = len(df)
    cost = p.cost_bps / 1e4
    lot_weights = p.multiplier ** np.arange(p.max_levels)
    unit_cash = capital / lot_weights.sum()

    cash, shares = capital, 0.0
    level, last_fill, start_sess, entry_i = 0, np.nan, -1, -1
    cost_basis = 0.0
    pending = None                 # ("buy", cash_amount) or ("sell",) for next open
    equity = np.empty(n)
    exposure = np.empty(n)
    trades = []

    for i in range(n):
        # 1) execute order decided on previous bar's close, at this bar's open
        if pending is not None:
            if pending[0] == "buy":
                amt = min(pending[1], cash / (1 + cost))
                q = amt / o[i]
                shares += q
                cash -= amt * (1 + cost)
                cost_basis += amt * (1 + cost)
                last_fill = o[i]
                if level == 0:
                    start_sess, entry_i = session[i], i
                level += 1
            else:  # sell all
                proceeds = shares * o[i] * (1 - cost)
                trades.append(dict(
                    entry_time=df.index[entry_i], exit_time=df.index[i],
                    levels=level, invested=cost_basis, pnl=proceeds - cost_basis,
                    ret_on_invested=proceeds / cost_basis - 1,
                    ret_on_sleeve=(proceeds - cost_basis) / capital,
                    reason=pending[1], bars=i - entry_i))
                cash += proceeds
                shares, level, cost_basis, last_fill = 0.0, 0, 0.0, np.nan
            pending = None

        equity[i] = cash + shares * c[i]
        exposure[i] = shares * c[i]
        if i == n - 1 or np.isnan(z[i]):
            continue

        # 2) decide on this bar's close
        if level == 0:
            if z[i] < -p.entry_z:
                pending = ("buy", unit_cash * lot_weights[0])
        else:
            avg = cost_basis / shares
            if c[i] >= avg * (1 + p.tp_pct):
                pending = ("sell", "take_profit")
            elif session[i] - start_sess >= p.time_stop_days:
                pending = ("sell", "time_stop")
            elif p.hard_stop is not None and c[i] <= avg * (1 - p.hard_stop):
                pending = ("sell", "hard_stop")
            elif level < p.max_levels and c[i] <= last_fill * (1 - p.step_pct):
                pending = ("buy", unit_cash * lot_weights[level])

    eq = pd.Series(equity, index=df.index)
    eq.attrs["exposure"] = pd.Series(exposure, index=df.index)
    return eq, pd.DataFrame(trades)


def run_portfolio(data: dict[str, pd.DataFrame], p: Params):
    cap = 1.0 / len(data)
    eqs, exps, trades = {}, {}, []
    for t, df in data.items():
        eq, tr = run_ticker(df, p, cap)
        eqs[t], exps[t] = eq, eq.attrs["exposure"]
        if len(tr):
            tr.insert(0, "ticker", t)
            trades.append(tr)
    eq = pd.DataFrame(eqs).sort_index().ffill().bfill()
    port = eq.sum(axis=1)
    port.attrs["exposure"] = pd.DataFrame(exps).sort_index().ffill().fillna(0).sum(axis=1) / port
    return port, eq, (pd.concat(trades, ignore_index=True) if trades else pd.DataFrame())


def stats(eq: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    daily = eq.groupby(eq.index.normalize()).last()
    r = daily.pct_change().dropna()
    yrs = len(daily) / 252
    dd = eq / eq.cummax() - 1
    s = dict(total_ret=eq.iloc[-1] / eq.iloc[0] - 1,
             cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
             vol=r.std() * np.sqrt(252),
             sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
             max_dd=dd.min())
    if "exposure" in eq.attrs:
        s["avg_exposure"] = eq.attrs["exposure"].mean()
        s["max_exposure"] = eq.attrs["exposure"].max()
    if trades is not None and len(trades):
        s |= dict(n_trades=len(trades),
                  win_rate=(trades.pnl > 0).mean(),
                  avg_win=trades.loc[trades.pnl > 0, "ret_on_sleeve"].mean(),
                  avg_loss=trades.loc[trades.pnl <= 0, "ret_on_sleeve"].mean(),
                  pct_time_stop=(trades.reason == "time_stop").mean(),
                  pct_full_ladder=(trades.levels == trades.levels.max()).mean())
    return s


if __name__ == "__main__":
    from fetch_data import load
    data = load("1h")
    p = Params()
    port, _, tr = run_portfolio(data, p)
    print(asdict(p))
    print(pd.Series(stats(port, tr)).round(4))
