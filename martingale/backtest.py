"""
Long-only martingale ("vault") backtest on 1h bars, run as one shared-cash portfolio.

Rules per ticker (one cycle at a time):
  Entry      : close z-score vs. rolling SMA < -ENTRY_Z -> first buy at next bar's open,
               sized FIRST_BUY_PCT of current total portfolio equity.
  Adds       : resting limit buys at fixed offsets below the FIRST fill price:
               -2.25%, -3.54%, -5.04%, -7.00%, -9.85%. Lot k = first lot * SIZE_MULT**k.
  Take profit: resting limit sell at average entry * (1 + 0.56%); sells TP_SELL_FRAC
               ("almost everything") and closes the cycle. The leftover runner is held
               on the side (it does not block a new cycle) and sold at its time stop.
               Its P&L is credited to the cycle that created it.
  Time stop  : TIME_STOP_DAYS trading sessions after the first fill -> sell all at next open.
  Fills      : limit orders fill intrabar when Low/High touch the level (at the open if
               the bar gaps through). If an add fills on a bar, take profit is not checked
               until the next bar (conservative intrabar ordering).
  Costs      : COST_BPS per side. Average entry includes buy costs.
  Leverage   : not capped; cash may go negative. Gross exposure is reported (no financing cost).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Params:
    sma_len: int = 35
    entry_z: float = 2.0
    add_levels: tuple = (0.0225, 0.0354, 0.0504, 0.07, 0.0985)
    size_mult: float = 2.0
    first_buy_pct: float = 0.00364
    tp_pct: float = 0.0056
    tp_sell_frac: float = 0.95
    time_stop_days: int = 5
    cost_bps: float = 5.0


@dataclass
class _Pos:
    shares: float = 0.0
    basis: float = 0.0          # remaining cost (cash paid incl. fees) of held shares
    spent: float = 0.0          # all cash paid in this cycle
    received: float = 0.0       # all cash received in this cycle
    first_fill: float = np.nan
    base_cash: float = 0.0
    fills: int = 0
    start_sess: int = -1
    entry_time: object = None
    tp_hit: bool = False
    max_invested: float = 0.0


def run_portfolio(data: dict[str, pd.DataFrame], p: Params, capital: float = 1.0):
    tickers = list(data)
    idx = sorted(set().union(*[d.index for d in data.values()]))
    idx = pd.DatetimeIndex(idx)
    sess = pd.factorize(idx.normalize())[0]
    al = {t: data[t].reindex(idx) for t in tickers}
    O = np.column_stack([al[t]["Open"].to_numpy() for t in tickers])
    H = np.column_stack([al[t]["High"].to_numpy() for t in tickers])
    L = np.column_stack([al[t]["Low"].to_numpy() for t in tickers])
    C = np.column_stack([al[t]["Close"].to_numpy() for t in tickers])
    Z = np.column_stack([((data[t]["Close"] - data[t]["Close"].rolling(p.sma_len).mean())
                          / data[t]["Close"].rolling(p.sma_len).std()).reindex(idx).to_numpy()
                         for t in tickers])
    last = pd.DataFrame(C).ffill().bfill().to_numpy()   # mark-to-market prices
    cost = p.cost_bps / 1e4
    levels = np.asarray(p.add_levels)

    cash = capital
    pos = [_Pos() for _ in tickers]
    pending = [None] * len(tickers)                      # "entry" | "exit"
    runners = [[] for _ in tickers]   # dicts: shares, basis, start_sess, trade, due
    n = len(idx)
    equity, gross = np.empty(n), np.empty(n)
    trades = []

    def buy(ps, cash_amt, px):
        nonlocal cash
        ps.shares += cash_amt / px
        paid = cash_amt * (1 + cost)
        cash -= paid
        ps.basis += paid
        ps.spent += paid
        ps.fills += 1
        ps.max_invested = max(ps.max_invested, ps.basis)

    def sell(ps, frac, px):
        nonlocal cash
        q = ps.shares * frac
        got = q * px * (1 - cost)
        cash += got
        ps.received += got
        ps.basis *= (1 - frac)
        ps.shares -= q

    def close_cycle(j, i, reason):
        ps = pos[j]
        if ps.shares > 0:   # leftover runner after a partial take profit
            runners[j].append(dict(shares=ps.shares, basis=ps.basis, start_sess=ps.start_sess,
                                   trade=len(trades), due=False))
        trades.append(dict(
            ticker=tickers[j], entry_time=ps.entry_time, exit_time=idx[i],
            fills=ps.fills, tp_hit=ps.tp_hit, reason=reason,
            max_invested=ps.max_invested, runner_pnl=0.0,
            pnl=ps.received - (ps.spent - ps.basis),          # runner cost not yet realised
            ret_on_invested=(ps.received - (ps.spent - ps.basis)) / ps.max_invested))
        pos[j] = _Pos()

    def mtm_all(i):
        return sum((q.shares + sum(r["shares"] for r in runners[k])) * last[i, k]
                   for k, q in enumerate(pos))

    for i in range(n):
        for j in range(len(tickers)):
            o, h, l, c = O[i, j], H[i, j], L[i, j], C[i, j]
            if np.isnan(o):
                continue
            ps = pos[j]

            # runners due for their time stop are sold at this open
            keep = []
            for rn in runners[j]:
                if rn["due"]:
                    got = rn["shares"] * o * (1 - cost)
                    cash += got
                    tr = trades[rn["trade"]]
                    tr["pnl"] += got - rn["basis"]
                    tr["runner_pnl"] = got - rn["basis"]
                    tr["ret_on_invested"] = tr["pnl"] / tr["max_invested"]
                else:
                    keep.append(rn)
            runners[j] = keep

            # orders decided at the previous close, filled at this open
            if pending[j] == "entry":
                buy(ps, ps.base_cash, o)
                ps.first_fill, ps.start_sess, ps.entry_time = o, sess[i], idx[i]
            elif pending[j] == "exit":
                sell(ps, 1.0, o)
                close_cycle(j, i, "time_stop")
                ps = pos[j]
            pending[j] = None

            # resting limit orders during the bar
            if ps.shares > 0:
                added = False
                while ps.fills <= len(levels):
                    lvl = ps.first_fill * (1 - levels[ps.fills - 1])
                    if l > lvl:
                        break
                    buy(ps, ps.base_cash * p.size_mult ** ps.fills, min(o, lvl))
                    added = True
                if not added:
                    target = ps.basis / ps.shares * (1 + p.tp_pct)
                    if h >= target:
                        sell(ps, p.tp_sell_frac, max(o, target))
                        ps.tp_hit = True
                        close_cycle(j, i, "take_profit")
                        ps = pos[j]

            # decisions at this bar's close
            if i < n - 1:
                for rn in runners[j]:
                    rn["due"] = sess[i] - rn["start_sess"] >= p.time_stop_days
            if ps.shares > 0:
                if sess[i] - ps.start_sess >= p.time_stop_days and i < n - 1:
                    pending[j] = "exit"
            elif i < n - 1 and not np.isnan(Z[i, j]) and Z[i, j] < -p.entry_z:
                eq_now = cash + mtm_all(i)
                ps.base_cash = p.first_buy_pct * eq_now
                pending[j] = "entry"

        mtm = mtm_all(i)
        equity[i] = cash + mtm
        gross[i] = mtm / equity[i]

    eq = pd.Series(equity, index=idx)
    eq.attrs["exposure"] = pd.Series(gross, index=idx)
    return eq, pd.DataFrame(trades)


def stats(eq: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    daily = eq.groupby(eq.index.normalize()).last()
    r = daily.pct_change().dropna()
    yrs = len(daily) / 252
    s = dict(total_ret=eq.iloc[-1] / eq.iloc[0] - 1,
             cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
             vol=r.std() * np.sqrt(252),
             sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
             max_dd=(eq / eq.cummax() - 1).min())
    if "exposure" in eq.attrs:
        s["avg_exposure"] = eq.attrs["exposure"].mean()
        s["max_exposure"] = eq.attrs["exposure"].max()
    if trades is not None and len(trades):
        s |= dict(n_trades=len(trades),
                  win_rate=(trades.pnl > 0).mean(),
                  avg_win=trades.loc[trades.pnl > 0, "pnl"].mean(),
                  avg_loss=trades.loc[trades.pnl <= 0, "pnl"].mean(),
                  worst=trades.pnl.min(),
                  tp_rate=trades.tp_hit.mean(),
                  full_ladder=(trades.fills == trades.fills.max()).mean())
    return s


if __name__ == "__main__":
    from fetch_data import load
    eq, tr = run_portfolio(load("1h"), Params())
    print(pd.Series(stats(eq, tr)).round(5))
