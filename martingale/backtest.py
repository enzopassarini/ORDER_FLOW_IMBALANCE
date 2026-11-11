"""
Martingale / grid ("vault") backtest on 1h bars, long and/or short, one shared-equity portfolio.

Each (ticker, side) pair is a "book" that runs one cycle at a time. Long and short books on the
same ticker are independent (a hedged grid can hold both).

Per cycle, for side s = +1 (long) / -1 (short):
  Entry      : mode "signal": long when close z-score vs. rolling SMA < -ENTRY_Z,
                              short when z > +ENTRY_Z.
               mode "always": open a new cycle whenever the book is flat (classic grid).
               First fill at the next bar's open, sized FIRST_BUY_PCT of current total equity.
  Adds       : resting limit orders at fixed offsets against the FIRST fill
               (long: below it, short: above it): 2.25%, 3.54%, 5.04%, 7.00%, 9.85%.
               Lot k = first lot * SIZE_MULT**k.
  Take profit: resting limit at average entry * (1 + s*TP_PCT); closes TP_SELL_FRAC
               ("almost everything"). The leftover runner is held aside (does not block a new
               cycle) and closed at its own time stop; its P&L is credited to its cycle.
  Time stop  : TIME_STOP_DAYS sessions after the first fill -> close everything at next open.
  Fills      : limits fill when Low/High touch the level (at the open on a gap). If an add fills
               on a bar, take profit is not checked until the next bar (conservative ordering).
  Costs      : COST_BPS per side on notional; average entry includes costs.
               BORROW_FEE (annual) charged daily on short notional.
  Leverage   : not capped; gross and net exposure are reported (no financing cost on longs).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Params:
    sides: tuple = (1,)            # (1,), (-1,), or (1, -1)
    mode: str = "signal"           # "signal" | "always"
    sma_len: int = 35
    entry_z: float = 2.0
    add_levels: tuple = (0.0225, 0.0354, 0.0504, 0.07, 0.0985)
    size_mult: float = 2.0
    first_buy_pct: float = 0.00364
    tp_pct: float = 0.0056
    tp_sell_frac: float = 0.95
    time_stop_days: int = 5
    cost_bps: float = 5.0
    borrow_fee: float = 0.01       # 1%/yr on short notional (B3 large caps typically ~0.3-2%)
    borrow_on_all: bool = False    # spread books: every position has a short leg


@dataclass
class _Pos:
    shares: float = 0.0     # signed
    basis: float = 0.0      # signed net cash put in for the held shares (incl. costs)
    spent: float = 0.0      # signed cash out over the cycle
    received: float = 0.0   # signed cash in over the cycle
    first_fill: float = np.nan
    base_cash: float = 0.0
    fills: int = 0
    start_sess: int = -1
    entry_time: object = None
    tp_hit: bool = False
    max_invested: float = 0.0


def run_portfolio(data: dict[str, pd.DataFrame], p: Params, capital: float = 1.0):
    tickers = list(data)
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    sess = pd.factorize(idx.normalize())[0]
    al = {t: data[t].reindex(idx) for t in tickers}
    O, H, L, C = (np.column_stack([al[t][f].to_numpy() for t in tickers])
                  for f in ("Open", "High", "Low", "Close"))
    Z = np.column_stack([((data[t]["Close"] - data[t]["Close"].rolling(p.sma_len).mean())
                          / data[t]["Close"].rolling(p.sma_len).std()).reindex(idx).to_numpy()
                         for t in tickers])
    last = pd.DataFrame(C).ffill().bfill().to_numpy()   # mark-to-market prices
    cost = p.cost_bps / 1e4
    levels = np.asarray(p.add_levels)
    books = [(j, s) for j in range(len(tickers)) for s in p.sides]

    cash = capital
    pos = {b: _Pos() for b in books}
    pending = {b: None for b in books}                   # "entry" | "exit"
    runners = {b: [] for b in books}
    n = len(idx)
    equity, gross, net = np.empty(n), np.empty(n), np.empty(n)
    trades = []

    def trade(ps, s, cash_amt, px):
        """Open/add: s*cash_amt of notional at px."""
        nonlocal cash
        ps.shares += s * cash_amt / px
        out = s * cash_amt + cash_amt * cost
        cash -= out
        ps.basis += out
        ps.spent += out
        ps.fills += 1
        ps.max_invested = max(ps.max_invested, abs(ps.basis))

    def close(ps, frac, px):
        nonlocal cash
        q = ps.shares * frac
        got = q * px - abs(q) * px * cost
        cash += got
        ps.received += got
        ps.basis *= (1 - frac)
        ps.shares -= q

    def close_cycle(b, i, reason):
        ps = pos[b]
        if ps.shares != 0:
            runners[b].append(dict(shares=ps.shares, basis=ps.basis, start_sess=ps.start_sess,
                                   trade=len(trades), due=False))
        pnl = ps.received - (ps.spent - ps.basis)
        trades.append(dict(ticker=tickers[b[0]], side="long" if b[1] > 0 else "short",
                           entry_time=ps.entry_time, exit_time=idx[i], fills=ps.fills,
                           tp_hit=ps.tp_hit, reason=reason, max_invested=ps.max_invested,
                           runner_pnl=0.0, pnl=pnl, ret_on_invested=pnl / ps.max_invested))
        pos[b] = _Pos()

    def mtm(i):
        return sum((pos[b].shares + sum(r["shares"] for r in runners[b])) * last[i, b[0]]
                   for b in books)

    def gross_mtm(i):
        return sum(abs(pos[b].shares + sum(r["shares"] for r in runners[b])) * last[i, b[0]]
                   for b in books)

    for i in range(n):
        new_day = i > 0 and sess[i] != sess[i - 1]
        if new_day and p.borrow_fee:
            held = [(pos[b].shares + sum(r["shares"] for r in runners[b])) * last[i - 1, b[0]]
                    for b in books]
            short_notional = sum(abs(v) if p.borrow_on_all else max(-v, 0) for v in held)
            cash -= short_notional * p.borrow_fee / 252

        for b in books:
            j, s = b
            o, h, l, c = O[i, j], H[i, j], L[i, j], C[i, j]
            if np.isnan(o):
                continue
            ps = pos[b]

            # runners due for their time stop: close at this open
            keep = []
            for rn in runners[b]:
                if rn["due"]:
                    got = rn["shares"] * o - abs(rn["shares"]) * o * cost
                    cash += got
                    tr = trades[rn["trade"]]
                    tr["runner_pnl"] = got - rn["basis"]
                    tr["pnl"] += got - rn["basis"]
                    tr["ret_on_invested"] = tr["pnl"] / tr["max_invested"]
                else:
                    keep.append(rn)
            runners[b] = keep

            # orders decided at the previous close, filled at this open
            if pending[b] == "entry":
                trade(ps, s, ps.base_cash, o)
                ps.first_fill, ps.start_sess, ps.entry_time = o, sess[i], idx[i]
            elif pending[b] == "exit":
                close(ps, 1.0, o)
                close_cycle(b, i, "time_stop")
                ps = pos[b]
            pending[b] = None

            # resting limit orders during the bar
            if ps.shares != 0:
                added = False
                while ps.fills <= len(levels):
                    lvl = ps.first_fill * (1 - s * levels[ps.fills - 1])
                    if (s > 0 and l > lvl) or (s < 0 and h < lvl):
                        break
                    fill = min(o, lvl) if s > 0 else max(o, lvl)
                    trade(ps, s, ps.base_cash * p.size_mult ** ps.fills, fill)
                    added = True
                if not added:
                    target = ps.basis / ps.shares * (1 + s * p.tp_pct)
                    if (s > 0 and h >= target) or (s < 0 and l <= target):
                        close(ps, p.tp_sell_frac, max(o, target) if s > 0 else min(o, target))
                        ps.tp_hit = True
                        close_cycle(b, i, "take_profit")
                        ps = pos[b]

            # decisions at this bar's close
            if i < n - 1:
                for rn in runners[b]:
                    rn["due"] = sess[i] - rn["start_sess"] >= p.time_stop_days
                if ps.shares != 0:
                    if sess[i] - ps.start_sess >= p.time_stop_days:
                        pending[b] = "exit"
                else:
                    z = Z[i, j]
                    go = (p.mode == "always" and not np.isnan(z)) or \
                         (p.mode == "signal" and not np.isnan(z) and s * z < -p.entry_z)
                    if go:
                        ps.base_cash = p.first_buy_pct * (cash + mtm(i))
                        pending[b] = "entry"

        m = mtm(i)
        equity[i] = cash + m
        gross[i] = gross_mtm(i) / equity[i]
        net[i] = m / equity[i]

    eq = pd.Series(equity, index=idx)
    eq.attrs["exposure"] = pd.Series(gross, index=idx)
    eq.attrs["net_exposure"] = pd.Series(net, index=idx)
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
        s["avg_net_exp"] = eq.attrs["net_exposure"].mean()
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
