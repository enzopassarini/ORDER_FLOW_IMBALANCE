"""
Local nominal yield curve from Tesouro Direto (Tesouro Transparente, daily since 2005).

Uses the fixed-rate bonds, Tesouro Prefixado (LTN, zero coupon) and Tesouro Prefixado com
Juros Semestrais (NTN-F), "Taxa Compra Manha" (morning buy yield, % a.a.). Each day the
yields are linearly interpolated in time to maturity (years) to constant maturities of
1, 2, 3 and 5 years; no extrapolation (missing if the curve does not reach that point;
5y is available from 2006).
Short rate: the Selic target in force on that day (COPOM decisions, effective the next day).

Outputs (same layout as the other macro_*.csv files: date, avail, value):
  data/curve_daily.csv           date, y1, y2, y3, y5, selic
  data/macro_curve_5y1y.csv      y5 - y1      (pp)
  data/macro_curve_2ysel.csv     y2 - Selic   (pp)
  data/macro_curve_y2.csv        y2 level     (%)
"""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
URL = ("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/"
       "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv")
TENORS = (1, 2, 3, 5)


def download():
    for i in range(4):
        try:
            r = requests.get(URL, timeout=300)
            r.raise_for_status()
            return r.content
        except Exception:
            if i == 3:
                raise
            time.sleep(2 ** (i + 1))


if __name__ == "__main__":
    d = pd.read_csv(io.BytesIO(download()), sep=";", decimal=",")
    d = d[d["Tipo Titulo"].isin(["Tesouro Prefixado", "Tesouro Prefixado com Juros Semestrais"])].copy()
    d["date"] = pd.to_datetime(d["Data Base"], dayfirst=True)
    d["mat"] = pd.to_datetime(d["Data Vencimento"], dayfirst=True)
    d["T"] = (d["mat"] - d["date"]).dt.days / 365.25
    d["y"] = pd.to_numeric(d["Taxa Compra Manha"], errors="coerce")
    d = d[(d["T"] > 0.05) & (d["y"] > 0)].dropna(subset=["y"])
    rows = []
    for dt, g in d.groupby("date"):
        g = g.groupby("T")["y"].mean().sort_index()
        row = {"date": dt}
        for t in TENORS:
            row[f"y{t}"] = float(np.interp(t, g.index, g.values)) if g.index.min() <= t <= g.index.max() else np.nan
        rows.append(row)
    cv = pd.DataFrame(rows).set_index("date").sort_index()
    cop = pd.read_csv(DATA / "copom_decisions.csv", parse_dates=["date"]).sort_values("date")
    eff = pd.Series(cop["selic_meta"].to_numpy(), index=cop["date"] + pd.Timedelta(days=1))
    cv["selic"] = eff.reindex(eff.index.union(cv.index)).ffill().reindex(cv.index)
    cv.to_csv(DATA / "curve_daily.csv", float_format="%.4f")
    for name, s in (("curve_5y1y", cv["y5"] - cv["y1"]), ("curve_2ysel", cv["y2"] - cv["selic"]),
                    ("curve_y2", cv["y2"])):
        out = s.dropna().rename("value").reset_index()
        out["avail"] = out["date"]
        out[["date", "avail", "value"]].to_csv(DATA / f"macro_{name}.csv", index=False, float_format="%.4f")
    print(f"{len(cv)} days {cv.index.min().date()} -> {cv.index.max().date()}")
    print(cv.resample("YE").mean().round(2).to_string())
