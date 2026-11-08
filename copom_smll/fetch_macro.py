"""
Download data for the macro channels.

  credit_default  BCB SGS 21083  default rate, business (PJ) credit, % (monthly, from Mar 2011)
  hh_dsr          BCB SGS 29034  household debt-service ratio, % of income (monthly, from Mar 2005)
  epu             policyuncertainty.com Brazil news-based EPU index (monthly)
  dxy             US dollar index, Yahoo Finance DX-Y.NYB (daily)
  embi            EMBI+ Brazil spread, bp, Ipeadata JPM366_EMBI366 (daily; series discontinued Jul 2024)

Monthly series carry an availability date (when the value could have been known) so tests
use no lookahead: credit_default month end + 45 days, hh_dsr month end + 75 days (the BCB
releases them ~5 and ~10 weeks after the reference month), epu month end + 10 days.
Output: data/macro_<name>.csv with columns date (reference), avail (first usable date), value.
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
LAG = {"credit_default": 45, "hh_dsr": 75, "epu": 10}


def get(url, **kw):
    for i in range(4):
        try:
            r = requests.get(url, timeout=90, **kw)
            r.raise_for_status()
            return r
        except Exception:
            if i == 3:
                raise
            time.sleep(2 ** (i + 1))


def monthly(df, name):
    df = df.dropna().sort_values("date")
    df["avail"] = df["date"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=LAG[name])
    df[["date", "avail", "value"]].to_csv(DATA / f"macro_{name}.csv", index=False)
    print(f"{name:15s} {len(df):5d} {df.date.min().date()} -> {df.date.max().date()}")


def sgs(code, name):
    d = pd.DataFrame(get(f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json").json())
    d["date"] = pd.to_datetime(d["data"], dayfirst=True)
    d["value"] = pd.to_numeric(d["valor"])
    monthly(d, name)


if __name__ == "__main__":
    sgs(21083, "credit_default")
    sgs(29034, "hh_dsr")
    x = pd.read_excel(io.BytesIO(get("https://www.policyuncertainty.com/media/Brazil_Policy_Uncertainty_Data.xlsx",
                                     headers={"User-Agent": "Mozilla/5.0"}).content))
    x = x[pd.to_numeric(x["year"], errors="coerce").notna()]
    x["date"] = pd.to_datetime(dict(year=x["year"].astype(int), month=x["month"].astype(int), day=1))
    x["value"] = x.iloc[:, 2].astype(float)
    monthly(x, "epu")
    j = get("https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?period1=0&period2=9999999999&interval=1d",
            headers={"User-Agent": "Mozilla/5.0"}).json()["chart"]["result"][0]
    d = pd.DataFrame({"date": pd.to_datetime(j["timestamp"], unit="s").normalize(),
                      "value": j["indicators"]["quote"][0]["close"]}).dropna().drop_duplicates("date", keep="last")
    d["avail"] = d["date"]
    d[["date", "avail", "value"]].to_csv(DATA / "macro_dxy.csv", index=False)
    print(f"dxy             {len(d):5d} {d.date.min().date()} -> {d.date.max().date()}")
    e = pd.DataFrame(get("http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='JPM366_EMBI366')").json()["value"])
    e["date"] = pd.to_datetime(e["VALDATA"].str[:10])
    e["value"] = e["VALVALOR"].astype(float)
    e["avail"] = e["date"]
    e[["date", "avail", "value"]].dropna().to_csv(DATA / "macro_embi.csv", index=False)
    print(f"embi            {len(e):5d} {e.date.min().date()} -> {e.date.max().date()}")
