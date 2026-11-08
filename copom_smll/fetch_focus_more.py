"""
Download more Focus (BCB) series and build Friday-dated weekly panels.

Annual expectations (baseCalculo = 0), median and cross-sectional std by reference year:
  primary   'Resultado primário'              (% of GDP, + = surplus)
  netdebt   'Dívida líquida do setor público' (% of GDP)
  ipca      'IPCA'                            (% y/y)
  selic     'Selic'                           (% a.a., year-end)
  gdp       'PIB Total'                       (% y/y)
12-month-ahead IPCA expectation (smoothed, baseCalculo = 0): ExpectativasMercadoInflacao12Meses.

Outputs: data/focus_weekly_<key>_median.csv, data/focus_weekly_<key>_std.csv,
         data/focus_weekly_ipca12m.csv
Weekly value = last survey day on or before each Friday (no Friday past the last survey).
"""
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
BASE = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
ANNUAL = {"primary": "Resultado primário", "netdebt": "Dívida líquida do setor público",
          "ipca": "IPCA", "selic": "Selic", "gdp": "PIB Total"}


def fetch(endpoint, flt, select, page=20000):
    rows, skip = [], 0
    while True:
        url = (f"{BASE}{endpoint}?$filter={quote(flt)}&$select={select}&$orderby=Data%20asc"
               f"&$format=json&$top={page}&$skip={skip}")
        for i in range(4):
            try:
                r = requests.get(url, timeout=90)
                r.raise_for_status()
                break
            except Exception:
                if i == 3:
                    raise
                time.sleep(2 ** (i + 1))
        v = r.json()["value"]
        rows += v
        if len(v) < page:
            return pd.DataFrame(rows)
        skip += page


def weekly(wide):
    fridays = pd.date_range(wide.index.min(), wide.index.max(), freq="W-FRI")
    out = wide.reindex(wide.index.union(fridays)).ffill().loc[fridays]
    out.index.name = "date"
    return out


if __name__ == "__main__":
    for key, name in ANNUAL.items():
        df = fetch("ExpectativasMercadoAnuais", f"Indicador eq '{name}' and baseCalculo eq 0",
                   "Data,DataReferencia,Mediana,DesvioPadrao")
        df["Data"] = pd.to_datetime(df["Data"])
        df["DataReferencia"] = df["DataReferencia"].astype(int)
        for col, suffix in (("Mediana", "median"), ("DesvioPadrao", "std")):
            w = weekly(df.pivot_table(index="Data", columns="DataReferencia", values=col))
            w.to_csv(DATA / f"focus_weekly_{key}_{suffix}.csv")
        print(f"{key:8s} {len(df):6d} rows {df.Data.min().date()} -> {df.Data.max().date()}")
    df = fetch("ExpectativasMercadoInflacao12Meses",
               "Indicador eq 'IPCA' and Suavizada eq 'S' and baseCalculo eq 0", "Data,Mediana,DesvioPadrao")
    df["Data"] = pd.to_datetime(df["Data"])
    w = weekly(df.set_index("Data")[["Mediana", "DesvioPadrao"]].groupby(level=0).last())
    w.to_csv(DATA / "focus_weekly_ipca12m.csv")
    print(f"ipca12m  {len(df):6d} rows {df.Data.min().date()} -> {df.Data.max().date()}")
