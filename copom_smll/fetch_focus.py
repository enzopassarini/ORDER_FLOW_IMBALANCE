"""
Download Focus survey (BCB) annual GDP expectations and build a weekly
"next-year GDP" expectations panel.

Output: data/focus_gdp_daily.csv  -> Data, DataReferencia, Mediana, Media, numeroRespondentes
        data/focus_gdp_weekly.csv -> Friday-dated median expectation for each reference year
"""
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
URL = ("https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
       "?$filter=Indicador%20eq%20'PIB%20Total'%20and%20baseCalculo%20eq%200"
       "&$select=Data,DataReferencia,Mediana,Media,numeroRespondentes"
       "&$orderby=Data%20asc&$format=json&$top={top}&$skip={skip}")


def fetch(page=20000):
    rows, skip = [], 0
    while True:
        for i in range(4):
            try:
                r = requests.get(URL.format(top=page, skip=skip), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
                r.raise_for_status()
                break
            except Exception:
                if i == 3:
                    raise
                time.sleep(2 ** (i + 1))
        v = r.json()["value"]
        rows += v
        if len(v) < page:
            break
        skip += page
    df = pd.DataFrame(rows)
    df["Data"] = pd.to_datetime(df["Data"])
    df["DataReferencia"] = df["DataReferencia"].astype(int)
    return df


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    df = fetch()
    df.to_csv(DATA / "focus_gdp_daily.csv", index=False)
    # weekly: last available survey value on or before each Friday (the Focus report date)
    wide = df.pivot_table(index="Data", columns="DataReferencia", values="Mediana")
    fridays = pd.date_range(wide.index.min(), wide.index.max(), freq="W-FRI")  # no Friday past the last survey
    weekly = wide.reindex(wide.index.union(fridays)).ffill().loc[fridays]
    weekly.index.name = "date"
    weekly.to_csv(DATA / "focus_gdp_weekly.csv")
    print(f"{len(df)} rows, {df.Data.min().date()} -> {df.Data.max().date()}; weekly {len(weekly)} Fridays")
