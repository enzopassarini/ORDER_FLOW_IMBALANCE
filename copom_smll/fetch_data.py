"""
Download daily closes for Ibovespa (IBOV) and Small Caps (SMLL) from B3,
and the history of COPOM Selic-target (meta Selic) decisions from the BCB.

Outputs (in ./data):
  indices_daily.csv  -> date, IBOV, SMLL
  copom_decisions.csv -> meeting, date, extraordinary, selic_meta, delta_bp
"""
import base64
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}
B3_URL = "https://sistemaswebb3-listados.b3.com.br/indexStatisticsProxy/IndexCall/GetPortfolioDay/"
BCB_COPOM_URL = "https://www.bcb.gov.br/api/servico/sitebcb/historicotaxasjuros"


def _get_json(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def _br_number(s):
    return float(s.replace(".", "").replace(",", "."))


def fetch_b3_index(index, start_year, end_year):
    """B3 returns one year per call as a (day x month) table of closes."""
    rows = []
    for year in range(start_year, end_year + 1):
        payload = json.dumps({"index": index, "language": "pt-br", "year": str(year)})
        token = base64.b64encode(payload.encode()).decode()
        data = _get_json(B3_URL + token)
        for rec in data.get("results") or []:
            for m in range(1, 13):
                v = rec.get(f"rateValue{m}")
                if v:
                    rows.append((date(year, m, rec["day"]), _br_number(v)))
    s = pd.Series(dict(rows), name=index).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def fetch_copom():
    data = _get_json(BCB_COPOM_URL)["conteudo"]
    df = pd.DataFrame(data)
    df = pd.DataFrame({
        "meeting": df["NumeroReuniaoCopom"].astype(int),
        "date": pd.to_datetime(df["DataReuniaoCopom"]).dt.tz_convert("America/Sao_Paulo").dt.date,
        "extraordinary": df["ReuniaoExtraordinaria"].astype(bool),
        "selic_meta": df["MetaSelic"].astype(float),
    })
    df = df.sort_values(["date", "meeting"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["delta_bp"] = (df["selic_meta"].diff() * 100).round(0)
    return df


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    this_year = date.today().year
    ibov = fetch_b3_index("IBOV", 1998, this_year)
    smll = fetch_b3_index("SMLL", 2005, this_year)
    pd.concat([ibov, smll], axis=1).rename_axis("date").to_csv(DATA / "indices_daily.csv")
    copom = fetch_copom()
    copom.to_csv(DATA / "copom_decisions.csv", index=False)
    print(f"IBOV: {len(ibov)} days {ibov.index.min().date()} -> {ibov.index.max().date()}")
    print(f"SMLL: {len(smll)} days {smll.index.min().date()} -> {smll.index.max().date()}")
    print(f"COPOM: {len(copom)} decisions {copom.date.min().date()} -> {copom.date.max().date()}")
