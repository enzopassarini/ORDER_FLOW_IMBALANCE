"""
Fed decisions: FOMC meeting dates (federalreserve.gov) + target rate (FRED).

Target rate: FRED DFEDTAR (single target, to 2008-12-15) spliced with DFEDTARU (upper
bound of the range, from 2008-12-16).
Decisions: every FOMC meeting (scheduled or unscheduled "Meeting" headings; conference
calls, notation votes and cancelled meetings excluded), plus any FRED target change not
within 3 days of a meeting (intermeeting moves such as 22 Jan 2008).
delta_bp = target 3 days after the decision - target the day before it.

Output: data/fed_decisions.csv (meeting, date, extraordinary, rate, delta_bp) in the same
layout as data/copom_decisions.csv.
"""
import io
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}  # federalreserve.gov; FRED rejects this UA, so FRED uses the default
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}


def get(url, headers=UA):
    for i in range(4):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            return r.text
        except Exception:
            if i == 3:
                raise
            time.sleep(2 ** (i + 1))


def last_day(year, month_txt, days_txt):
    """'April/May', '30-1' -> date(year, 5, 1); 'January', '30-31*' -> date(year, 1, 31)."""
    name = month_txt.split("/")[-1].strip()
    month = next(i for full, i in MONTHS.items() if full.startswith(name[:3]))  # 'Feb', 'Sept', ...
    day = int(re.findall(r"\d+", days_txt)[-1])
    return date(year, month, day)


def meeting_dates(first_year=2005):
    out = []
    this_year = date.today().year
    # historical pages: headings like "January 30-31 Meeting - 2007", "March 15 (unscheduled) Meeting - 2020"
    for y in range(first_year, this_year + 1):
        try:
            html = get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
        except Exception:
            continue
        for h in re.findall(r"<h5[^>]*>([^<]*)</h5>", html):
            if "Meeting" not in h or "cancelled" in h:
                continue
            m = re.match(r"\s*([A-Za-z/]+)\s+([\d\-]+)\s*(\(unscheduled\))?", h)
            if m:
                out.append((last_day(y, m.group(1), m.group(2)), bool(m.group(3))))
    # current calendar page: "<year> FOMC Meetings" sections with month / date cells
    html = get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    parts = re.split(r"(\d{4}) FOMC Meetings", html)
    for y, body in zip(parts[1::2], parts[2::2]):
        y = int(y)
        months = re.findall(r'fomc-meeting__month[^>]*><strong>([^<]+)</strong>', body)
        days = re.findall(r'fomc-meeting__date[^>]*>([^<]+)<', body)
        for mo, dy in zip(months, days):
            if re.search(r"\d", dy):
                out.append((last_day(y, mo, dy), "unscheduled" in dy.lower()))
    df = pd.DataFrame(out, columns=["date", "extraordinary"]).drop_duplicates("date")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def target_rate():
    s = []
    for sid in ("DFEDTAR", "DFEDTARU"):
        t = pd.read_csv(io.StringIO(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", headers={})))
        t.columns = ["date", "rate"]
        t["date"] = pd.to_datetime(t["date"])
        t["rate"] = pd.to_numeric(t["rate"], errors="coerce")
        s.append(t.dropna().set_index("date")["rate"])
    rate = pd.concat([s[0][s[0].index < "2008-12-16"], s[1][s[1].index >= "2008-12-16"]]).sort_index()
    return rate.asfreq("D").ffill()


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    rate = target_rate()
    meet = meeting_dates()
    meet = meet[(meet["date"] >= "2005-01-01") & (meet["date"] <= rate.index[-1] - pd.Timedelta(days=3))]
    # intermeeting target changes
    chg = rate[rate.diff().fillna(0) != 0].index
    chg = chg[chg >= "2005-01-01"]
    extra = [d for d in chg if (abs(meet["date"] - d) > pd.Timedelta(days=3)).all()]
    meet = pd.concat([meet, pd.DataFrame({"date": extra, "extraordinary": True})]).sort_values("date")
    meet = meet.reset_index(drop=True)
    before = rate.reindex(meet["date"] - pd.Timedelta(days=1)).to_numpy()
    after = rate.reindex(meet["date"] + pd.Timedelta(days=3)).to_numpy()
    meet["rate"] = after
    meet["delta_bp"] = ((after - before) * 100).round(0)
    meet.insert(0, "meeting", range(1, len(meet) + 1))
    meet.to_csv(DATA / "fed_decisions.csv", index=False)
    # check: every target change is assigned to some decision
    assigned = meet["delta_bp"].abs().sum()
    total = (rate[rate.index >= "2004-12-31"].diff().abs().sum() * 100).round(0)
    print(f"{len(meet)} decisions {meet.date.min().date()} -> {meet.date.max().date()} "
          f"({meet.extraordinary.sum()} unscheduled/intermeeting); |delta| assigned {assigned:.0f} bp of {total:.0f} bp")
    print(meet["delta_bp"].value_counts().sort_index().to_dict())
    print(meet[meet.delta_bp != 0].tail(8).to_string(index=False))
