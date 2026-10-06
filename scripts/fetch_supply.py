"""Monthly housing supply by province (si/do) from the Ministry of Land, Infrastructure and Transport statistics portal (stat.molit.go.kr, no key needed).

    python scripts/fetch_supply.py [--out kbforecast/data/supply_sido_monthly.csv.gz]

Series (all housing types, all sectors; units = dwellings):
  permits      form 1946  "주택건설 인허가실적(월별 누계)" 2007-01 onwards, year-to-date cumulative (differenced in kbforecast/supply.py)
  starts       form 5386  "주택건설 착공실적(월계)"        2011-01 onwards, monthly
  completions  form 5372  "주택건설 준공실적(월계)"        2010-07 onwards, monthly ("사용검사실적")
The portal serves final / revised figures as of today (later months of the latest year are preliminary until the September confirmation), so the history is not a
point-in-time record of what was known at each date; kbforecast/supply.py applies an assumed publication lag. Province level only (no si/gun/gu).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd
import requests

BASE = "https://stat.molit.go.kr"
FORMS = {"permits": (1946, "200701"), "starts": (5386, "201101"), "completions": (5372, "201007")}
MAX_MONTHS = 48  # the portal allows at most 60 months per request


def get(session: requests.Session, form: int, start: str, end: str, retries: int = 8) -> list[dict]:
    last = None
    for attempt in range(retries):
        try:
            r = session.get(f"{BASE}/portal/stat/data.do", params={"formId": form, "styleNum": 1, "apprYn": "Y", "startDate": start, "endDate": end}, timeout=90)
            payload = r.json()
            if payload.get("result"):
                return payload["data"]
            last = payload.get("msg")
            break
        except Exception as exc:  # network hiccups through the egress proxy are common
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"form {form} {start}..{end}: {last}")


def fetch(session: requests.Session, form: int, first: str, last: str) -> pd.DataFrame:
    periods = pd.period_range(first[:4] + "-" + first[4:], last[:4] + "-" + last[4:], freq="M")
    rows = []
    for i in range(0, len(periods), MAX_MONTHS):
        chunk = periods[i:i + MAX_MONTHS]
        rows += get(session, form, chunk[0].strftime("%Y%m"), chunk[-1].strftime("%Y%m"))
    df = pd.DataFrame(rows)
    df = df.rename(columns={"0": "ym", "1": "group", "2": "sub", "3": "sido", "4": "value"})
    total = df["group"].str.replace(r"\s+", "", regex=True).eq("총계") & df["sub"].str.replace(r"\s+", "", regex=True).eq("총계")
    out = df[total].copy()
    out["value"] = pd.to_numeric(out["value"].astype(str).str.replace(",", ""), errors="coerce")
    out["ym"] = out["ym"].astype(str).str[:7]  # newest months carry a preliminary marker such as "2026-08 p)"
    return out[["ym", "sido", "value"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("kbforecast/data/supply_sido_monthly.csv.gz"))
    parser.add_argument("--last", default=None, help="last month YYYYMM (default: this month)")
    args = parser.parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"})
    session.get(f"{BASE}/portal/cate/statView.do?hRsId=468", timeout=60)
    last = args.last or pd.Timestamp.today().strftime("%Y%m")
    parts = []
    for kind, (form, first) in FORMS.items():
        # the newest months may not exist yet: step back until the portal accepts the end month
        end = last
        while True:
            try:
                frame = fetch(session, form, first, end)
                break
            except RuntimeError as exc:
                end = (pd.Period(end[:4] + "-" + end[4:], freq="M") - 1).strftime("%Y%m")
                if end < first:
                    raise
        frame["kind"] = kind
        print(kind, len(frame), frame["ym"].min(), frame["ym"].max(), flush=True)
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    out = out.pivot_table(index=["ym", "sido"], columns="kind", values="value", aggfunc="first").reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, compression="gzip")
    print(f"saved {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
