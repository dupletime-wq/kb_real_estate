"""Monthly resident-registration population and households for every si/gun/gu (Ministry of the Interior and Safety, jumin.mois.go.kr; no key needed).

    python scripts/fetch_population.py [--out kbforecast/data/pop_sigungu_monthly.csv.gz] [--start 2008-01]

The statistic is the month-end count (publicly available from the following month; the files here are what the site serves TODAY, i.e. final
data: revisions after first release are not observable). Region names change over time, so rows are stored with the 10-digit administrative code
and the name as served. Mapping to KB region keys is in kbforecast/regional.py.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import re
import time

import pandas as pd
import requests

PAGE = "https://jumin.mois.go.kr/statMonth.do"
CSV = "https://jumin.mois.go.kr/downloadCsv.do?searchYearMonth=month&xlsStats=2"


def decode(raw: bytes) -> str:
    for enc in ("utf-8", "cp949"):
        try:
            text = raw.decode(enc)
            if "행정구역" in text[:200] or enc == "cp949":
                return text
        except UnicodeDecodeError:
            continue
    return raw.decode("cp949", errors="replace")


def fetch_range(session: requests.Session, y0: int, m0: int, y1: int, m1: int, retries: int = 6) -> pd.DataFrame:
    data = dict(sltOrgType="1", sltOrgLvl1="A", sltOrgLvl2="", gender="gender", genderPer="genderPer", generation="generation", sltUndefType="", searchYearStart=str(y0),
                searchMonthStart=f"{m0:02d}", searchYearEnd=str(y1), searchMonthEnd=f"{m1:02d}", sltOrderType="1", sltOrderValue="ASC", category="month")
    last = None
    for attempt in range(retries):
        try:
            session.get(PAGE, timeout=40)
            r = session.post(CSV, data=data, headers={"Referer": PAGE}, timeout=120)
            r.raise_for_status()
            text = decode(r.content)
            frame = pd.read_csv(io.StringIO(text), dtype=str)
            return frame
        except Exception as exc:  # network hiccups through the egress proxy are common
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{y0}-{m0:02d}..{y1}-{m1:02d}: {last}")


def tidy(frame: pd.DataFrame) -> pd.DataFrame:
    label = frame.columns[0]
    names = frame[label].str.strip()
    code = names.str.extract(r"\((\d{10})\)\s*$")[0]
    name = names.str.replace(r"\s*\(\d{10}\)\s*$", "", regex=True).str.strip()
    rows = []
    for col in frame.columns[1:]:
        m = re.match(r"(\d{4})년?(\d{2})월?_(.*)", col.strip())
        if not m:
            continue
        ym, field = f"{m.group(1)}-{m.group(2)}", m.group(3).strip()
        key = {"총인구수": "pop", "세대수": "households"}.get(field)
        if key is None:
            continue
        values = pd.to_numeric(frame[col].str.replace(",", "").str.strip(), errors="coerce")
        rows.append(pd.DataFrame({"ym": ym, "code": code, "name": name, "field": key, "value": values}))
    long = pd.concat(rows, ignore_index=True)
    wide = long.pivot_table(index=["ym", "code", "name"], columns="field", values="value", aggfunc="first").reset_index()
    return wide.dropna(subset=["code"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("kbforecast/data/pop_sigungu_monthly.csv.gz"))
    parser.add_argument("--start", default="2008-01")
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    now = pd.Timestamp.today()
    parts = []
    for year in range(int(args.start[:4]), now.year + 1):
        m0 = int(args.start[5:]) if year == int(args.start[:4]) else 1
        m1 = 12 if year < now.year else now.month
        frame = fetch_range(session, year, m0, year, m1)
        part = tidy(frame)
        print(year, len(part), part["ym"].min(), part["ym"].max(), flush=True)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).sort_values(["code", "ym"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, compression="gzip")
    print(f"saved {len(out)} rows, {out['code'].nunique()} codes, {out['ym'].min()}..{out['ym'].max()} -> {args.out}")


if __name__ == "__main__":
    main()
