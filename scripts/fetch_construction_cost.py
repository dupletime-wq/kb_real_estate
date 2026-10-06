"""Construction-cost proxy series from the Bank of Korea ECOS API (needs an API key: --key-file or the ECOS_KEY environment variable; the key is never printed or stored).

    ECOS_KEY=... python scripts/fetch_construction_cost.py [--out kbforecast/data/construction_cost_ecos.csv.gz]

Series (national, not Seoul-specific; ECOS has no regional construction-cost series):
  materials  producer price index (2020=100, monthly, table 404Y016) of 8 construction inputs: ordinary rebar, section steel, ready-mixed concrete, Portland cement, sand,
             plywood, plate glass, water-based paint. These items are fixed in advance; the composite (kbforecast/construction.py) is their equal-weight geometric mean.
  wage       hourly nominal wage index of the construction industry (2020=100, quarterly, table 901Y102, item A10008, from 2011Q1).
The Korea Institute of Civil Engineering and Building Technology (KICT) construction cost index itself has no public API; it is built from the same kind of inputs
(producer price items plus construction wages).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import pandas as pd
import requests

URL = "https://ecos.bok.or.kr/api/StatisticSearch"
MATERIALS = {
    "rebar": "30712101AA", "section_steel": "30712201AA", "ready_mix_concrete": "30623101AA", "cement": "30622101AA", "sand": "20122101AA", "plywood": "30311201AA",
    "plate_glass": "30611101AA", "paint": "30561102AA",
}


def call(key: str, stat: str, cycle: str, start: str, end: str, item: str, retries: int = 6) -> pd.DataFrame:
    url = "/".join([URL, key, "json", "kr", "1", "100000", stat, cycle, start, end, item])
    last = None
    for attempt in range(retries):
        try:
            payload = requests.get(url, timeout=60).json()
            block = payload.get("StatisticSearch")
            if block is None:
                raise RuntimeError(payload.get("RESULT", {}).get("MESSAGE", "unexpected ECOS response"))
            return pd.DataFrame(block["row"])[["TIME", "DATA_VALUE"]]
        except RuntimeError:
            raise
        except Exception as exc:  # network hiccups
            last = type(exc).__name__
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{stat}/{item}: {last}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("kbforecast/data/construction_cost_ecos.csv.gz"))
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get("ECOS_KEY", "")
    if not key:
        raise SystemExit("ECOS key needed (--key-file or ECOS_KEY)")
    this_month = pd.Timestamp.today()
    end_m = this_month.strftime("%Y%m")
    end_q = f"{this_month.year}Q{(this_month.month - 1) // 3 + 1}"
    rows = []
    for name, item in MATERIALS.items():
        frame = call(key, "404Y016", "M", "200101", end_m, item)
        frame["series"] = f"ppi_{name}"
        rows.append(frame)
    wage = call(key, "901Y102", "Q", "2011Q1", end_q, "A10008")
    wage["series"] = "wage_construction"
    rows.append(wage)
    out = pd.concat(rows, ignore_index=True).rename(columns={"TIME": "period", "DATA_VALUE": "value"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, compression="gzip")
    print(out.groupby("series")["period"].agg(["min", "max", "count"]))
    print(f"saved {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
