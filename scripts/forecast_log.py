"""Pre-registered forward forecast log (see kbforecast/forecastlog.py).

    python scripts/forecast_log.py log   path/to/KB_주간시계열.xlsx [--hub] [--log-dir forecast_log] [--all-regions]
    python scripts/forecast_log.py score path/to/KB_주간시계열.xlsx [--hub] [--log-dir forecast_log]

`log` fits the production engine on the data up to the latest week, and stores the live-origin forecasts of every anchor horizon
with commit / data fingerprint / model configuration. Run it each time new weeks arrive (the file name carries origin date and data id,
so a repeat on the same data does nothing and revised data never overwrites an earlier record). `score` compares matured entries with
prices that were actually observed. Commit the log directory; do not edit entries after the fact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kbforecast.engine import fit_engine  # noqa: E402
from kbforecast.forecastlog import load_log, score_log, summarize, write_log  # noqa: E402
from kbforecast.hub import extend_kb_panel, fetch_hub  # noqa: E402
from kbforecast.kb_panel import parse_kb_panel, seoul_region_keys  # noqa: E402
from kbforecast.overlay import load_base_rate, load_cd91  # noqa: E402


def load_panel(path: Path, use_hub: bool):
    kb = parse_kb_panel(path.read_bytes())
    if use_hub:
        kb, report = extend_kb_panel(kb, fetch_hub(kb.hierarchy))
        print(report.message)
    return kb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["log", "score"])
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--hub", action="store_true", help="extend the workbook with newer weeks from the KB data hub first")
    parser.add_argument("--log-dir", type=Path, default=Path("forecast_log"))
    parser.add_argument("--all-regions", action="store_true", help="log all 197 regions (default: Seoul series, national and the aggregates)")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    kb = load_panel(args.workbook, args.hub)
    if args.command == "log":
        fit = fit_engine(kb, rate=load_base_rate(None), cd=load_cd91(None))
        regions = None if args.all_regions else seoul_region_keys(kb.hierarchy) | {"전국", "수도권", "6개광역시", "기타지방"}
        path, message = write_log(fit, kb, args.log_dir, regions, args.note)
        print(message)
        return
    log = load_log(args.log_dir)
    scored = score_log(log, kb)
    pd.set_option("display.width", 200)
    matured = int(scored["y"].notna().sum())
    print(f"기록 {len(log)}행 중 실현된 {matured}행 (데이터 마지막 주 {kb.sale.index.max().date()})")
    print(summarize(scored, kb).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
