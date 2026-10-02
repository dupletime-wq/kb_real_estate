"""Pre-registered forward forecast log (kbforecast/forecastlog.py documents the file scheme).

    python scripts/forecast_log.py log     path/to/KB_주간시계열.xlsx [--hub] [--variants current,B_all] [--history trade_history/<date>/seoul_daily_counts.csv.gz]
    python scripts/forecast_log.py score   path/to/KB_주간시계열.xlsx [--hub] [--realized latest|first_seen|first_published]
    python scripts/forecast_log.py compare path/to/KB_주간시계열.xlsx [--hub] --base current --cand B_all [--realized latest|first_seen|first_published]
    python scripts/forecast_log.py variants

`log` fits the engine on the data up to the latest week and stores the live-origin forecasts of every anchor horizon with commit / data
fingerprint / library versions / full model configuration. `--variants` stores the current model and candidate models for the SAME origin and data
(one engine fit per variant); a repeat on the same origin, data and configuration does nothing, a different configuration or data vintage is a new file.
Every run (log and score) also appends the weekly index values it sees for the first time to `forecast_log/realized_first_seen.csv` (with the collection lag and a
late-collection flag), so forecasts can be scored with the value this system FIRST SAW (`--realized first_seen`), with today's value (`latest`), and, only for weeks whose
vintage was verified separately (`vintage_verified.csv`), with the first PUBLISHED value (`first_published`). `compare` joins two configurations on
identical origin / region / horizon / data vintage and reports point accuracy and interval coverage in separate columns.
Commit the log directory; never edit entries.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kbforecast.engine import fit_engine  # noqa: E402
from kbforecast.forecastlog import compare_logged, load_log, pair_models, record_first_seen, score_log, summarize, write_log  # noqa: E402
from kbforecast.hub import extend_kb_panel, fetch_hub  # noqa: E402
from kbforecast.kb_panel import parse_kb_panel, seoul_region_keys  # noqa: E402
from kbforecast.overlay import load_base_rate, load_cd91  # noqa: E402
from kbforecast.variants import NAMED_VARIANTS  # noqa: E402


def load_panel(path: Path, use_hub: bool):
    kb = parse_kb_panel(path.read_bytes())
    if use_hub:
        kb, report = extend_kb_panel(kb, fetch_hub(kb.hierarchy))
        print(report.message)
    return kb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["log", "score", "compare", "variants"])
    parser.add_argument("workbook", type=Path, nargs="?")
    parser.add_argument("--hub", action="store_true", help="extend the workbook with newer weeks from the KB data hub first")
    parser.add_argument("--log-dir", type=Path, default=Path("forecast_log"))
    parser.add_argument("--all-regions", action="store_true", help="log all 197 regions (default: Seoul series, national and the aggregates)")
    parser.add_argument("--variants", default="current")
    parser.add_argument("--history", type=Path, help="trade history CSV (needed by the V_* variants)")
    parser.add_argument("--realized", default="latest", choices=["latest", "first_seen", "first_published"])
    parser.add_argument("--base", default="current")
    parser.add_argument("--cand")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    if args.command == "variants":
        for name, v in NAMED_VARIANTS.items():
            print(f"{name:16s} {v.as_dict()}")
        return
    if args.workbook is None:
        parser.error("workbook is required")
    kb = load_panel(args.workbook, args.hub)
    archive = args.log_dir / "realized_first_seen.csv"
    verified = args.log_dir / "vintage_verified.csv"
    existing = load_log(args.log_dir)
    since = existing["origin"].min() if len(existing) else kb.last_date
    n_new = record_first_seen(kb, archive, since=since)
    print(f"first-seen archive: {n_new} new (region, week) values (first seen by this system; not necessarily first published)")
    if args.command == "log":
        history = pd.read_csv(args.history, dtype={"sgg_cd": str}) if args.history else None
        regions = None if args.all_regions else seoul_region_keys(kb.hierarchy) | {"전국", "수도권", "6개광역시", "기타지방"}
        for name in (v.strip() for v in args.variants.split(",") if v.strip()):
            if name not in NAMED_VARIANTS:
                sys.exit(f"unknown variant {name!r}; run `variants` to list them")
            variant = NAMED_VARIANTS[name]
            if any(f.startswith("vol_") for f in variant.extra_features) and history is None:
                sys.exit(f"variant {name} needs --history")
            fit = fit_engine(kb, rate=load_base_rate(None), cd=load_cd91(None), variant=None if variant.is_current else variant, volume_history=history)
            path, message = write_log(fit, kb, args.log_dir, regions, args.note)
            print(f"[{name}] {message}")
        return
    scored = score_log(load_log(args.log_dir), kb, args.realized, archive, verified)
    pd.set_option("display.width", 200)
    if args.command == "score":
        print(f"기록 {len(scored)}행 중 실현된 {int(scored['y'].notna().sum())}행 (realized={args.realized}, 데이터 마지막 주 {kb.sale.index.max().date()})")
        print(summarize(scored, kb).round(3).to_string(index=False))
        return
    if not args.cand:
        parser.error("--cand is required for compare")
    pairs = pair_models(scored, args.base, args.cand)
    print(compare_logged(pairs, seoul_region_keys(kb.hierarchy)).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
