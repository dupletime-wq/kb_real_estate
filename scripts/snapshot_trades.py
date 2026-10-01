"""Collect Seoul apartment sale counts from the MOLIT API (kbforecast/trades.py explains the point-in-time design).

    DATA_GO_KR_KEY=... python scripts/snapshot_trades.py snapshot [--months 4] [--out-dir trade_snapshots]   # weekly, point-in-time
    DATA_GO_KR_KEY=... python scripts/snapshot_trades.py history  [--from 2006-01] [--to 2026-10] [--out-dir trade_history]  # once; final/revised data,
        # written to <out-dir>/parts/ so several ranges can run side by side (the proxy is slow per call), then:
    python scripts/snapshot_trades.py merge [--out-dir trade_history]                                          # parts -> <out-dir>/<date>/

The key comes from the environment (or --key-file); it is never written to the repository or printed. Commit the output directories.
The development account allows 10,000 calls per day: a snapshot is ~100 calls, the full history ~6,300.
"""
from __future__ import annotations

import argparse
from datetime import date
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kbforecast.forecastlog import git_commit  # noqa: E402
from kbforecast.trades import collect, http_getter, months_back, write_snapshot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["snapshot", "history", "merge"])
    parser.add_argument("--months", type=int, default=4, help="snapshot: number of latest contract months (current included)")
    parser.add_argument("--from", dest="start", default="2006-01", help="history: first contract month YYYY-MM")
    parser.add_argument("--to", dest="end", default=None, help="history: last contract month YYYY-MM (default: current)")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()

    if args.mode == "merge":
        import json

        import pandas as pd

        out = args.out_dir or Path("trade_history")
        files = sorted((out / "parts").glob("counts_*.csv.gz"))
        if not files:
            sys.exit("no parts to merge")
        counts = pd.concat([pd.read_csv(f, dtype={"sgg_cd": str}) for f in files], ignore_index=True).drop_duplicates(["sgg_cd", "deal_date"])
        failed = sorted({k for f in (out / "parts").glob("failed_*.json") for k in json.loads(f.read_text())})
        months = sorted({d[:7].replace("-", "") for d in counts["deal_date"]})
        path = write_snapshot(counts, out, months, git_commit(Path(__file__).resolve().parents[1]), kind="history", failed=failed)
        print(f"merged {len(files)} parts -> {path} ({len(counts)} rows, {int(counts['n_all'].sum())} deals); failed district-months: {len(failed)}")
        return
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get("DATA_GO_KR_KEY", "").strip()
    if not key:
        sys.exit("no service key: set DATA_GO_KR_KEY or pass --key-file")
    today = date.today()
    if args.mode == "snapshot":
        months = months_back(today, args.months)
        out = args.out_dir or Path("trade_snapshots")
    else:
        y, m = (int(x) for x in args.start.split("-"))
        ey, em = (int(x) for x in args.end.split("-")) if args.end else (today.year, today.month)
        months = sorted(months_back(date(ey, em, 1), (ey - y) * 12 + em - m + 1))
        out = args.out_dir or Path("trade_history")
    get = http_getter(key)
    counts, failed = collect(get, months, progress=lambda text: print(text, flush=True), retry_pause=30.0)
    if args.mode == "history":
        import json

        (out / "parts").mkdir(parents=True, exist_ok=True)
        tag = f"{min(months)}_{max(months)}"
        counts.to_csv(out / "parts" / f"counts_{tag}.csv.gz", index=False, compression="gzip")
        (out / "parts" / f"failed_{tag}.json").write_text(json.dumps(failed), encoding="utf-8")
        print(f"part {tag} saved ({len(counts)} rows, {int(counts['n_all'].sum())} deals); failed district-months: {len(failed)}")
        sys.exit(1 if failed else 0)
    path = write_snapshot(counts, out, months, git_commit(Path(__file__).resolve().parents[1]), kind=args.mode, failed=failed)
    print(f"saved {path} ({len(counts)} rows, {int(counts['n_all'].sum())} deals, {counts['sgg_cd'].nunique()} districts); failed district-months: {len(failed)}")
    if failed:
        print("failed (recorded in meta.json):", ", ".join(failed[:20]))
        sys.exit(1)


if __name__ == "__main__":
    main()
