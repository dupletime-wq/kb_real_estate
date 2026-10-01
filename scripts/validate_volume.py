"""Pre-registered test of Seoul apartment trading volume as a feature (rule fixed in README, "실거래량: point-in-time 방침").

    python scripts/validate_volume.py path/to/KB_주간시계열.xlsx [--history trade_history/<date>/seoul_daily_counts.csv.gz]

Features (Seoul city, its two halves and the 25 districts only; every other region is left empty for these columns), both built from
contract counts that were *knowable at the as-of date* under the assumed reporting lag (`trades.ASSUMED_LAG_WEEKS`: 12 weeks before
2020-05-25, 8 weeks after; the history is final/revised data, so this lag is the only point-in-time device):
  tv_ratio  = log((4-week count + 1) / (mean of the 4-week count over the previous 156 weeks, at least 104 weeks + 1))
  tv_chg13  = log((4-week count + 1) / (4-week count 13 weeks earlier + 1))
Variants: primary = net counts (reported minus cancelled; the only definition that means the same in every year, because cancelled rows
exist only from 2020) with the assumed lag; sensitivity = lag + 4 weeks; sensitivity = gross counts.
Evaluation is the same walk-forward as scripts/validate_long.py (observed-only labels, identical rows, 26-week refits, 13/26/52 weeks).
Adoption rule, fixed beforehand: adopt only if (a) the Seoul-28 MAE falls at >= 2 of the 3 horizons, (b) at each of those horizons it
falls in >= 3 of the 4 periods (2014-19, 2020-21, 2022-23, 2024+), and (c) (a) and (b) also hold with the lag + 4 weeks and with gross
counts. Otherwise: no basis to adopt under these conditions.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kbforecast.engine import blend_model, model_columns  # noqa: E402
from kbforecast.evaluation import PERIODS, WFConfig, walk_forward  # noqa: E402
from kbforecast.features import build_features, make_targets  # noqa: E402
from kbforecast.forecastlog import git_commit  # noqa: E402
from kbforecast.kb_panel import parse_kb_panel, seoul_region_keys  # noqa: E402
from kbforecast.trades import ASSUMED_LAG_WEEKS, LAG_SWITCH, seoul_weekly, volume_features, weekly_net_and_gross  # noqa: E402

HORIZONS = (13, 26, 52)
pd.set_option("display.width", 220)


def with_features(fs, feats: dict[str, pd.DataFrame]):
    cols = {name: frame.stack(future_stack=True).reindex(fs.X.index).astype("float32") for name, frame in feats.items()}
    return dataclasses.replace(fs, X=pd.concat([fs.X.drop(columns=list(cols), errors="ignore"), pd.DataFrame(cols)], axis=1))


def period_mae(frame: pd.DataFrame, pred: str) -> dict[str, float]:
    out = {}
    dates = frame.index.get_level_values("date")
    for name, (a, b) in PERIODS.items():
        part = frame[(dates >= a) & (dates <= b)]
        out[name] = float((part[pred] - part["y"]).abs().mean() * 100.0) if len(part) else np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("validation_runs"))
    args = parser.parse_args()
    history_path = args.history or Path(sorted(glob.glob("trade_history/20*/seoul_daily_counts.csv.gz"))[-1])
    history = pd.read_csv(history_path, dtype={"sgg_cd": str})

    kb = parse_kb_panel(args.workbook.read_bytes())
    fs = build_features(kb)
    cols = model_columns(fs)
    dates = fs.log_price.index
    seoul = sorted(seoul_region_keys(kb.hierarchy) & set(kb.sale.columns))
    net_w, gross_w = weekly_net_and_gross(history, dates)
    specs = {
        "primary (net, assumed lag)": volume_features(seoul_weekly(net_w, kb.hierarchy)),
        "lag + 4 weeks": volume_features(seoul_weekly(net_w, kb.hierarchy), extra_lag_weeks=4),
        "gross counts": volume_features(seoul_weekly(gross_w, kb.hierarchy)),
    }
    run_dir = args.out_dir / f"volume_{dates[-1]:%Y%m%d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"script": "scripts/validate_volume.py", "git_commit": git_commit(Path(__file__).resolve().parents[1]), "history_file": str(history_path),
              "data_fingerprint": kb.fingerprint, "last_date": str(dates[-1].date()), "lag_weeks": ASSUMED_LAG_WEEKS, "lag_switch_asof": str(LAG_SWITCH.date()),
              "features": ["tv_ratio", "tv_chg13"], "horizons": list(HORIZONS), "labels": "observed-only", "refit_every": 26, "eval_step": 2}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    rows, per_rows, saved = [], [], []
    verdict_inputs: dict[str, dict[int, tuple[bool, int]]] = {s: {} for s in specs}
    for h in HORIZONS:
        y = make_targets(fs.log_price, h, kb.observed["sale"])
        cfg = WFConfig(horizon=h, first_origin="2014-01-06", eval_step=2, refit_every=26)
        base = walk_forward(fs, cols, blend_model(h), cfg, y=y)
        frames = {"base": base}
        for name, feats in specs.items():
            frames[name] = walk_forward(with_features(fs, feats), cols + ["tv_ratio", "tv_chg13"], blend_model(h), cfg, y=y)
        common = base.dropna(subset=["y", "pred"]).index
        for fr in frames.values():
            common = common.intersection(fr.dropna(subset=["pred"]).index)
        wide = pd.DataFrame({"y": base.loc[common, "y"], **{n: fr.loc[common, "pred"] for n, fr in frames.items()}})
        in_seoul = wide.index.get_level_values("region").isin(seoul)
        print(f"\n== {h} weeks: {len(wide)} identical rows, Seoul-28 rows {int(in_seoul.sum())}")
        for label, mask in (("Seoul 28", in_seoul), ("All regions", np.ones(len(wide), bool))):
            sub = wide[mask]
            for n in frames:
                rows.append({"horizon": h, "set": label, "model": n, "n": len(sub), "MAE": float((sub[n] - sub["y"]).abs().mean() * 100.0)})
        s = wide[in_seoul]
        pb = period_mae(s, "base")
        for name in specs:
            pp = period_mae(s, name)
            improved = sum(pp[p] < pb[p] for p in PERIODS)
            per_rows.append({"horizon": h, "spec": name, **{f"base_{p}": pb[p] for p in PERIODS}, **{f"{p}": pp[p] for p in PERIODS}, "periods_improved": improved})
            overall = float((s[name] - s["y"]).abs().mean()) < float((s["base"] - s["y"]).abs().mean())
            verdict_inputs[name][h] = (overall, improved)
        saved.append(s.reset_index().assign(horizon=h))
    table = pd.DataFrame(rows)
    per = pd.DataFrame(per_rows)
    print("\nMAE (pp), identical rows:")
    print(table.pivot_table(index=["set", "horizon"], columns="model", values="MAE").round(3).to_string())
    print("\nSeoul-28 MAE by period (pp): base vs each variant")
    print(per.round(2).to_string(index=False))
    passed = {}
    for name, d in verdict_inputs.items():
        better = [h for h, (ov, _) in d.items() if ov]
        passed[name] = len(better) >= 2 and all(d[h][1] >= 3 for h in better)
        print(f"rule for [{name}]: horizons improved {better}, periods improved per improving horizon {[d[h][1] for h in better]} -> {'PASS' if passed[name] else 'fail'}")
    adopt = all(passed.values())
    print("\nADOPT" if adopt else "\nNO BASIS TO ADOPT under these conditions (pre-registered rule not met)")
    table.to_csv(run_dir / "summary_mae.csv", index=False, float_format="%.4f")
    per.to_csv(run_dir / "summary_periods.csv", index=False, float_format="%.4f")
    pd.concat(saved).to_csv(run_dir / "predictions_seoul.csv.gz", index=False, float_format="%.5f", compression="gzip")
    (run_dir / "verdict.json").write_text(json.dumps({"rule_passed_per_spec": passed, "adopt": adopt}, indent=2), encoding="utf-8")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
