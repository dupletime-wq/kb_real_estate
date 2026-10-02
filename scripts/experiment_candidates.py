"""Candidate runs on the KB workbook: baseline reproduction, HGB early-stopping record, candidate A on the fresh baseline, and the
pre-specified families B (sentiment x price), C (observation quality), D (HGB iteration count), V (monthly trading volume).

    python scripts/experiment_candidates.py path/to/KB_주간시계열.xlsx [--families B,C,D,V] [--history trade_history/<date>/seoul_daily_counts.csv.gz]
                                                                         [--out-dir experiments/candidates_<last date>] [--n-boot 2000]

Needs the workbook (it is not part of the repository). Protocol (scripts/eval_frozen.py and kbforecast/evalsuite.py explain the rules):
observed-only labels, 26-week refits, 2-week origin spacing, identical evaluation rows for baseline and candidate, final forecast = blend +
Seoul rate overlay up to 52 weeks. Decisions use 13/26/52 weeks only (Seoul-28 mean relative MAE change, outside-Seoul and per-period checks,
block-bootstrap interval); 78/104 weeks are reported separately. Candidates that pass alone are combined and validated again; candidates
that improve but fail a check are listed as 'hold' and should be logged forward (scripts/forecast_log.py log --variants ...).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kbforecast import evalsuite as E  # noqa: E402
from kbforecast.experiments import RunConfig, candidate_A_report, load_frozen_baselines, reproduction_check, run_feature_experiments, variant_frame  # noqa: E402
from kbforecast.engine import model_columns  # noqa: E402
from kbforecast.features import build_features, make_targets  # noqa: E402
from kbforecast.kb_panel import parse_kb_panel, seoul_region_keys  # noqa: E402
from kbforecast.overlay import load_base_rate, load_cd91, rate_signal_weekly  # noqa: E402
from kbforecast.variants import CURRENT, NAMED_VARIANTS  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--families", default="B,C,D,V")
    parser.add_argument("--history", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    kb = parse_kb_panel(args.workbook.read_bytes())
    out = args.out_dir or Path("experiments") / f"candidates_{kb.last_date:%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(n_boot=args.n_boot)
    log = E.ExperimentLog(Path("experiments/log.jsonl"))
    fams = [f.strip().upper() for f in args.families.split(",") if f.strip()]
    history = None
    if "V" in fams:
        files = sorted(glob.glob("trade_history/20*/seoul_daily_counts.csv.gz"))
        path = args.history or (Path(files[-1]) if files else None)
        if path is None or not Path(path).exists():
            print("volume family skipped: no trade history file (run scripts/snapshot_trades.py history, needs a data.go.kr service key)")
            fams.remove("V")
        else:
            history = pd.read_csv(path, dtype={"sgg_cd": str})
    fs = build_features(kb)
    cols = model_columns(fs)
    dates = fs.log_price.index
    seoul = seoul_region_keys(kb.hierarchy) & set(fs.log_price.columns)
    z = rate_signal_weekly(load_base_rate(None), load_cd91(None), dates)
    observed = kb.observed["sale"] if kb.observed is not None else None

    # 1. fresh baseline, reproduction check against the frozen artifacts, HGB behaviour record
    base_frames, hgb_rec = {}, {}
    for h in (13, 26, 52, 78, 104):
        rec: list = []
        base_frames[h] = variant_frame(kb, fs, cols, CURRENT, h, make_targets(fs.log_price, h, observed), cfg, z, seoul, history, hgb_record=rec)
        hgb_rec[h] = rec
    try:
        frozen, _, _ = load_frozen_baselines()
        rep = reproduction_check(base_frames, frozen)
        rep.to_csv(out / "baseline_reproduction.csv", index=False, float_format="%.4f")
        print("\n== baseline reproduction: Seoul-28 MAE, fresh run vs frozen artifact (library versions differ)\n" + rep.round(4).to_string(index=False))
    except FileNotFoundError:
        print("frozen artifacts not found: reproduction check skipped")
    hgb_summary = {str(h): {"fits": len(r), "early_stopping_active": int(sum(x["early_stopping_active"] for x in r)), "median_n_iter": float(np.median([x["n_iter"] for x in r])) if r else None,
                            "max_iter": r[0]["max_iter"] if r else None, "min_n_train": min((x["n_train"] for x in r), default=None)} for h, r in hgb_rec.items()}
    (out / "hgb_baseline_behaviour.json").write_text(json.dumps(hgb_summary, indent=2), encoding="utf-8")
    print("\n== HGB behaviour in the baseline fits (production settings)\n" + json.dumps(hgb_summary, indent=1))
    meta = {"workbook_sha256": kb.fingerprint, "last_date": str(kb.last_date.date()), "feature_columns": cols, "labels": "observed-only" if observed is not None else "filled values allowed (no observed mask)", "refit_every": cfg.refit_every, "families": fams}
    (out / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1b. where the baseline errors concentrate, by what was known at the origin: trailing price trend and buyer-index change (causal terciles)
    state_rows = []
    for h, f in base_frames.items():
        regs = f.index.get_level_values("region").unique()
        sets = E.region_sets(regs)
        for qname, col, labels in (("price trend r26", "r26", ("lowest third", "middle third", "highest third")), ("buyer index 13w change", "buyer_d13", ("falling most", "middle", "rising most"))):
            if col not in fs.X.columns:
                continue
            st = E.origin_state(fs.X[col])
            for gname in ("서울 28", "서울 외"):
                if sets.get(gname):
                    t = E.decompose_by_state(f, "pred", st, labels, h, sets[gname])
                    t.insert(1, "state_variable", qname)
                    t.insert(2, "group", gname)
                    state_rows.append(t)
    if state_rows:
        state_tbl = pd.concat(state_rows, ignore_index=True)
        state_tbl.to_csv(out / "decomposition_by_origin_state.csv", index=False, float_format="%.4f")
        print("\n== baseline error by state known at the origin\n" + state_tbl.round(2).to_string(index=False))

    # 2. candidate A on the fresh baseline (all regions at every horizon, so the 'global' scope is testable at 13/26 weeks too)
    table, verdicts = candidate_A_report(base_frames, dates, out, log, cfg.n_boot)
    table.to_csv(out / "candidate_A_summary.csv", index=False, float_format="%.4f")
    (out / "candidate_A_verdicts.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n== candidate A verdicts")
    for name, v in verdicts.items():
        print(f"{name:45s} {v['verdict']}  score={v['score']:+.2f}%  {'; '.join(v['reasons'][:3])}")

    # 3. feature / model families
    variants = [v for name, v in NAMED_VARIANTS.items() if name != "current" and name[0] in fams]
    groups = [("B_C_D", [v for v in variants if not v.name.startswith("V_")], None, out)]
    vol = [v for v in variants if v.name.startswith("V_")]
    if vol:
        from kbforecast.candidates import build_candidate_features
        from kbforecast.trades import verify_history

        print("\n== trade history checks:", json.dumps({k: (v if not isinstance(v, list) or len(v) < 8 else f"{len(v)} items") for k, v in verify_history(history).items()}, ensure_ascii=False))
        from kbforecast.trades import verify_mapping_against_kb

        print("== district names vs KB panel:", verify_mapping_against_kb(kb))
        vf = build_candidate_features(kb, ("vol_chg3",), history)["vol_chg3"]["서울특별시"]
        common_start = vf.first_valid_index()
        print(f"== volume features exist from {common_start.date()}; baseline AND candidates are trained on labels from that date only (common-period comparison)")
        groups.append(("V", vol, common_start, out / "volume_common_period"))
    for label, group, train_start, gout in groups:
        if not group:
            continue
        result = run_feature_experiments(kb, cfg, group, gout, log, history, train_start=train_start)
        print(f"\n== family verdicts [{label}] (pre-set rule; 13/26/52 weeks decide, 78/104 reported)")
        for name, v in result["verdicts"].items():
            print(f"{name:40s} {v['verdict']}  score={v['score']:+.2f}%  extra-horizon rel%={v.get('extra_horizons', {})}  {'; '.join(v['reasons'][:2])}")
        print(f"passed alone: {result['passed']}; combination: {result['combo']}; candidates recorded so far: {result['n_candidates']}")
        hold = [n for n, v in result["verdicts"].items() if v["verdict"] == "보류"]
        print(f"hold (log forward, keep the current model): {hold}")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
