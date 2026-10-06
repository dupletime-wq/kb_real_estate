"""Candidate runs on the KB workbook, in a fixed order:

  1. baseline reproduction against the frozen artifact (identity of data / features / labels / schedule, then row-level label, forecast and overlay
     differences). On the workbook the frozen artifact was made from (sha256 match) a verdict other than `pass` STOPS the run unless
     --accept-reproduction-differences is given (then the reason must be understood and is recorded). Any other workbook (e.g. a newer one) is run
     as `new_workbook`: the comparison is informational (extra weeks and revisions are expected) and every output goes to its own directory.
  2. HGB early-stopping record of the baseline, error decomposition by state known at the origin, candidate A on the fresh baseline.
  3. Families B (sentiment x price), C (observation quality), D (HGB iteration count) and V (monthly trading volume, common-period training).
     Two products per family group: an EXPLORATORY scorecard (candidate vs baseline over the whole period, with raw and multiplicity-adjusted p-values and
     the logged number of candidates) and the external validation of the SELECTION PROCEDURE (choose on origins closed before each window, freeze, grade on
     the window; combinations only from candidates that passed in the selection period).

    python scripts/experiment_candidates.py path/to/KB_주간시계열.xlsx [--families B,C,D,V] [--history trade_history/<date>/seoul_daily_counts.csv.gz]
                                                                         [--out-dir ...] [--n-boot 2000] [--accept-reproduction-differences]

Nothing here changes the production model: verdicts are exploratory and the engine only uses a variant if it is named explicitly; the final check is the
forward forecast log (scripts/forecast_log.py log --variants ...). The historical years were used by earlier experiments and are not untouched data.
The weekly trading-volume test (`tv_*`, scripts/validate_volume.py) and the monthly volume candidates (`vol_*`, family V here) are different experiments.
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
from kbforecast.experiments import RunConfig, candidate_A_report, load_frozen_baselines, reproduction_report, run_feature_experiments, variant_frame  # noqa: E402
from kbforecast.engine import model_columns  # noqa: E402
from kbforecast.features import build_features, make_targets  # noqa: E402
from kbforecast.forecastlog import git_commit, library_versions  # noqa: E402
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
    parser.add_argument("--truncate-to", help="cut the workbook panel to this last week (e.g. 2026-08-24) to reproduce the frozen artifact from a NEWER workbook: the fingerprint cannot match, so the gate rests on labels, rows, features, schedule and overlay")
    parser.add_argument("--reproduction-only", action="store_true", help="stop after the baseline reproduction report")
    parser.add_argument("--accept-reproduction-differences", action="store_true", help="continue although the baseline reproduction verdict is not 'pass' (the reason is recorded)")
    args = parser.parse_args()

    from kbforecast.kb_panel import truncate_panel

    kb = parse_kb_panel(args.workbook.read_bytes())
    frozen, _, fmeta = load_frozen_baselines()
    truncated = bool(args.truncate_to)
    if truncated:
        kb = truncate_panel(kb, args.truncate_to)
    same_workbook = kb.fingerprint == fmeta["long_config"]["data_fingerprint"]
    gated = same_workbook or truncated  # the gate applies when the run is meant to reproduce the frozen artifact
    mode = "reproduction" if same_workbook else ("reproduction_by_truncation" if truncated else "new_workbook")
    out = args.out_dir or Path("experiments") / (f"candidates_{kb.fingerprint[:10]}_{kb.last_date:%Y%m%d}" if gated else f"new_workbook_{kb.fingerprint[:10]}_{kb.last_date:%Y%m%d}")
    if truncated:
        out = out.with_name(out.name + "_truncated")
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
    meta = {"mode": mode, "workbook_sha256": kb.fingerprint, "frozen_workbook_sha256": fmeta["long_config"]["data_fingerprint"], "last_date": str(kb.last_date.date()), "git_commit": git_commit(Path(__file__).resolve().parents[1]),
            "versions": library_versions(), "feature_columns": cols, "labels": "observed-only" if observed is not None else "filled values allowed (no observed mask)", "refit_every": cfg.refit_every, "families": fams,
            "evidence": E.EVIDENCE_LEVEL}
    (out / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mode: {mode} -> {out}")

    # 1. baseline reproduction (before any candidate)
    base_frames, hgb_rec = {}, {}
    for h in (13, 26, 52, 78, 104):
        rec: list = []
        base_frames[h] = variant_frame(kb, fs, cols, CURRENT, h, make_targets(fs.log_price, h, observed), cfg, z, seoul, history, hgb_record=rec)
        hgb_rec[h] = rec
    rep = reproduction_report(
        base_frames, frozen, fingerprint=kb.fingerprint, frozen_fingerprint=fmeta["long_config"]["data_fingerprint"], fresh_columns=cols, frozen_columns=fmeta["long_feature_columns"],
        refit_every=cfg.refit_every, frozen_refit_every=fmeta["long_config"]["refit_every"], labels_observed=observed is not None, frozen_labels=fmeta["long_config"]["labels"],
        require_same_fingerprint=not truncated,
    )
    rep["table"].to_csv(out / "baseline_reproduction.csv", index=False, float_format="%.6f")
    (out / "baseline_reproduction.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "table"}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n== baseline reproduction vs frozen artifact: {rep['verdict'].upper()}  checks={rep['checks']}  problems={rep['problems']}  warnings={rep['warnings']}")
    print(rep["table"].round(6).to_string(index=False))
    if gated and rep["verdict"] != "pass" and not args.accept_reproduction_differences:
        sys.exit(f"STOP: the fresh baseline does not reproduce the frozen artifact ({rep['verdict']}). Find the cause (see {out}/baseline_reproduction.csv); "
                 "rerun with --accept-reproduction-differences only after it is understood.")
    if args.reproduction_only:
        print(f"reproduction-only: stopped after the report ({out})")
        return
    if not gated:
        print("(a different workbook: differences are informational; this run is stored separately and is not the reproduction of the frozen baseline)")

    hgb_summary = {str(h): {"fits": len(r), "early_stopping_active": int(sum(x["early_stopping_active"] for x in r)), "median_n_iter": float(np.median([x["n_iter"] for x in r])) if r else None,
                            "max_iter": r[0]["max_iter"] if r else None, "min_n_train": min((x["n_train"] for x in r), default=None)} for h, r in hgb_rec.items()}
    (out / "hgb_baseline_behaviour.json").write_text(json.dumps(hgb_summary, indent=2), encoding="utf-8")
    print("\n== HGB behaviour in the baseline fits (production settings)\n" + json.dumps(hgb_summary, indent=1))

    # 2. where the baseline errors concentrate (state known at the origin) and candidate A on the fresh baseline
    state_rows = []
    for h, f in base_frames.items():
        sets = E.region_sets(f.index.get_level_values("region").unique())
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
    table, verdicts = candidate_A_report(base_frames, dates, out, log, cfg.n_boot)
    table = log.annotate(table, "p_abs_two_sided")
    table.to_csv(out / "candidate_A_summary.csv", index=False, float_format="%.4f")
    (out / "candidate_A_verdicts.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n== candidate A (exploratory): score horizons are listed; a score over fewer than 13/26/52 weeks is NOT a 13/26/52-week average")
    for name, v in verdicts.items():
        print(f"{name:45s} {v['verdict']}  score={v['score']:+.2f}% over {v['score_horizons']}  {'; '.join(v['reasons'][:2])}")

    # 3. families
    variants = [v for name, v in NAMED_VARIANTS.items() if name != "current" and name[0] in fams]
    groups = [("B_C_D", [v for v in variants if not v.name.startswith("V_")], None, out)]
    vol = [v for v in variants if v.name.startswith("V_")]
    if vol:
        from kbforecast.candidates import build_candidate_features
        from kbforecast.trades import verify_history, verify_mapping_against_kb

        print("\n== trade history checks:", json.dumps({k: (v if not isinstance(v, list) or len(v) < 8 else f"{len(v)} items") for k, v in verify_history(history).items()}, ensure_ascii=False))
        print("== district names vs KB panel:", verify_mapping_against_kb(kb))
        common_start = build_candidate_features(kb, ("vol_chg3",), history)["vol_chg3"]["서울특별시"].first_valid_index()
        print(f"== monthly volume features exist from {common_start.date()}; baseline AND candidates are trained on labels from that date only (common-period comparison)")
        groups.append(("V (monthly volume, common period)", vol, common_start, out / "volume_common_period"))
    for label, group, train_start, gout in groups:
        if not group:
            continue
        result = run_feature_experiments(kb, cfg, group, gout, log, history, train_start=train_start)
        print(f"\n== [{label}] 1. EXPLORATORY scorecard (same rows choose and grade; not corrected for the number of tries unless stated)")
        print(f"   {result['multiple_testing']}")
        sc = result["scorecard"]
        show = sc[(sc["group"] == "서울 28") & (~sc["candidate"].str.contains("Ridge only"))][["candidate", "horizon", "MAE_base", "MAE_cand", "rel_pct", "p_raw", "p_bonferroni_logged", "p_bonferroni_with_prior_assumed", "p_holm_this_table", "boot90_rel_lo", "boot90_rel_hi"]]
        print(show.round(3).to_string(index=False))
        for name, v in result["verdicts"].items():
            print(f"   {name:36s} {v['verdict']}(exploratory)  score={v['score']:+.2f}% over {v['score_horizons']}  extra-horizon rel%={v.get('extra_horizons', {})}  {'; '.join(v['reasons'][:2])}")
        print(f"\n== [{label}] 2. EXTERNAL validation of the selection procedure (choose on closed origins, freeze, grade on the next window)")
        print(result["selection_history"][["window", "chosen", "passing_internal", "reason"]].to_string(index=False))
        ss = result["selection_summary"]
        if len(ss):
            print(ss[ss["group"].isin(["서울 28", "서울 외", "전체"])][["horizon", "group", "n", "MAE_base", "MAE_selected", "rel_pct", "p_raw", "p_bonferroni_logged", "boot90_rel_lo", "boot90_rel_hi"]].round(3).to_string(index=False))
        d = result["selection_decision"]
        print(f"   procedure verdict: {d['verdict']}  score={d['score']:+.2f}% over {d['score_horizons']}  {'; '.join(d['reasons'][:3])}")
        print(f"   {d['evidence']}")
    print(f"\nNo production setting was changed. Candidates that improve are only candidates for forward logging (scripts/forecast_log.py log --variants ...). Saved to {out}")


if __name__ == "__main__":
    main()
