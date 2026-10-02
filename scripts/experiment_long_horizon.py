"""Long-horizon round: 52 / 104 / 208-week forecasts are the PRIMARY horizons here (78 weeks is reported alongside).

    python scripts/experiment_long_horizon.py path/to/KB_주간시계열.xlsx [--n-boot 2000] [--out-dir ...] [--reproduction-cut 2026-08-24]

Order: (1) the baseline of the long horizons (52/78/104/208) is reproduced from the workbook cut at --reproduction-cut against the frozen artifact
(validation_runs/long_observed_*, which has all four horizons); a verdict other than `pass` stops the run (--accept-reproduction-differences to go on);
(2) context: the baseline model against the pooled historical mean and 'no change' on identical rows; (3) the pre-specified long-horizon candidates
  R_alpha_cv      Ridge alpha chosen on a purged time-ordered validation block (grid 3e3..1e6, MAE)
  R_group_alpha   one such alpha for the Seoul series and one for the rest (outside Seoul the pooled mean beats the model at 208 weeks, Seoul does not)
  L_longmem       + 104-week return and deviation from the 156 / 260-week mean of the price
  L_valuation     + price-to-jeonse level and its deviation from the region's own past
  L_all           both feature sets
as an exploratory scorecard (same rows choose and grade; raw and multiplicity-adjusted p) and as an external validation of the selection procedure.
Decisions use 52/104/208 weeks with equal weight (Seoul 28 relative MAE change, outside-Seoul, period and bootstrap checks of evalsuite.decide).
At 208 weeks realised labels end in 2022-08, so the 2022-23 window is partial and 2024+ has no 208-week rows: that horizon rests on about two market cycles
of origins at most; non-overlapping origins are about two. Nothing here changes the production model or adds a 208-week forecast to the app.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kbforecast import evalsuite as E  # noqa: E402
from kbforecast.engine import model_columns  # noqa: E402
from kbforecast.experiments import RunConfig, load_frozen_baselines, naive_context, reproduction_report, run_feature_experiments, variant_frame  # noqa: E402
from kbforecast.features import build_features, make_targets  # noqa: E402
from kbforecast.forecastlog import git_commit, library_versions  # noqa: E402
from kbforecast.kb_panel import parse_kb_panel, seoul_region_keys, truncate_panel  # noqa: E402
from kbforecast.overlay import load_base_rate, load_cd91, rate_signal_weekly  # noqa: E402
from kbforecast.variants import CURRENT, NAMED_VARIANTS  # noqa: E402

LONG = (52, 104, 208)
CANDIDATES = ("R_alpha_cv", "R_group_alpha", "L_longmem", "L_valuation", "L_all")
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--reproduction-cut", default="2026-08-24", help="cut the workbook here to reproduce the frozen baseline (its last week); '' skips the check")
    parser.add_argument("--min-train-rows", type=int, default=5000)
    parser.add_argument("--accept-reproduction-differences", action="store_true")
    args = parser.parse_args()

    kb = parse_kb_panel(args.workbook.read_bytes())
    cfg = RunConfig(n_boot=args.n_boot, min_train_rows=args.min_train_rows, primary=LONG, extra=(78,))
    out = args.out_dir or Path("experiments") / f"long_horizon_{kb.fingerprint[:10]}_{kb.last_date:%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    log = E.ExperimentLog(Path("experiments/log.jsonl"))
    meta = {"workbook_sha256": kb.fingerprint, "last_date": str(kb.last_date.date()), "git_commit": git_commit(Path(__file__).resolve().parents[1]), "versions": library_versions(),
            "primary_horizons": list(LONG), "extra_horizons": [78], "refit_every": cfg.refit_every, "labels": "observed-only" if kb.observed is not None else "filled values allowed", "candidates": list(CANDIDATES), "evidence": E.EVIDENCE_LEVEL}
    (out / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1. reproduce the long-horizon baseline before anything else
    if args.reproduction_cut:
        cut = truncate_panel(kb, args.reproduction_cut)
        frozen, _, fmeta = load_frozen_baselines(horizons=(52, 78, 104, 208))
        fs = build_features(cut)
        cols = model_columns(fs)
        seoul = seoul_region_keys(cut.hierarchy) & set(fs.log_price.columns)
        z = rate_signal_weekly(load_base_rate(None), load_cd91(None), fs.log_price.index)
        obs = cut.observed["sale"] if cut.observed is not None else None
        fresh = {h: variant_frame(cut, fs, cols, CURRENT, h, make_targets(fs.log_price, h, obs), cfg, z, seoul) for h in (52, 78, 104, 208)}
        rep = reproduction_report(fresh, frozen, fingerprint=cut.fingerprint, frozen_fingerprint=fmeta["long_config"]["data_fingerprint"], fresh_columns=cols, frozen_columns=fmeta["long_feature_columns"],
                                  refit_every=cfg.refit_every, frozen_refit_every=fmeta["long_config"]["refit_every"], labels_observed=obs is not None, frozen_labels=fmeta["long_config"]["labels"], require_same_fingerprint=False)
        rep["table"].to_csv(out / "baseline_reproduction.csv", index=False, float_format="%.6f")
        (out / "baseline_reproduction.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "table"}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"== baseline reproduction at 52/78/104/208 weeks (workbook cut at {args.reproduction_cut}): {rep['verdict'].upper()}  problems={rep['problems']} warnings={rep['warnings']}")
        print(rep["table"][["horizon", "rows_frozen", "frozen_rows_missing_in_fresh", "y_max_abs_diff", "final_mean_abs_diff", "final_max_abs_diff", "MAE_seoul28_frozen", "MAE_seoul28_fresh"]].round(6).to_string(index=False))
        if rep["verdict"] != "pass" and not args.accept_reproduction_differences:
            sys.exit(f"STOP: the long-horizon baseline does not reproduce ({rep['verdict']}); see {out}/baseline_reproduction.csv")

    # 2. context: model vs pooled historical mean vs no change on identical rows
    ctx = naive_context(kb, cfg, (52, 78, 104, 208))
    ctx.to_csv(out / "context_model_vs_naive.csv", index=False, float_format="%.4f")
    print("\n== the baseline model against naive forecasts (identical rows; MAE in log-return percentage points)")
    print(ctx[ctx["group"].isin(["서울 28", "서울시 지수", "서울 25개 구", "서울 외", "전체"])][["horizon", "group", "n", "MAE_model", "MAE_pooled_mean", "MAE_no_change", "model_vs_pooled_mean_pct", "bias_model", "bias_pooled_mean"]].round(2).to_string(index=False))

    # 3. candidates
    variants = [NAMED_VARIANTS[n] for n in CANDIDATES]
    result = run_feature_experiments(kb, cfg, variants, out, log)
    print(f"\n== 1. EXPLORATORY scorecard (52/104/208 decide, 78 reported; same rows choose and grade)\n   {result['multiple_testing']}")
    sc = result["scorecard"]
    wanted = ["candidate", "horizon", "n", "MAE_base", "MAE_cand", "rel_pct", "rel_2014-2019", "rel_2020-2021", "rel_2022-2023", "rel_2024+", "p_raw", "p_bonferroni_logged", "p_holm_this_table", "boot90_rel_lo", "boot90_rel_hi"]
    show = sc[(sc["group"] == "서울 28") & (~sc["candidate"].str.contains("Ridge only"))][[c for c in wanted if c in sc.columns]]
    print(show.round(3).to_string(index=False))
    out_show = sc[(sc["group"].isin(["서울 외", "전체"])) & (~sc["candidate"].str.contains("Ridge only"))][["candidate", "horizon", "group", "MAE_base", "MAE_cand", "rel_pct"]]
    print("\n-- outside Seoul / all regions\n" + out_show.round(3).to_string(index=False))
    for name, v in result["verdicts"].items():
        print(f"   {name:34s} {v['verdict']}(exploratory)  score={v['score']:+.2f}% over {v['score_horizons']}  78w rel%={v.get('extra_horizons', {})}  {'; '.join(v['reasons'][:2])}")
    print("\n== 2. EXTERNAL validation of the selection procedure")
    print(result["selection_history"][["window", "chosen", "passing_internal", "reason"]].to_string(index=False))
    ss = result["selection_summary"]
    if len(ss):
        print(ss[ss["group"].isin(["서울 28", "서울 외", "전체"])][["horizon", "group", "n", "MAE_base", "MAE_selected", "rel_pct", "p_raw", "p_bonferroni_logged", "boot90_rel_lo", "boot90_rel_hi"]].round(3).to_string(index=False))
    d = result["selection_decision"]
    print(f"   procedure verdict: {d['verdict']}  score={d['score']:+.2f}% over {d['score_horizons']}  {'; '.join(d['reasons'][:3])}\n   {d['evidence']}")
    print(f"\nNo production setting was changed and nothing was added to the app. Saved to {out}")


if __name__ == "__main__":
    main()
