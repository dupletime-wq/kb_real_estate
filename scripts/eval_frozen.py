"""Baseline decomposition and candidate A (median-residual correction) on the FROZEN predictions committed in validation_runs/.

    python scripts/eval_frozen.py [--out-dir experiments/frozen_20260824] [--n-boot 2000]

Why frozen predictions: this needs no KB workbook. The baseline is the pooled Ridge+HGB blend produced by scripts/validate_long.py and
scripts/validate_volume.py (observed-only labels, 26-week refits, 2-week origin spacing, first origin 2014-01-06, workbook sha256 in
the config.json files). Coverage of those artifacts:
    52 / 78 / 104 weeks : every region (validation_runs/long_observed_20260824/predictions_h*.csv.gz)
    13 / 26 weeks       : the 28 Seoul series only (validation_runs/volume_20260824/predictions_seoul.csv.gz, column `base`)
so the outside-Seoul and all-region numbers exist for 52/78/104 weeks only. Nothing is re-fitted and no number is filled in.
Final baseline forecast = raw blend + Seoul rate overlay for horizons <= 52 weeks (recomputed here from the stored raw forecasts with the
repository's rate data for 13/26 weeks, checked against the stored overlay column at 52 weeks); the raw blend alone at 78/104 weeks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kbforecast import evalsuite as E  # noqa: E402
from kbforecast.experiments import candidate_A_report, load_frozen_baselines  # noqa: E402
from kbforecast.forecastlog import git_commit  # noqa: E402

LONG = Path("validation_runs/long_observed_20260824")
HORIZONS = (13, 26, 52, 78, 104)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/frozen_20260824"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    frames, dates, meta = load_frozen_baselines()
    import sklearn, scipy  # noqa: E401

    meta["this_run"] = {"git_commit": git_commit(Path(__file__).resolve().parents[1]), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    meta["settings_of_the_frozen_baseline"] = {
        "model": "pooled Ridge (alpha per horizon) + HistGradientBoosting; Ridge alone from 78 weeks", "labels": "observed prices only", "rate_overlay": "Seoul only, anchors <= 52 weeks, base rate + CD91 (1-day lag)",
        "refit_every_weeks": 26, "engine_default_refit_every_weeks": 39, "origin_spacing_weeks": 2,
        "hgb_early_stopping": "scikit-learn default 'auto' (on when > 10,000 training rows; random 10% validation split, not time ordered)",
    }
    (out / "baseline_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- decomposition of the final baseline error: period x group x horizon, then by the price-trend state known at the origin
    dec, states = [], []
    for h, f in frames.items():
        groups = E.region_sets(f.index.get_level_values("region").unique())
        groups = {k: v for k, v in groups.items() if v}
        for col, label in (("raw", "blend (before overlay)"), ("pred", "final baseline")):
            t = E.decompose(f, col, h, groups)
            t.insert(2, "forecast", label)
            dec.append(t)
        if f["r26"].notna().any():
            st = E.origin_state(f["r26"])
            for gname in ("서울 28", "서울 외"):
                if groups.get(gname):
                    s = E.decompose_by_state(f, "pred", st, ("trailing 26w return: lowest third", "middle third", "highest third"), h, groups[gname])
                    s.insert(1, "group", gname)
                    states.append(s)
    dec = pd.concat(dec, ignore_index=True)
    dec.to_csv(out / "decomposition_by_period_group_horizon.csv", index=False, float_format="%.4f")
    state_tbl = pd.concat(states, ignore_index=True) if states else pd.DataFrame()
    state_tbl.to_csv(out / "decomposition_by_price_trend_state.csv", index=False, float_format="%.4f")
    show = dec[(dec["forecast"] == "final baseline") & dec["group"].isin(["서울 28", "서울시 지수", "서울 25개 구", "서울 외", "전체"])]
    print("\n== final baseline, MAE (log-return pp) / bias (pred - actual) by group and period")
    for h in HORIZONS:
        t = show[show["horizon"] == h].pivot_table(index="group", columns="period", values="MAE_log_pp").round(2)
        b = show[show["horizon"] == h].pivot_table(index="group", columns="period", values="bias_pred_minus_actual_pp").round(2)
        print(f"\n[{h}w] MAE\n{t.to_string()}\n[{h}w] bias\n{b.to_string()}")
    print("\n== by price-trend state at the origin (final baseline)\n" + state_tbl.round(2).to_string(index=False))

    # --- candidate A: median-residual correction
    log = E.ExperimentLog(Path("experiments/log.jsonl"))
    table, verdicts = candidate_A_report(frames, dates, out, log, args.n_boot)
    table.to_csv(out / "candidate_A_summary.csv", index=False, float_format="%.4f")
    print("\n== candidate A: Seoul-28 MAE (log-return pp), relative change vs baseline, HAC p (abs-error), 90% block-bootstrap CI (relative %)")
    print(table.round(3).to_string(index=False))
    (out / "candidate_A_verdicts.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n== verdicts (pre-set rule)")
    for name, v in verdicts.items():
        print(f"{name:45s} {v['verdict']}  score={v['score']:+.2f}%  {'; '.join(v['reasons'][:3])}")
    # reference comparisons of EXISTING settings (not new candidates): the production overlay and the pooled-mean baseline, with absolute-error tests
    ref = []
    for h in (52, 78, 104):
        d = pd.read_csv(LONG / f"predictions_h{h}.csv.gz", parse_dates=["date"]).set_index(["date", "region"]).sort_index()
        g = {k: v for k, v in E.region_sets(d.index.get_level_values("region").unique()).items() if v}
        for base_col, cand_col, label in (("model", "model+overlay", "overlay vs raw blend"), ("pooled_mean", "model", "blend vs pooled historical mean")):
            r = E.compare(d, base_col, cand_col, h, g, args.n_boot)
            for gname in ("서울 28", "서울 외", "전체"):
                if gname in r["groups"]:
                    x = r["groups"][gname]
                    ref.append({"comparison": label, "horizon": h, "group": gname, "MAE_base": x["base"]["MAE_log_pp"], "MAE_cand": x["cand"]["MAE_log_pp"], "rel_pct": x["rel_MAE_pct"], "p_abs": x["hac_abs"]["p_two_sided"], "p_sq": x["hac_sq"]["p_two_sided"], "boot90_rel_lo": x["boot_abs"]["rel_lo_pct"], "boot90_rel_hi": x["boot_abs"]["rel_hi_pct"]})
            log.record(label, "reference", {"horizon": h}, [h], {}, None, "frozen predictions")
    ref = pd.DataFrame(ref)
    ref.to_csv(out / "reference_comparisons.csv", index=False, float_format="%.4f")
    print("\n== reference comparisons of existing settings (absolute-error HAC / block bootstrap)\n" + ref.round(3).to_string(index=False))
    print(f"\ncandidates recorded in experiments/log.jsonl: {log.n_candidates()}; saved to {out}")


if __name__ == "__main__":
    main()
