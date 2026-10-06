"""Outside-Seoul long-horizon shrinkage toward the pooled historical mean: the single candidate `OS_mean50` (rule and weight fixed in kbforecast/outside.py before any result).

    python scripts/experiment_outside_seoul.py path/to/KB_주간시계열.xlsx [--n-boot 2000] [--out-dir ...]

Baseline and pooled mean are computed exactly as in scripts/experiment_long_horizon.py (observed-only labels, 26-week refits, 2-week origin spacing, first origin 2014-01-06):
the baseline is the production model (Ridge alone from 78 weeks); the pooled mean is the walk-forward average of closed labels. Horizons 104 and 208 decide; 52 and 78 weeks are
reported with the same formula; weights 0.25 / 0.75 are DESCRIPTIVE sensitivity (not candidates). Nothing is changed in production and nothing is added to the app.
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
from kbforecast import outside as O  # noqa: E402
from kbforecast.engine import model_columns  # noqa: E402
from kbforecast.evaluation import walk_forward  # noqa: E402
from kbforecast.experiments import RunConfig, variant_frame  # noqa: E402
from kbforecast.features import build_features, make_targets  # noqa: E402
from kbforecast.forecastlog import git_commit, library_versions  # noqa: E402
from kbforecast.kb_panel import parse_kb_panel, seoul_region_keys  # noqa: E402
from kbforecast.overlay import load_base_rate, load_cd91, rate_signal_weekly  # noqa: E402
from kbforecast.variants import CURRENT  # noqa: E402

HORIZONS = (52, 78, 104, 208)
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    kb = parse_kb_panel(args.workbook.read_bytes())
    cfg = RunConfig(n_boot=args.n_boot, min_train_rows=5000, primary=O.PRIMARY, extra=(52, 78))
    out = args.out_dir or Path("experiments") / f"outside_seoul_{kb.fingerprint[:10]}_{kb.last_date:%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    log = E.ExperimentLog(Path("experiments/log.jsonl"))
    (out / "run_metadata.json").write_text(json.dumps({
        "workbook_sha256": kb.fingerprint, "last_date": str(kb.last_date.date()), "git_commit": git_commit(Path(__file__).resolve().parents[1]), "versions": library_versions(),
        "candidate": "OS_mean50", "weight": O.WEIGHT, "decision_horizons": list(O.PRIMARY), "reported_horizons": [52, 78], "labels": "observed-only", "evidence": E.EVIDENCE_LEVEL}, ensure_ascii=False, indent=2), encoding="utf-8")

    fs = build_features(kb)
    cols = model_columns(fs)
    dates = fs.log_price.index
    seoul = seoul_region_keys(kb.hierarchy) & set(fs.log_price.columns)
    z_rate = rate_signal_weekly(load_base_rate(None), load_cd91(None), dates)
    observed = kb.observed["sale"] if kb.observed is not None else None

    def const(Xtr, ytr, Xp):
        return np.full(len(Xp), float(np.mean(ytr)))

    per_horizon: dict[int, dict] = {}
    rows, sens = [], []
    for h in HORIZONS:
        y = make_targets(fs.log_price, h, observed)
        base = variant_frame(kb, fs, cols, CURRENT, h, y, cfg, z_rate, seoul)
        mean = walk_forward(fs, ["r1"], const, cfg.wf(h), y=y)
        wide = pd.DataFrame({"y": base["y"], "base": base["pred"], "mean": mean["pred"].reindex(base.index)}).dropna()
        wide["cand"] = O.shrink_outside_seoul(wide, "base", "mean", seoul, O.WEIGHT)
        in_seoul = wide.index.get_level_values("region").isin(seoul)
        assert (wide.loc[in_seoul, "cand"] == wide.loc[in_seoul, "base"]).all()  # Seoul unchanged by construction
        groups = {k: v for k, v in E.region_sets(wide.index.get_level_values("region").unique()).items() if v}
        res = E.compare(wide, "base", "cand", h, groups, args.n_boot)
        per_horizon[h] = res
        for gname, g in res["groups"].items():
            rows.append({"horizon": h, "group": gname, "n": g["n"], "origins": g["origins"], "MAE_base": g["base"]["MAE_log_pp"], "MAE_cand": g["cand"]["MAE_log_pp"], "rel_pct": g["rel_MAE_pct"],
                         "bias_base": g["base"]["bias_pred_minus_actual_pp"], "bias_cand": g["cand"]["bias_pred_minus_actual_pp"], "p_abs": g["hac_abs"]["p_two_sided"], "p_sq": g["hac_sq"]["p_two_sided"],
                         "boot90_rel_lo": g["boot_abs"]["rel_lo_pct"], "boot90_rel_hi": g["boot_abs"]["rel_hi_pct"], **{f"rel_{k}": v["rel_pct"] for k, v in g["periods"].items()}})
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):  # descriptive sensitivity: weight on the model for non-Seoul regions (1.0 = baseline, 0.0 = pooled mean)
            c = O.shrink_outside_seoul(wide, "base", "mean", seoul, w)
            sub = ~in_seoul
            sens.append({"horizon": h, "weight_on_model": w, "MAE_outside_Seoul": float((c[sub] - wide.loc[sub, "y"]).abs().mean() * 100)})
    table = log.annotate(pd.DataFrame(rows), "p_abs")
    verdict = O.decide_outside(per_horizon)
    log.record("OS_mean50", "O", {"weight": O.WEIGHT, "applies_to": "regions outside the Seoul 28 series", "mean": "pooled walk-forward mean of closed labels"}, [104, 208], {"verdict": verdict["verdict"], "score": verdict["score"]}, verdict, "outside-Seoul long-horizon shrinkage")
    table.to_csv(out / "scorecard_exploratory.csv", index=False, float_format="%.4f")
    pd.DataFrame(sens).pivot(index="weight_on_model", columns="horizon", values="MAE_outside_Seoul").to_csv(out / "sensitivity_descriptive.csv", float_format="%.4f")
    (out / "verdict_exploratory.json").write_text(json.dumps({**verdict, "evidence": E.EVIDENCE_LEVEL, "multiple_testing": log.summary_text()}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    show = table[table["group"].isin(["서울 28", "서울 외", "전체"])]
    cols_show = ["horizon", "group", "n", "MAE_base", "MAE_cand", "rel_pct", "bias_base", "bias_cand", "p_raw", "p_bonferroni_with_prior_assumed", "boot90_rel_lo", "boot90_rel_hi"] + [c for c in table.columns if c.startswith("rel_20")]
    print("== OS_mean50 vs baseline (log-return MAE pp; negative rel% = better); 104/208 decide, 52/78 reported\n" + show[cols_show].round(3).to_string(index=False))
    print("\n== descriptive sensitivity: outside-Seoul MAE by weight on the model (1.0 = baseline, 0.0 = pooled mean only)\n" + pd.DataFrame(sens).pivot(index="weight_on_model", columns="horizon", values="MAE_outside_Seoul").round(3).to_string())
    print(f"\nVERDICT (exploratory): {verdict['verdict']}  score={verdict['score']:+.2f}%  {'; '.join(verdict['reasons'])}\n{E.EVIDENCE_LEVEL}\n{log.summary_text()}")
    print(f"\nNo production setting was changed and nothing was added to the app. Saved to {out}")


if __name__ == "__main__":
    main()
