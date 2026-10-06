"""Outside-Seoul long-horizon shrinkage toward the pooled historical mean (single pre-specified candidate `OS_mean50`).

Why: at 208 weeks the pooled historical mean beats the model outside Seoul (MAE 11.89 vs 13.51 log-return pp, model +13.6% worse) while the model beats the mean in Seoul
(-9.6%); at 104 weeks the model still beats the mean outside Seoul (-8.7%). Nothing was tuned: the weight is fixed at 0.5 before any result is seen (a weight chosen on
the same history would be one more selection step; 0.25 / 0.75 are printed as DESCRIPTIVE sensitivity only and are not candidates).

Candidate: for every region that is NOT one of the 28 Seoul series, final forecast = 0.5 * model forecast + 0.5 * pooled historical mean (the same walk-forward pooled mean
of closed labels that `experiments.naive_context` and the engine's `hist_mean` baseline use); Seoul series keep the model forecast unchanged (by design, verified).

Pre-set decision rule (primary horizons 104 and 208 weeks, equal weight; 52 / 78 weeks are reported, not used):
  채택(exploratory)  outside-Seoul relative MAE change averaged over 104/208 <= -1.0%, AND < 0 at each of the two, AND the 90% moving-block bootstrap interval of each
                     excludes zero (upper bound < 0), AND no period (2014-19, 2020-21, 2022-23, 2024+; averaged over the horizons that have it) of outside Seoul worsens by more
                     than +2%, AND the all-region MAE does not worsen at either horizon
  보류               average < 0 but any condition above fails
  기각               average >= 0
Every verdict is exploratory (same history chooses and grades; the weight was not tuned, but the candidate was motivated by the 208-week result it is graded on); the
production model is never changed automatically and nothing is added to the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT = 0.5
PRIMARY = (104, 208)
ADOPT_MAX_SCORE = -1.0
PERIOD_MAX = 2.0


def shrink_outside_seoul(frame: pd.DataFrame, pred_col: str, mean_col: str, seoul: set[str], weight: float = WEIGHT) -> pd.Series:
    """Forecast of the candidate: Seoul rows keep `pred_col`; every other row is weight * pred + (1 - weight) * pooled mean."""
    in_seoul = frame.index.get_level_values("region").isin(seoul)
    blended = weight * frame[pred_col] + (1.0 - weight) * frame[mean_col]
    return pd.Series(np.where(in_seoul, frame[pred_col], blended), index=frame.index)


def decide_outside(per_horizon: dict[int, dict], primary: tuple[int, ...] = PRIMARY) -> dict:
    """Pre-set verdict from `evalsuite.compare` results (groups '서울 외', '전체', '서울 28') at the primary horizons."""
    reasons: list[str] = []
    missing = [h for h in primary if h not in per_horizon or "서울 외" not in per_horizon[h]["groups"]]
    if missing:
        return {"verdict": "기각", "score": float("nan"), "reasons": [f"no outside-Seoul comparison at {missing}"], "score_horizons": []}
    rel = {h: per_horizon[h]["groups"]["서울 외"]["rel_MAE_pct"] for h in primary}
    score = float(np.mean(list(rel.values())))
    if score >= 0:
        return {"verdict": "기각", "score": score, "reasons": ["outside-Seoul mean relative MAE change is not negative"], "score_horizons": list(primary), "per_horizon_rel": rel}
    if score > ADOPT_MAX_SCORE:
        reasons.append(f"score {score:.2f}% is above {ADOPT_MAX_SCORE}%")
    for h in primary:
        g = per_horizon[h]["groups"]["서울 외"]
        if g["rel_MAE_pct"] >= 0:
            reasons.append(f"outside Seoul is not better at {h}w ({g['rel_MAE_pct']:+.2f}%)")
        if not g["boot_abs"]["rel_hi_pct"] < 0:
            reasons.append(f"bootstrap 90% interval at {h}w includes zero (upper {g['boot_abs']['rel_hi_pct']:+.2f}%)")
        if "전체" in per_horizon[h]["groups"] and per_horizon[h]["groups"]["전체"]["rel_MAE_pct"] > 0:
            reasons.append(f"all-region MAE worsens at {h}w ({per_horizon[h]['groups']['전체']['rel_MAE_pct']:+.2f}%)")
    periods: dict[str, list[float]] = {}
    for h in primary:
        for name, p in per_horizon[h]["groups"]["서울 외"]["periods"].items():
            periods.setdefault(name, []).append(p["rel_pct"])
    for name, vals in periods.items():
        if float(np.mean(vals)) > PERIOD_MAX:
            reasons.append(f"period {name} worsens by {np.mean(vals):.2f}% (avg over horizons)")
    return {"verdict": "채택" if not reasons else "보류", "score": score, "reasons": reasons, "score_horizons": list(primary), "per_horizon_rel": rel}
