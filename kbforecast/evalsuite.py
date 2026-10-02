"""MAE-oriented evaluation suite: error decomposition, loss-differential tests that respect overlapping horizons, and a pre-set decision rule.

Units. `y` and `pred` are cumulative LOG returns (the unit of the engine and of every earlier README table). The main metric is
`MAE_log_pp = mean(|pred - y|) * 100`. `MAE_simple_pp = mean(|expm1(pred) - expm1(y)|) * 100` is auxiliary (same horizon, ordinary
return, percentage points); the two are never mixed in one column. Residual = actual - prediction, bias = prediction - actual.

Tests. Candidates are compared on the loss difference per forecast ORIGIN: for each origin the regions of a group are averaged together
first (regions share the market, so they are not independent), then the series over origins is tested with a Newey-West (HAC) variance
whose lag covers the overlap of h-week windows, and with a moving-block bootstrap over origins (all regions of an origin stay together).
Nothing here is an "untouched" test set: the 2024+ rows were already used by earlier experiments, and every candidate that is tried is
recorded in the experiment log so that p-values can be read against the number of tries (`ExperimentLog`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .evaluation import PERIODS, newey_west_var
from .trades import SEOUL_GROUPS, SEOUL_GU_CODES

PRIMARY_HORIZONS = (13, 26, 52)
EXTRA_HORIZONS = (78, 104)
EVAL_STEP = 2
# --- pre-set decision rule (fixed before any candidate is run; see `decide`) ---
ADOPT_MAX_SCORE = -1.0  # mean over 13/26/52w of the Seoul-28 relative MAE change (%), must be at most this
OUTSIDE_SEOUL_MAX = 0.5  # relative MAE change (%) outside Seoul may not exceed this at any of 13/26/52w
PERIOD_MAX = 2.0  # no period may worsen the Seoul-28 MAE by more than this (% , averaged over 13/26/52w)
BOOT_LEVEL = 0.90  # bootstrap interval level; the upper end must be below 0 at >= 2 of the 3 horizons


def region_sets(regions: Iterable[str]) -> dict[str, set[str]]:
    """Evaluation groups by name: Seoul city index, the 2 Seoul halves + 25 districts, 'Seoul 28', everything outside Seoul, all."""
    regs = set(regions)
    city = {"서울특별시"} & regs
    gu = set(SEOUL_GU_CODES) & regs
    groups = set(SEOUL_GROUPS) & regs
    seoul = city | gu | groups
    return {"서울 28": seoul, "서울시 지수": city, "서울 25개 구": gu, "서울 외": regs - seoul, "전체": regs}


def _mask(frame: pd.DataFrame, regions: set[str]) -> np.ndarray:
    return np.asarray(frame.index.get_level_values("region").isin(regions))


def error_stats(frame: pd.DataFrame, pred: str = "pred") -> dict[str, float]:
    d = frame.dropna(subset=["y", pred])
    if d.empty:
        return {k: float("nan") for k in ("n", "MAE_log_pp", "MAE_simple_pp", "RMSE_log_pp", "bias_pred_minus_actual_pp", "median_residual_actual_minus_pred_pp")} | {"n": 0}
    e = d[pred] - d["y"]
    simple = np.expm1(d[pred]) - np.expm1(d["y"])
    return {
        "n": int(len(d)), "MAE_log_pp": float(e.abs().mean() * 100), "MAE_simple_pp": float(simple.abs().mean() * 100),
        "RMSE_log_pp": float(math.sqrt((e**2).mean()) * 100), "bias_pred_minus_actual_pp": float(e.mean() * 100),
        "median_residual_actual_minus_pred_pp": float((-e).median() * 100),
    }


def period_slices(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = frame.index.get_level_values("date")
    return {name: frame[(dates >= a) & (dates <= b)] for name, (a, b) in PERIODS.items()}


def decompose(frame: pd.DataFrame, pred: str, horizon: int, groups: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """Error table by group x period (plus 'all periods') for one horizon."""
    groups = groups or region_sets(frame.index.get_level_values("region").unique())
    rows = []
    for gname, regs in groups.items():
        if not regs:
            continue
        sub = frame[_mask(frame, regs)]
        for pname, part in {"전체 기간": sub, **period_slices(sub)}.items():
            rows.append({"horizon": horizon, "group": gname, "period": pname, **error_stats(part, pred)})
    return pd.DataFrame(rows)


def origin_state(trend: pd.Series, n_bins: int = 3, min_obs: int = 2000) -> pd.Series:
    """Causal tercile state of a quantity known at the origin (e.g. the trailing 26-week return, or a sentiment change).

    Thresholds at origin t come from all values observed up to and including t (expanding quantiles over the pooled panel), never from
    later dates, so a row's state is what an analyst could have computed then. Returns 0..n_bins-1 (NaN until `min_obs` values exist).
    """
    t = trend.dropna()
    dates = t.index.get_level_values("date")
    order = np.argsort(dates.to_numpy(), kind="stable")
    sorted_vals = t.to_numpy()[order]
    sorted_dates = dates.to_numpy()[order]
    uniq, first_idx, counts = np.unique(sorted_dates, return_index=True, return_counts=True)
    state = np.full(len(sorted_vals), np.nan)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    for k, date in enumerate(uniq):
        end = first_idx[k] + counts[k]  # everything up to and including this origin
        if end < min_obs:
            continue
        cuts = np.quantile(sorted_vals[:end], qs)
        block = sorted_vals[first_idx[k]:end]
        state[first_idx[k]:end] = np.searchsorted(cuts, block, side="right")
    out = pd.Series(np.nan, index=t.index)
    out.iloc[order] = state
    return out.reindex(trend.index)


def decompose_by_state(frame: pd.DataFrame, pred: str, state: pd.Series, labels: tuple[str, ...], horizon: int, group: set[str]) -> pd.DataFrame:
    sub = frame[_mask(frame, group)]
    st = state.reindex(sub.index)
    rows = []
    for k, label in enumerate(labels):
        part = sub[(st == k).to_numpy()]
        if len(part):
            rows.append({"horizon": horizon, "state": label, **error_stats(part, pred)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- loss differentials and tests
def origin_losses(frame: pd.DataFrame, base: str, cand: str, regions: set[str] | None = None, kind: str = "abs") -> pd.DataFrame:
    """Per origin: cross-sectional mean loss of the baseline and of the candidate (pp), on rows where both predictions and y exist."""
    d = frame.dropna(subset=["y", base, cand])
    if regions is not None:
        d = d[_mask(d, regions)]
    eb, ec = (d[base] - d["y"]) * 100, (d[cand] - d["y"]) * 100
    if kind == "abs":
        lb, lc = eb.abs(), ec.abs()
    elif kind == "sq":
        lb, lc = eb**2, ec**2
    else:
        raise ValueError(kind)
    out = pd.DataFrame({"base": lb, "cand": lc}).groupby(level="date").mean()
    return out


def hac_test(diff: np.ndarray, horizon: int, eval_step: int = EVAL_STEP) -> dict[str, float]:
    """Mean of the origin-level loss difference (cand - base; negative = candidate better) with a Newey-West t-statistic."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return {"n_origins": n, "mean_diff": float(d.mean()) if n else float("nan"), "t": float("nan"), "p_two_sided": float("nan"), "p_improvement": float("nan")}
    from scipy import stats

    lags = int(math.ceil(horizon / eval_step)) + 1
    t = d.mean() / math.sqrt(newey_west_var(d, lags) / n)
    return {"n_origins": n, "mean_diff": float(d.mean()), "t": float(t), "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(t)))), "p_improvement": float(stats.norm.cdf(t))}


def block_bootstrap(base: np.ndarray, cand: np.ndarray, horizon: int, n_boot: int = 2000, seed: int = 0, level: float = BOOT_LEVEL, eval_step: int = EVAL_STEP) -> dict[str, float]:
    """Moving-block bootstrap over origins (circular, block length = number of origins an h-week window overlaps).

    The unit resampled is an origin with all its regions already averaged, so regions of the same origin always travel together.
    Returns percentile intervals of the mean loss difference (pp) and of the relative change (%).
    """
    b, c = np.asarray(base, float), np.asarray(cand, float)
    ok = np.isfinite(b) & np.isfinite(c)
    b, c = b[ok], c[ok]
    n = len(b)
    if n < 20:
        return {"diff_lo": float("nan"), "diff_hi": float("nan"), "rel_lo_pct": float("nan"), "rel_hi_pct": float("nan"), "block": 0}
    block = int(min(max(math.ceil(horizon / eval_step), 2), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    mb, mc = b[idx].mean(axis=1), c[idx].mean(axis=1)
    diff, rel = mc - mb, (mc / mb - 1.0) * 100.0
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"diff_lo": float(np.percentile(diff, lo)), "diff_hi": float(np.percentile(diff, hi)), "rel_lo_pct": float(np.percentile(rel, lo)), "rel_hi_pct": float(np.percentile(rel, hi)), "block": block}


def compare(frame: pd.DataFrame, base: str, cand: str, horizon: int, groups: dict[str, set[str]], n_boot: int = 2000, seed: int = 0) -> dict:
    """Full comparison of one candidate prediction column against the baseline on the identical rows of `frame`."""
    common = frame.dropna(subset=["y", base, cand])
    out: dict = {"horizon": horizon, "base": base, "cand": cand, "groups": {}}
    for gname, regs in groups.items():
        sub = common[_mask(common, regs)] if regs else common.iloc[0:0]
        if sub.empty:
            continue
        sb, sc = error_stats(sub, base), error_stats(sub, cand)
        losses = origin_losses(common, base, cand, regs, "abs")
        sq = origin_losses(common, base, cand, regs, "sq")
        rel = (sc["MAE_log_pp"] / sb["MAE_log_pp"] - 1) * 100 if sb["MAE_log_pp"] else float("nan")
        periods = {}
        for pname, part in period_slices(sub).items():
            if len(part):
                pb, pc = error_stats(part, base), error_stats(part, cand)
                periods[pname] = {"n": pb["n"], "MAE_base": pb["MAE_log_pp"], "MAE_cand": pc["MAE_log_pp"], "rel_pct": (pc["MAE_log_pp"] / pb["MAE_log_pp"] - 1) * 100}
        out["groups"][gname] = {
            "n": sb["n"], "origins": int(len(losses)), "base": sb, "cand": sc, "diff_MAE_log_pp": sc["MAE_log_pp"] - sb["MAE_log_pp"], "rel_MAE_pct": rel,
            "hac_abs": hac_test((losses["cand"] - losses["base"]).to_numpy(), horizon),
            "hac_sq": hac_test((sq["cand"] - sq["base"]).to_numpy(), horizon),
            "boot_abs": block_bootstrap(losses["base"].to_numpy(), losses["cand"].to_numpy(), horizon, n_boot, seed),
            "periods": periods,
        }
    return out


def selection_score(per_horizon: dict[int, dict]) -> float:
    """Pre-set primary criterion: mean over 13/26/52w of the Seoul-28 relative MAE change (%). Each horizon counts equally, so the
    larger errors of longer horizons cannot dominate."""
    vals = [per_horizon[h]["groups"]["서울 28"]["rel_MAE_pct"] for h in PRIMARY_HORIZONS if h in per_horizon and "서울 28" in per_horizon[h]["groups"]]
    return float(np.mean(vals)) if vals else float("nan")


def decide(per_horizon: dict[int, dict], outside_unchanged_by_design: bool = False) -> dict:
    """Pre-set rule. 채택 (adopt) only if every check passes; 보류 (hold: log forward, keep the current model) if the score is negative but a
    check fails; 기각 (reject) if the score is not negative. Missing horizons make the verdict 보류 at best."""
    reasons: list[str] = []
    score = selection_score(per_horizon)
    have = [h for h in PRIMARY_HORIZONS if h in per_horizon and "서울 28" in per_horizon[h]["groups"]]
    if len(have) < len(PRIMARY_HORIZONS):
        reasons.append(f"missing primary horizons {sorted(set(PRIMARY_HORIZONS) - set(have))}")
    if not np.isfinite(score) or score >= 0:
        return {"verdict": "기각", "score": score, "reasons": reasons + ["Seoul-28 mean relative MAE change is not negative"]}
    ok = True
    if score > ADOPT_MAX_SCORE:
        ok = False
        reasons.append(f"score {score:.2f}% is above {ADOPT_MAX_SCORE}%")
    for h in have:
        g = per_horizon[h]["groups"].get("서울 외")
        if g is None:
            if not outside_unchanged_by_design:
                ok = False
                reasons.append(f"outside-Seoul effect cannot be verified at {h}w")
        elif g["rel_MAE_pct"] > OUTSIDE_SEOUL_MAX:
            ok = False
            reasons.append(f"outside Seoul worsens at {h}w by {g['rel_MAE_pct']:.2f}%")
    for pname in PERIODS:
        vals = [per_horizon[h]["groups"]["서울 28"]["periods"][pname]["rel_pct"] for h in have if pname in per_horizon[h]["groups"]["서울 28"]["periods"]]
        if vals and np.mean(vals) > PERIOD_MAX:
            ok = False
            reasons.append(f"period {pname} worsens by {np.mean(vals):.2f}% (avg over horizons)")
    n_sig = sum(1 for h in have if per_horizon[h]["groups"]["서울 28"]["boot_abs"]["rel_hi_pct"] < 0)
    if n_sig < 2:
        ok = False
        reasons.append(f"bootstrap {int(BOOT_LEVEL * 100)}% interval excludes zero at only {n_sig} of {len(have)} horizons")
    if len(have) < len(PRIMARY_HORIZONS):
        ok = False
    return {"verdict": "채택" if ok else "보류", "score": score, "reasons": reasons}


# ----------------------------------------------------------------------------- experiment registry
@dataclass
class ExperimentLog:
    """Append-only record of every candidate that is evaluated, so multiplicity can be read off at the end."""

    path: Path

    def record(self, candidate: str, family: str, params: dict, horizons: list[int], result: dict, decision: dict | None = None, note: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "candidate": candidate, "family": family, "params": params,
            "horizons": horizons, "result": result, "decision": decision, "note": note,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=float) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def n_candidates(self) -> int:
        """Distinct candidates tried (family 'reference' entries are comparisons of existing settings, not new tries)."""
        return len({(e["family"], e["candidate"]) for e in self.entries() if e["family"] != "reference"})

    def adjusted_p(self, p: float) -> float:
        """Bonferroni adjustment over every candidate recorded so far (conservative; reported next to the raw p, never instead of it)."""
        return float(min(1.0, p * max(self.n_candidates(), 1)))
