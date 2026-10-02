"""Experiment runners shared by the scripts: the frozen-prediction analysis (no workbook needed) and the full candidate runs (workbook needed).

Everything here follows the pre-set protocol of `evalsuite` (identical rows, observed-only labels, 26-week refits, the same overlay and the
same labels for baseline and candidate; verdicts from `evalsuite.decide`; every candidate goes into the experiment log).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evalsuite as E
from . import models as M
from .calibration import BIAS_CANDIDATES, median_bias_correction, walk_forward_select
from .candidates import build_candidate_features, with_candidates
from .engine import OVERLAY_MAX_HORIZON, RIDGE_ALPHA, blend_model, model_columns
from .evaluation import WFConfig, walk_forward
from .features import FeatureSet, build_features, make_targets
from .kb_panel import KBPanel, seoul_region_keys
from .overlay import apply_seoul_rate_overlay, load_base_rate, load_cd91, rate_signal_weekly
from .variants import CURRENT, NAMED_VARIANTS, EngineVariant

FROZEN_LONG = Path("validation_runs/long_observed_20260824")
FROZEN_VOLUME = Path("validation_runs/volume_20260824")
FROZEN_HORIZONS = (13, 26, 52, 78, 104)


# ----------------------------------------------------------------------------- frozen predictions (no workbook)
def load_frozen_baselines(long_dir: Path = FROZEN_LONG, volume_dir: Path = FROZEN_VOLUME) -> tuple[dict[int, pd.DataFrame], pd.DatetimeIndex, dict]:
    """h -> frame indexed (date, region): y, raw (blend), pred (final baseline forecast), r26 (trailing 26-week return at the origin if stored).

    52/78/104 weeks: every region, from validate_long. 13/26 weeks: the 28 Seoul series only, from validate_volume (its `base` column).
    Final forecast = raw blend + Seoul rate overlay up to 52 weeks (recomputed from the stored raw forecast for 13/26 weeks; at 52 weeks
    it is checked against the stored overlay column), the raw blend from 78 weeks.
    """
    cfg_long = json.loads((long_dir / "config.json").read_text(encoding="utf-8"))
    cfg_vol = json.loads((volume_dir / "config.json").read_text(encoding="utf-8"))
    dates = pd.date_range("2008-04-07", cfg_long["last_date"], freq="W-MON")
    z = rate_signal_weekly(load_base_rate(None), load_cd91(None), dates)
    frames: dict[int, pd.DataFrame] = {}
    notes: dict = {"overlay_check_h52": None}
    vol = pd.read_csv(volume_dir / "predictions_seoul.csv.gz", parse_dates=["date"])
    for h in FROZEN_HORIZONS:
        if h in (13, 26):
            d = vol[vol["horizon"] == h].set_index(["date", "region"]).sort_index()
            f = pd.DataFrame({"y": d["y"], "raw": d["base"], "r26": np.nan})
            seoul = E.region_sets(f.index.get_level_values("region").unique())["서울 28"]
            adj, _ = apply_seoul_rate_overlay(f[["y", "raw"]].rename(columns={"raw": "pred"}), z, h, dates, seoul)
            f["pred"] = adj["pred"]
        else:
            d = pd.read_csv(long_dir / f"predictions_h{h}.csv.gz", parse_dates=["date"]).set_index(["date", "region"]).sort_index()
            f = pd.DataFrame({"y": d["y"], "raw": d["model"], "r26": d["drift26"] * 26.0 / h})
            if h <= OVERLAY_MAX_HORIZON:
                f["pred"] = d["model+overlay"]
                seoul = E.region_sets(f.index.get_level_values("region").unique())["서울 28"]
                adj, _ = apply_seoul_rate_overlay(f[["y", "raw"]].rename(columns={"raw": "pred"}), z, h, dates, seoul)
                notes["overlay_check_h52"] = {"max_abs_diff_recomputed_vs_stored": float(np.nanmax(np.abs(adj["pred"].to_numpy() - f["pred"].to_numpy())))}
            else:
                f["pred"] = f["raw"]
        frames[h] = f
    meta = {
        "long_config": {k: cfg_long[k] for k in ("git_commit", "data_fingerprint", "last_date", "labels", "engine_version", "refit_every", "eval_step", "first_origin", "ridge_alpha", "interval_levels")},
        "long_feature_columns": cfg_long["feature_columns"], "volume_config": {k: cfg_vol[k] for k in ("git_commit", "data_fingerprint", "labels", "refit_every")}, **notes,
    }
    return frames, dates, meta


def candidate_A_report(frames: dict[int, pd.DataFrame], dates: pd.DatetimeIndex, out: Path, log: E.ExperimentLog, n_boot: int = 2000, external_start: str = "2020-01-06") -> tuple[pd.DataFrame, dict]:
    """Candidate A on final baseline forecasts: fixed configurations on the full period and the walk-forward-selected one on 2020+."""
    rows, per_cand = [], {}
    for h, f in frames.items():
        base = f[["y", "pred"]].copy()
        regs = base.index.get_level_values("region").unique()
        sets = E.region_sets(regs)
        seoul = sets["서울 28"]
        groups = {k: v for k, v in sets.items() if v}
        has_outside = bool(sets["서울 외"])
        results = {}
        for cfg in (c for c in BIAS_CANDIDATES if not c.is_none):
            if cfg.scope == "global" and not has_outside:
                continue  # without outside-Seoul residuals a 'global' median would just be a Seoul median
            corrected, _ = median_bias_correction(base, dates, h, cfg, seoul)
            both = base.rename(columns={"pred": "base"}).assign(cand=corrected["pred"])
            results[cfg.name] = E.compare(both, "base", "cand", h, groups, n_boot)
        selected, table = walk_forward_select(base, dates, h, seoul)
        table.to_csv(out / f"selection_history_h{h}.csv", index=False, float_format="%.4f")
        ext = base.index.get_level_values("date") >= pd.Timestamp(external_start)
        both = base[ext].rename(columns={"pred": "base"}).assign(cand=selected.loc[ext, "pred"])
        results["A_selected (walk-forward, external 2020+)"] = E.compare(both, "base", "cand", h, groups, n_boot)
        for name, res in results.items():
            if "서울 28" not in res["groups"]:  # e.g. the external window has no rows at this horizon: nothing to report, nothing invented
                continue
            per_cand.setdefault(name, {})[h] = res
            g = res["groups"]["서울 28"]
            rows.append({
                "candidate": name, "horizon": h, "n_rows": g["n"], "MAE_base": g["base"]["MAE_log_pp"], "MAE_cand": g["cand"]["MAE_log_pp"], "rel_pct": g["rel_MAE_pct"],
                "bias_base": g["base"]["bias_pred_minus_actual_pp"], "bias_cand": g["cand"]["bias_pred_minus_actual_pp"], "RMSE_base": g["base"]["RMSE_log_pp"], "RMSE_cand": g["cand"]["RMSE_log_pp"],
                "p_abs_two_sided": g["hac_abs"]["p_two_sided"], "p_sq_two_sided": g["hac_sq"]["p_two_sided"], "boot90_rel_lo": g["boot_abs"]["rel_lo_pct"], "boot90_rel_hi": g["boot_abs"]["rel_hi_pct"],
                "outside_rel_pct": res["groups"].get("서울 외", {}).get("rel_MAE_pct", np.nan), "all_rel_pct": res["groups"].get("전체", {}).get("rel_MAE_pct", np.nan),
            })
            log.record(name, "A", {"horizon": h}, [h], {"rel_MAE_seoul28_pct": g["rel_MAE_pct"], "p_abs": g["hac_abs"]["p_two_sided"]}, None, "frozen predictions" if not has_outside else "")
    verdicts = {}
    for name, per_h in per_cand.items():
        primary = {h: r for h, r in per_h.items() if h in E.PRIMARY_HORIZONS}
        d = E.decide(primary, outside_unchanged_by_design=name.startswith("seoul") or name.startswith("A_selected"))
        verdicts[name] = {**d, "horizons_available": sorted(per_h)}
    return pd.DataFrame(rows), verdicts


# ----------------------------------------------------------------------------- candidate runs on the workbook
@dataclass(frozen=True)
class RunConfig:
    first_origin: str = "2014-01-06"
    refit_every: int = 26
    eval_step: int = 2
    min_train_rows: int = 5000
    n_boot: int = 2000
    primary: tuple[int, ...] = E.PRIMARY_HORIZONS
    extra: tuple[int, ...] = E.EXTRA_HORIZONS

    def wf(self, h: int) -> WFConfig:
        return WFConfig(horizon=h, first_origin=self.first_origin, eval_step=self.eval_step, refit_every=self.refit_every, min_train_rows=self.min_train_rows)


def mask_targets(y: pd.Series, start: pd.Timestamp | None) -> pd.Series:
    """Drop the labels of origins before `start` (NaN): neither training nor scoring can use them. None leaves y untouched."""
    return y if start is None else y.where(y.index.get_level_values("date") >= pd.Timestamp(start))


def variant_frame(
    kb: KBPanel, fs: FeatureSet, cols: list[str], variant: EngineVariant, h: int, y: pd.Series, cfg: RunConfig, z_rate: pd.Series | None, seoul: set[str],
    volume_history: pd.DataFrame | None = None, ridge_only: bool = False, hgb_record: list | None = None,
) -> pd.DataFrame:
    """Final forecasts (y, pred_raw, pred) of one variant at one horizon, exactly as the engine would make them."""
    dates = fs.log_price.index
    fsv, colsv = fs, list(cols)
    if variant.extra_features:
        fsv = with_candidates(fs, build_candidate_features(kb, variant.extra_features, volume_history, variant.volume_extra_lag_weeks))
        colsv = colsv + [c for c in variant.extra_features if c not in colsv]
    model_fn = M.ridge_model(RIDGE_ALPHA.get(h, 1e5)) if ridge_only else blend_model(h, variant.hgb_mode, hgb_record)
    pred = walk_forward(fsv, colsv, model_fn, cfg.wf(h), y=y)
    pred["pred_raw"] = pred["pred"]
    if not ridge_only and z_rate is not None and h <= OVERLAY_MAX_HORIZON:
        adj, _ = apply_seoul_rate_overlay(pred[["y", "pred"]], z_rate, h, dates, seoul)
        pred["pred"] = adj["pred"]
    if variant.bias is not None:
        pred = median_bias_correction(pred, dates, h, variant.bias, seoul)[0]
    return pred


def _common_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(frames)
    idx = frames[names[0]].dropna(subset=["y", "pred"]).index
    for n in names[1:]:
        idx = idx.intersection(frames[n].dropna(subset=["pred"]).index)
    wide = pd.DataFrame({"y": frames[names[0]].loc[idx, "y"]})
    for n, f in frames.items():
        wide[n] = f.loc[idx, "pred"]
    return wide


def run_feature_experiments(
    kb: KBPanel, cfg: RunConfig, variants: list[EngineVariant], out: Path, log: E.ExperimentLog, volume_history: pd.DataFrame | None = None,
    rate=None, cd=None, with_ridge_check: bool = True, train_start: pd.Timestamp | None = None, windows=None, min_internal_rows: int | None = None,
) -> dict:
    """Two separate products, never to be confused.

    1. A scorecard of every candidate against the baseline over the whole evaluation period at the primary horizons (13/26/52 weeks decide,
       78/104 reported). It is EXPLORATORY: the same rows choose and grade, the historical years were already used by earlier experiments,
       and the 90% bootstrap interval is not corrected for the number of candidates tried (raw and multiplicity-adjusted p-values are printed
       side by side). Its verdicts never change the production model.
    2. An external validation of the SELECTION PROCEDURE (kbforecast/selection.py): for each external window the candidate or combination is
       chosen on origins whose labels had closed before the window, frozen, and graded on the window; combinations are only formed from
       candidates that passed in that selection period.

    `train_start` removes the labels of earlier origins for baseline AND candidates alike (e.g. the first date a new data source exists).
    """
    from . import selection as S

    out.mkdir(parents=True, exist_ok=True)
    windows = windows or S.EXTERNAL_WINDOWS
    fs = build_features(kb)
    cols = model_columns(fs)
    dates = fs.log_price.index
    seoul = seoul_region_keys(kb.hierarchy) & set(fs.log_price.columns)
    z_rate = rate_signal_weekly(rate or load_base_rate(None), cd or load_cd91(None), dates)
    observed = kb.observed["sale"] if kb.observed is not None else None
    groups_all = {k: v for k, v in E.region_sets(fs.log_price.columns).items() if v}

    def targets(h: int) -> pd.Series:
        return mask_targets(make_targets(fs.log_price, h, observed), train_start)

    results: dict[str, dict[int, dict]] = {}
    saved: list[pd.DataFrame] = []
    horizons = list(cfg.primary) + list(cfg.extra)
    hgb_log: dict[str, list] = {}
    wides: dict[int, pd.DataFrame] = {}
    for h in horizons:
        y = targets(h)
        frames = {"baseline": variant_frame(kb, fs, cols, CURRENT, h, y, cfg, z_rate, seoul, volume_history)}
        for v in variants:
            rec: list = []
            frames[v.name] = variant_frame(kb, fs, cols, v, h, y, cfg, z_rate, seoul, volume_history, hgb_record=rec)
            hgb_log.setdefault(f"{v.name}@{h}", rec)
        wide = _common_rows(frames)
        wides[h] = wide
        for v in variants:
            res = E.compare(wide.rename(columns={"baseline": "base"}), "base", v.name, h, groups_all, cfg.n_boot)
            results.setdefault(v.name, {})[h] = res
            g = res["groups"]["서울 28"]
            log.record(v.name, v.name.split("_")[0], {"horizon": h, **v.as_dict()}, [h], {"rel_MAE_seoul28_pct": g["rel_MAE_pct"], "p_abs": g["hac_abs"]["p_two_sided"]})
        saved.append(wide[wide.index.get_level_values("region").isin(E.region_sets(wide.index.get_level_values("region").unique())["서울 28"])].reset_index().assign(horizon=h))
        if with_ridge_check and any(v.extra_features for v in variants):
            r_frames = {"baseline": variant_frame(kb, fs, cols, CURRENT, h, y, cfg, None, seoul, volume_history, ridge_only=True)}
            for v in variants:
                if v.extra_features:
                    r_frames[v.name] = variant_frame(kb, fs, cols, v, h, y, cfg, None, seoul, volume_history, ridge_only=True)
            rw = _common_rows(r_frames)
            for v in variants:
                if v.extra_features:
                    results.setdefault(v.name + " [Ridge only]", {})[h] = E.compare(rw.rename(columns={"baseline": "base"}), "base", v.name, h, groups_all, cfg.n_boot)

    # ---- 1. exploratory scorecard (with the multiplicity columns attached to the printed/saved output)
    verdicts = {}
    for name, per_h in results.items():
        primary = {h: r for h, r in per_h.items() if h in cfg.primary}
        verdicts[name] = {**E.decide(primary), "extra_horizons": {h: per_h[h]["groups"]["서울 28"]["rel_MAE_pct"] for h in per_h if h in cfg.extra}}
    rows = []
    for name, per_h in results.items():
        for h, r in per_h.items():
            for gname in ("서울 28", "서울시 지수", "서울 25개 구", "서울 외", "전체"):
                g = r["groups"].get(gname)
                if g:
                    rows.append({"candidate": name, "horizon": h, "group": gname, "n": g["n"], "origins": g["origins"], "MAE_base": g["base"]["MAE_log_pp"], "MAE_cand": g["cand"]["MAE_log_pp"],
                                 "MAE_simple_base": g["base"]["MAE_simple_pp"], "MAE_simple_cand": g["cand"]["MAE_simple_pp"], "rel_pct": g["rel_MAE_pct"], "RMSE_base": g["base"]["RMSE_log_pp"], "RMSE_cand": g["cand"]["RMSE_log_pp"],
                                 "bias_base": g["base"]["bias_pred_minus_actual_pp"], "bias_cand": g["cand"]["bias_pred_minus_actual_pp"], "p_abs": g["hac_abs"]["p_two_sided"], "p_sq": g["hac_sq"]["p_two_sided"],
                                 "boot90_rel_lo": g["boot_abs"]["rel_lo_pct"], "boot90_rel_hi": g["boot_abs"]["rel_hi_pct"], **{f"rel_{p}": d["rel_pct"] for p, d in g["periods"].items()}})
    scorecard = log.annotate(pd.DataFrame(rows), "p_abs")
    scorecard.to_csv(out / "candidate_scorecard_exploratory.csv", index=False, float_format="%.4f")
    (out / "verdicts_exploratory.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.concat(saved).to_csv(out / "predictions_seoul28.csv.gz", index=False, float_format="%.5f", compression="gzip")
    if hgb_log:
        (out / "hgb_fits.json").write_text(json.dumps(hgb_log, ensure_ascii=False, default=float), encoding="utf-8")

    # ---- 2. external validation of the selection procedure
    by_name = {v.name: v for v in variants}
    combo_cache: dict[str, tuple[str, dict[int, pd.Series]]] = {}

    def combo_fn(passing: list[str]):
        feats = tuple(dict.fromkeys(f for n in passing for f in by_name[n].extra_features))
        hgb = next((by_name[n].hgb_mode for n in passing if by_name[n].hgb_mode != "auto"), "auto")  # `passing` is ordered best internal score first
        combo = EngineVariant("combo_" + "+".join(sorted(passing)), feats, hgb, volume_extra_lag_weeks=max(by_name[n].volume_extra_lag_weeks for n in passing))
        if combo.name not in combo_cache:
            series = {}
            for h in horizons:
                f = variant_frame(kb, fs, cols, combo, h, targets(h), cfg, z_rate, seoul, volume_history)
                series[h] = f["pred"]
                g = E.compare(wides[h].assign(**{combo.name: f["pred"].reindex(wides[h].index)}).rename(columns={"baseline": "base"}), "base", combo.name, h, groups_all, cfg.n_boot)
                log.record(combo.name, "combo", {"horizon": h, **combo.as_dict()}, [h], {"rel_MAE_seoul28_pct": g["groups"]["서울 28"]["rel_MAE_pct"], "p_abs": g["groups"]["서울 28"]["hac_abs"]["p_two_sided"]})
            combo_cache[combo.name] = (combo.name, series)
        return combo_cache[combo.name]

    history, ext = S.run_selection(wides, "baseline", [v.name for v in variants], windows, combo_fn, cfg.primary, cfg.extra, min_internal_rows or S.MIN_INTERNAL_SEOUL_ROWS)
    history.to_csv(out / "selection_history.csv", index=False, float_format="%.4f")
    report = S.external_report(ext, cfg.n_boot, cfg.primary)
    sel_rows = []
    for h, r in report["per_horizon"].items():
        for gname in ("서울 28", "서울시 지수", "서울 25개 구", "서울 외", "전체"):
            g = r["groups"].get(gname)
            if g:
                sel_rows.append({"horizon": h, "group": gname, "n": g["n"], "origins": g["origins"], "MAE_base": g["base"]["MAE_log_pp"], "MAE_selected": g["cand"]["MAE_log_pp"], "rel_pct": g["rel_MAE_pct"],
                                 "RMSE_base": g["base"]["RMSE_log_pp"], "RMSE_selected": g["cand"]["RMSE_log_pp"], "bias_base": g["base"]["bias_pred_minus_actual_pp"], "bias_selected": g["cand"]["bias_pred_minus_actual_pp"],
                                 "p_abs": g["hac_abs"]["p_two_sided"], "boot90_rel_lo": g["boot_abs"]["rel_lo_pct"], "boot90_rel_hi": g["boot_abs"]["rel_hi_pct"], **{f"rel_{p}": d["rel_pct"] for p, d in g["periods"].items()}})
    log.record("selection_procedure", "selection", {"windows": [list(w) for w in windows], "candidates": [v.name for v in variants]}, list(cfg.primary), {"score": report["decision"]["score"]})
    sel_tbl = log.annotate(pd.DataFrame(sel_rows), "p_abs") if sel_rows else pd.DataFrame()
    sel_tbl.to_csv(out / "selection_external_summary.csv", index=False, float_format="%.4f")
    if ext:
        pd.concat([f.reset_index().assign(horizon=h) for h, f in ext.items()], ignore_index=True).to_csv(out / "selection_external_predictions.csv.gz", index=False, float_format="%.5f", compression="gzip")
    (out / "selection_verdict.json").write_text(json.dumps({**report["decision"], "share_baseline_chosen": report["share_baseline_chosen"], "multiple_testing": log.summary_text()}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return {
        "verdicts": verdicts, "passed": [n for n, v in verdicts.items() if v["verdict"] == "채택" and "[Ridge only]" not in n], "selection_history": history,
        "selection_decision": report["decision"], "n_candidates": log.n_candidates(), "multiple_testing": log.summary_text(), "scorecard": scorecard, "selection_summary": sel_tbl,
    }


TOL_LABEL = 1e-5  # log-return units: stored labels have 5 decimals
TOL_PRED_MEAN = 5e-4  # mean |final forecast difference| allowed before the reproduction is flagged (0.05 percentage points)
TOL_MAE_REL = 0.5  # % difference of the Seoul-28 MAE allowed before the reproduction is flagged


def reproduction_report(
    fresh: dict[int, pd.DataFrame], frozen: dict[int, pd.DataFrame], *, fingerprint: str, frozen_fingerprint: str, fresh_columns: list[str], frozen_columns: list[str],
    refit_every: int, frozen_refit_every: int, labels_observed: bool, frozen_labels: str, require_same_fingerprint: bool = True,
) -> dict:
    """Is the fresh baseline run the same experiment as the frozen artifact? Checks the identity of data, features, labels and schedule first,
    then compares row by row on the common rows: labels (y), raw blend forecasts, final forecasts (blend + overlay) and the overlay term itself,
    and finally the Seoul-28 / all-region MAE. Verdict: 'pass', 'warn' (identical setup, forecasts differ more than the tolerance - typically
    library versions) or 'fail' (a different dataset, rows, labels, features or schedule). Candidate experiments must not start on 'fail' or 'warn'
    until the cause is understood; `fresh` frames need y, pred_raw, pred, `frozen` frames y, raw, pred. With `require_same_fingerprint=False`
    (a panel cut from a NEWER workbook) the fingerprint cannot match; identical labels on the common rows are then the evidence that the
    history was not revised, and the check is reported as unverifiable instead of failed.
    """
    checks = {
        "data_fingerprint_equal": fingerprint == frozen_fingerprint if require_same_fingerprint else None,  # None: cannot be verified (a cut of another file)
        "feature_columns_equal": list(fresh_columns) == list(frozen_columns),
        "refit_every_equal": int(refit_every) == int(frozen_refit_every),
        "labels_equal": (frozen_labels == "observed") == bool(labels_observed),
    }
    rows = []
    for h, f in frozen.items():
        if h not in fresh:
            continue
        n = fresh[h].rename(columns={"pred_raw": "raw"})
        fz = f.dropna(subset=["y", "pred"])
        nz = n.dropna(subset=["y", "pred"])
        common = fz.index.intersection(nz.index)
        frozen_only = fz.index.difference(nz.index)
        fresh_only = nz.index.difference(fz.index)
        a, b = fz.loc[common], nz.loc[common]
        dy = (a["y"] - b["y"]).abs()
        dp, dr = (a["pred"] - b["pred"]).abs(), (a["raw"] - b["raw"]).abs()
        d_overlay = ((a["pred"] - a["raw"]) - (b["pred"] - b["raw"])).abs()
        sets = E.region_sets(common.get_level_values("region").unique())
        seoul = common.get_level_values("region").isin(sets["서울 28"])
        def mae(frame, sel):
            return float((frame["pred"] - frame["y"]).abs()[sel].mean() * 100) if sel.any() else float("nan")
        row = {
            "horizon": h, "rows_frozen": int(len(fz)), "rows_fresh": int(len(nz)), "common_rows": int(len(common)), "frozen_rows_missing_in_fresh": int(len(frozen_only)), "fresh_extra_rows": int(len(fresh_only)),
            "y_max_abs_diff": float(dy.max()) if len(dy) else float("nan"), "y_rows_differing": int((dy > TOL_LABEL).sum()),
            "raw_mean_abs_diff": float(dr.mean()), "raw_max_abs_diff": float(dr.max()), "raw_corr": float(np.corrcoef(a["raw"], b["raw"])[0, 1]) if len(common) > 2 else float("nan"),
            "final_mean_abs_diff": float(dp.mean()), "final_median_abs_diff": float(dp.median()), "final_p99_abs_diff": float(dp.quantile(0.99)), "final_max_abs_diff": float(dp.max()),
            "final_corr": float(np.corrcoef(a["pred"], b["pred"])[0, 1]) if len(common) > 2 else float("nan"), "overlay_term_max_abs_diff": float(d_overlay.max()),
            "MAE_seoul28_frozen": mae(a, seoul), "MAE_seoul28_fresh": mae(b, seoul), "MAE_all_frozen": mae(a, np.ones(len(a), bool)), "MAE_all_fresh": mae(b, np.ones(len(b), bool)),
        }
        row["MAE_seoul28_rel_diff_pct"] = (row["MAE_seoul28_fresh"] / row["MAE_seoul28_frozen"] - 1) * 100 if row["MAE_seoul28_frozen"] else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    problems = [k for k, ok in checks.items() if ok is False]
    if len(table):
        if (table["frozen_rows_missing_in_fresh"] > 0).any():
            problems.append("frozen evaluation rows are missing in the fresh run")
        if (table["y_max_abs_diff"] > TOL_LABEL).any():
            problems.append("labels differ")
        if (table["overlay_term_max_abs_diff"] > 1e-4).any():
            problems.append("overlay term differs")
    else:
        problems.append("no common horizon")
    warn = []
    if len(table):
        if (table["final_mean_abs_diff"] > TOL_PRED_MEAN).any():
            warn.append("final forecasts differ more than the tolerance")
        if (table["MAE_seoul28_rel_diff_pct"].abs() > TOL_MAE_REL).any():
            warn.append("Seoul-28 MAE differs more than the tolerance")
    verdict = "fail" if problems else ("warn" if warn else "pass")
    return {"verdict": verdict, "checks": checks, "problems": problems, "warnings": warn, "table": table,
            "tolerances": {"label": TOL_LABEL, "final_mean_abs_diff": TOL_PRED_MEAN, "seoul28_mae_rel_pct": TOL_MAE_REL}}
