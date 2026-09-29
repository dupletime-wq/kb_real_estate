"""Production forecasting pipeline: pooled direct multi-horizon Ridge+HGB blend with conformal intervals.

Design decisions are the ones that survived walk-forward validation (see tests/validation notes):
  * pool all regions in the workbook (pooling is the single largest gain vs. per-region fits)
  * features: own momentum/vol + KB sentiment indices + region-vs-parent relative momentum
    (broadcast macro / market-level features overfit in out-of-time tests and are excluded by default)
  * one direct model per anchor horizon; the weekly path is interpolated between anchors
  * intervals: online conformal on walk-forward residuals, normalised by trailing volatility
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import models as M
from .evaluation import WFConfig, baseline_predictions, walk_forward
from .features import FeatureSet, build_features, make_targets
from .intervals import conformal_quantiles, scale_from_vol
from .kb_panel import KBPanel

ANCHORS = (4, 8, 13, 20, 26, 39, 52)
REL_FEATURES = ("rel13", "rel26", "cs_rank13", "cs_rank26")
# validated per-horizon settings: ridge alpha (stronger shrinkage for longer horizons), conformal levels for ~80% coverage
RIDGE_ALPHA = {4: 10000.0, 8: 30000.0, 13: 30000.0, 20: 100000.0, 26: 100000.0, 39: 100000.0, 52: 100000.0}
CONFORMAL_LEVELS = {4: (0.07, 0.93), 8: (0.07, 0.93), 13: (0.07, 0.93), 20: (0.05, 0.95), 26: (0.05, 0.95), 39: (0.05, 0.95), 52: (0.05, 0.95)}
HGB_KW = {
    "default": dict(),
    26: dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=6, min_samples_leaf=400, l2=10.0),
}


def model_columns(fs: FeatureSet, use_macro: bool = False) -> list[str]:
    groups = fs.groups
    cols = list(groups.get("own", [])) + list(groups.get("own2", [])) + list(groups.get("sentiment", [])) + list(groups.get("sent2", []))
    cols += [c for c in REL_FEATURES if c in fs.X.columns]
    if use_macro:
        cols += list(groups.get("macro", []))
    return cols


def blend_model(horizon: int) -> M.ModelFn:
    alpha = RIDGE_ALPHA.get(horizon, 100000.0)
    hgb_kw = HGB_KW.get(horizon, HGB_KW["default"])
    return M.blend([M.ridge_model(alpha), M.hgb_model(**hgb_kw)])


@dataclass(frozen=True)
class EngineFit:
    """Everything needed to forecast any region in the panel, computed once per workbook (kept small for caching)."""

    log_price: pd.DataFrame  # wide log sale index
    anchors: tuple[int, ...]
    columns: list[str]
    predictions: dict[int, pd.DataFrame]  # per anchor: walk-forward + live-origin rows with y/pred/lo/hi
    baselines: dict[int, pd.DataFrame]  # per anchor: rw / drift26 predictions aligned to `predictions`
    contributions: dict[int, pd.DataFrame]  # per anchor: ridge-component contribution by feature group (rows = regions)
    hierarchy: pd.DataFrame
    kb_fingerprint: str
    last_date: pd.Timestamp
    use_macro: bool
    settings: dict = field(default_factory=dict)


def fit_engine(
    kb: KBPanel,
    macro_weekly: pd.DataFrame | None = None,
    anchors: tuple[int, ...] = ANCHORS,
    first_origin: str | None = None,
    refit_every: int = 39,
    eval_step: int = 2,
    use_macro: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> EngineFit:
    """Walk-forward fit at every anchor horizon (also yields the live-origin forecasts and calibrated intervals)."""
    fs = build_features(kb, macro_weekly if use_macro else None)
    cols = model_columns(fs, use_macro)
    dates = fs.log_price.index
    if first_origin is None:
        first_origin = str(max(pd.Timestamp("2014-01-06"), dates[-1] - pd.Timedelta(weeks=416)).date())
    vol = 0.5 * fs.X["vol52"] + 0.5 * fs.X["vol13"]
    preds: dict[int, pd.DataFrame] = {}
    bases: dict[int, pd.DataFrame] = {}
    contrib: dict[int, pd.DataFrame] = {}
    for i, h in enumerate(anchors):
        if progress:
            progress(i, len(anchors), f"{h}주 예측 모델 학습·검증 중")
        cfg = WFConfig(horizon=h, first_origin=first_origin, eval_step=eval_step, refit_every=refit_every)
        pred = walk_forward(fs, cols, blend_model(h), cfg)
        pred = _append_live_origin(fs, cols, blend_model(h), h, pred, cfg)
        lq, uq = CONFORMAL_LEVELS.get(h, (0.05, 0.95))
        preds[h] = conformal_quantiles(pred, scale_from_vol(vol, h), h, dates, lower_q=lq, upper_q=uq, window_weeks=260)
        bases[h] = pd.DataFrame(
            {"rw": baseline_predictions(fs, h, "rw").reindex(pred.index), "drift26": baseline_predictions(fs, h, "drift26").reindex(pred.index)}
        )
        contrib[h] = _ridge_contributions(fs, cols, h)
    if progress:
        progress(len(anchors), len(anchors), "완료")
    settings = {"first_origin": first_origin, "refit_every": refit_every, "eval_step": eval_step, "anchors": list(anchors)}
    return EngineFit(fs.log_price, tuple(anchors), cols, preds, bases, contrib, kb.hierarchy, kb.fingerprint, dates[-1], use_macro, settings)


def _feature_group_map(fs: FeatureSet, cols: list[str]) -> dict[str, str]:
    out = {}
    for group, names in fs.groups.items():
        for n in names:
            out[n] = group
    for n in REL_FEATURES:
        out[n] = "market"
    return {c: out.get(c, "other") for c in cols}


GROUP_LABELS = {"own": "자체 모멘텀·변동성", "own2": "자체 모멘텀(보조)", "sentiment": "KB 심리지표", "sent2": "KB 심리지표(보조)", "market": "지역 상대강도", "macro": "거시지표", "jeonse": "전세"}


def _ridge_contributions(fs: FeatureSet, cols: list[str], h: int) -> pd.DataFrame:
    """Interpretability: the ridge half of the blend, decomposed into (coefficient x standardized feature) by group,
    at the live origin. Values are cumulative-return contributions in percentage points (relative to the mean forecast)."""
    from sklearn.linear_model import Ridge

    X, y = fs.X, make_targets(fs.log_price, h)
    n = len(fs.log_price)
    date_pos = pd.Series(np.arange(n), index=fs.log_price.index)
    row_pos = date_pos.reindex(X.index.get_level_values("date")).to_numpy()
    usable = X[["r52", "vol52"]].notna().all(axis=1).to_numpy()
    tr = usable & y.notna().to_numpy() & (row_pos + h <= n - 1)
    live = usable & (row_pos == n - 1)
    if not live.any() or tr.sum() < 5000:
        return pd.DataFrame()
    from .models import _prepare

    a, b = _prepare(X.loc[tr, cols], X.loc[live, cols])
    yt = y.to_numpy()[tr]
    model = Ridge(alpha=RIDGE_ALPHA.get(h, 100000.0)).fit(a, yt - yt.mean())
    parts = b * model.coef_[None, :]
    gmap = _feature_group_map(fs, cols)
    frame = pd.DataFrame(parts, index=X.index[live].get_level_values("region"), columns=cols)
    grouped = frame.T.groupby(pd.Series(gmap)).sum().T * 100.0
    return grouped.rename(columns=GROUP_LABELS)


def _append_live_origin(fs: FeatureSet, cols: list[str], model_fn: M.ModelFn, h: int, pred: pd.DataFrame, cfg: WFConfig) -> pd.DataFrame:
    last = fs.log_price.index[-1]
    have = pred.index.get_level_values("date")
    if len(pred) and (have == last).any():
        return pred
    X = fs.X
    y = make_targets(fs.log_price, h)
    date_pos = pd.Series(np.arange(len(fs.log_price)), index=fs.log_price.index)
    row_pos = date_pos.reindex(X.index.get_level_values("date")).to_numpy()
    usable = X[["r52", "vol52"]].notna().all(axis=1).to_numpy()
    tr = usable & y.notna().to_numpy() & (row_pos + h <= len(date_pos) - 1)
    live = usable & (row_pos == len(date_pos) - 1)
    if not live.any() or tr.sum() < cfg.min_train_rows:
        return pred
    p = model_fn(X.loc[tr, cols], y.to_numpy()[tr], X.loc[live, cols])
    extra = pd.DataFrame({"y": y.to_numpy()[live], "pred": p}, index=X.index[live])
    return pd.concat([pred, extra]).sort_index()


@dataclass(frozen=True)
class RegionForecast:
    region: str
    horizon: int
    origin: pd.Timestamp
    last_value: float
    path: pd.DataFrame  # columns: date, p10, p50, p90 (index level)
    anchor_table: pd.DataFrame  # per anchor: pred/lo/hi cumulative return (pct)


def forecast_region(fit: EngineFit, region: str, horizon: int) -> RegionForecast:
    if region not in fit.log_price.columns:
        raise KeyError(region)
    last = fit.last_date
    last_level = float(np.exp(fit.log_price[region].iloc[-1]))
    rows = []
    for h in fit.anchors:
        df = fit.predictions[h]
        try:
            r = df.loc[(last, region)]
        except KeyError:
            continue
        if not np.isfinite(r["pred"]):
            continue
        lo = r["lo"] if np.isfinite(r["lo"]) else r["pred"]
        hi = r["hi"] if np.isfinite(r["hi"]) else r["pred"]
        rows.append({"h": h, "pred": float(r["pred"]), "lo": float(min(lo, r["pred"])), "hi": float(max(hi, r["pred"]))})
    if not rows:
        raise ValueError(f"{region}: 예측에 필요한 이력이 부족합니다.")
    table = pd.DataFrame(rows)
    steps = np.arange(1, horizon + 1)
    xs = np.concatenate([[0], table["h"].to_numpy(dtype=float)])

    def interp(col: str) -> np.ndarray:
        ys = np.concatenate([[0.0], table[col].to_numpy()])
        return np.interp(steps, xs, ys)

    dates = pd.date_range(last + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    path = pd.DataFrame(
        {
            "date": dates,
            "p10": last_level * np.exp(interp("lo")),
            "p50": last_level * np.exp(interp("pred")),
            "p90": last_level * np.exp(interp("hi")),
        }
    )
    tbl = table.assign(pred_pct=(np.exp(table["pred"]) - 1) * 100, lo_pct=(np.exp(table["lo"]) - 1) * 100, hi_pct=(np.exp(table["hi"]) - 1) * 100)
    return RegionForecast(region, horizon, last, last_level, path, tbl[["h", "pred_pct", "lo_pct", "hi_pct"]])


def validation_summary(fit: EngineFit, regions: tuple[str, ...] | None, horizons: tuple[int, ...] = (13, 26, 52)) -> pd.DataFrame:
    """Walk-forward accuracy vs simple baselines on the realised part of the engine's own predictions."""
    rows = []
    for h in horizons:
        if h not in fit.predictions:
            continue
        df = fit.predictions[h].join(fit.baselines[h])
        if regions is not None:
            df = df[df.index.get_level_values("region").isin(regions)]
        df = df.dropna(subset=["y", "pred", "drift26"])
        if df.empty:
            continue
        err = df["pred"] - df["y"]
        e_dr = df["drift26"] - df["y"]
        e_rw = df["rw"] - df["y"]
        rows.append(
            {
                "horizon": h,
                "n": int(len(df)),
                "model_MAE_pp": float(err.abs().mean() * 100),
                "drift26_MAE_pp": float(e_dr.abs().mean() * 100),
                "randomwalk_MAE_pp": float(e_rw.abs().mean() * 100),
                "skill_vs_drift26": float(1 - (err**2).mean() / (e_dr**2).mean()),
                "interval_coverage": float(((df["y"] >= df["lo"]) & (df["y"] <= df["hi"])).mean()),
                "from": df.index.get_level_values("date").min(),
                "to": df.index.get_level_values("date").max(),
            }
        )
    return pd.DataFrame(rows)
