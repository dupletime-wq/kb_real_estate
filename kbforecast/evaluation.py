"""Walk-forward (expanding-window, purged) evaluation for the pooled direct forecasters."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .features import FeatureSet, make_targets
from .models import ModelFn

REQUIRED_FEATURES = ("r52", "vol52")  # a row is usable only when the basic price history exists


@dataclass(frozen=True)
class WFConfig:
    horizon: int
    first_origin: str = "2014-01-06"
    eval_step: int = 2  # weeks between evaluated origins
    refit_every: int = 26  # weeks between refits
    train_regions: tuple[str, ...] | None = None  # None = all regions
    eval_regions: tuple[str, ...] | None = None
    min_train_rows: int = 5000


def _usable(X: pd.DataFrame) -> pd.Series:
    return X[list(REQUIRED_FEATURES)].notna().all(axis=1)


def walk_forward(fs: FeatureSet, feature_cols: list[str], model_fn: ModelFn, cfg: WFConfig, y: pd.Series | None = None) -> pd.DataFrame:
    """Returns rows [date, region, y, pred] for every evaluated origin (y is NaN when the future is unobserved)."""
    h = cfg.horizon
    dates = fs.log_price.index
    y = make_targets(fs.log_price, h) if y is None else y
    X = fs.X
    usable = _usable(X)
    date_pos = pd.Series(np.arange(len(dates)), index=dates)
    row_date_pos = date_pos.reindex(X.index.get_level_values("date")).to_numpy()
    regions = X.index.get_level_values("region")
    train_ok = usable.to_numpy() & y.notna().to_numpy()
    if cfg.train_regions is not None:
        train_ok &= regions.isin(cfg.train_regions)
    eval_ok = usable.to_numpy()
    if cfg.eval_regions is not None:
        eval_ok &= regions.isin(cfg.eval_regions)

    first = int(date_pos.index.searchsorted(pd.Timestamp(cfg.first_origin)))
    last = len(dates) - 1  # origins beyond len-1-h have no realised y but are still predicted (forecast rows)
    out = []
    origin = first
    while origin <= last:
        block_end = min(origin + cfg.refit_every, last + 1)
        tr = train_ok & (row_date_pos + h <= origin)  # purge: label window must close by the refit origin
        if tr.sum() < cfg.min_train_rows:
            origin = block_end
            continue
        Xtr = X.loc[tr, feature_cols]
        ytr = y.to_numpy()[tr]
        ev = eval_ok & (row_date_pos >= origin) & (row_date_pos < block_end) & ((row_date_pos - origin) % cfg.eval_step == 0)
        if ev.any():
            Xev = X.loc[ev, feature_cols]
            pred = model_fn(Xtr, ytr, Xev)
            out.append(pd.DataFrame({"y": y.to_numpy()[ev], "pred": pred}, index=Xev.index))
        origin = block_end
    if not out:
        return pd.DataFrame(columns=["y", "pred"])
    return pd.concat(out).sort_index()


def baseline_predictions(fs: FeatureSet, horizon: int, kind: str) -> pd.Series:
    X = fs.X
    if kind == "rw":
        return pd.Series(0.0, index=X.index)
    if kind.startswith("drift"):
        k = int(kind[5:])
        return X[f"r{k}"] * (horizon / k)
    raise ValueError(kind)


def newey_west_var(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    n = len(d)
    var = float(np.dot(d, d) / n)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(np.dot(d[lag:], d[:-lag]) / n)
    return max(var, 1e-18)


def dm_test(loss_a: pd.Series, loss_b: pd.Series, horizon: int, eval_step: int) -> tuple[float, float]:
    """Diebold-Mariano on per-origin cross-sectional mean loss differentials (a - b); negative => A better."""
    d = (loss_a - loss_b).groupby(level="date").mean().dropna().to_numpy()
    n = len(d)
    if n < 20:
        return float("nan"), float("nan")
    lags = int(math.ceil(horizon / eval_step)) + 1
    stat = d.mean() / math.sqrt(newey_west_var(d, lags) / n)
    from scipy import stats as st

    return float(stat), float(2 * (1 - st.norm.cdf(abs(stat))))


PERIODS = {
    "2014-2019": ("2014-01-01", "2019-12-31"),
    "2020-2021": ("2020-01-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024+": ("2024-01-01", "2030-12-31"),
}


def score(pred: pd.DataFrame, name: str, base_rw: pd.Series, base_drift: pd.Series, horizon: int, eval_step: int, mom_pred: pd.Series | None = None) -> dict:
    """Metrics on realised rows. Errors in cumulative-return percentage points."""
    df = pred.dropna(subset=["y", "pred"]).copy()
    idx = df.index
    err = (df["pred"] - df["y"])
    e_rw = base_rw.reindex(idx) - df["y"]
    e_dr = base_drift.reindex(idx) - df["y"]
    row = {
        "model": name,
        "n": len(df),
        "MAE_pp": float(err.abs().mean() * 100),
        "RMSE_pp": float(math.sqrt((err**2).mean()) * 100),
        "skill_vs_RW": float(1 - (err**2).mean() / (e_rw**2).mean()),
        "skill_vs_drift26": float(1 - (err**2).mean() / (e_dr**2).mean()),
    }
    stat, p = dm_test(err**2, e_dr**2, horizon, eval_step)
    row["DM_vs_drift26_t"], row["DM_p"] = stat, p
    if mom_pred is not None:
        e_m = mom_pred.reindex(idx) - df["y"]
        row["skill_vs_linmom"] = float(1 - (err**2).mean() / (e_m**2).mean())
        s2, p2 = dm_test(err**2, e_m**2, horizon, eval_step)
        row["DM_vs_linmom_t"], row["DM_vs_linmom_p"] = s2, p2
    # direction: model sign vs realised sign; momentum benchmark = sign(r26)
    row["dir_acc"] = float((np.sign(df["pred"]) == np.sign(df["y"])).mean())
    return row


def score_by_period(pred: pd.DataFrame, base_drift: pd.Series, base_rw: pd.Series) -> pd.DataFrame:
    rows = []
    df = pred.dropna(subset=["y", "pred"])
    dates = df.index.get_level_values("date")
    for label, (a, b) in PERIODS.items():
        m = (dates >= a) & (dates <= b)
        if m.sum() == 0:
            continue
        sub = df[m]
        err = sub["pred"] - sub["y"]
        e_dr = base_drift.reindex(sub.index) - sub["y"]
        e_rw = base_rw.reindex(sub.index) - sub["y"]
        rows.append({"period": label, "n": len(sub), "MAE_pp": float(err.abs().mean() * 100), "skill_vs_RW": float(1 - (err**2).mean() / (e_rw**2).mean()), "skill_vs_drift26": float(1 - (err**2).mean() / (e_dr**2).mean())})
    return pd.DataFrame(rows)
