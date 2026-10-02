"""Pooled-panel regressors used by the direct multi-horizon forecaster."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# A model function fits on (X_train, y_train) and returns predictions for X_pred.
ModelFn = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


def _prepare(X_train: pd.DataFrame, X_pred: pd.DataFrame, clip: float = 5.0):
    """Median-impute with train medians, standardize with train stats, winsorize to +/-clip sigma (numpy fast path)."""
    a = X_train.to_numpy(dtype=np.float64, copy=True)
    b = X_pred.to_numpy(dtype=np.float64, copy=True)
    med = np.nanmedian(a, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for arr in (a, b):
        rows, cols = np.where(np.isnan(arr))
        arr[rows, cols] = med[cols]
    mu = a.mean(axis=0)
    sd = a.std(axis=0, ddof=1)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    a = np.clip((a - mu) / sd, -clip, clip)
    b = np.clip((b - mu) / sd, -clip, clip)
    return np.nan_to_num(a), np.nan_to_num(b)


def ridge_model(alpha: float = 300.0) -> ModelFn:
    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        a, b = _prepare(X_train, X_pred)
        y_mu = float(np.mean(y_train))
        model = Ridge(alpha=alpha).fit(a, y_train - y_mu)
        return model.predict(b) + y_mu

    return fit_predict


HGB_DEFAULTS = dict(max_iter=200, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=200, l2=5.0, seed=0, row_stride=2)


def _hgb(max_iter: int, learning_rate: float, max_leaf_nodes: int, min_samples_leaf: int, l2: float, seed: int, early_stopping) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2, random_state=seed, early_stopping=early_stopping,
    )


def hgb_model(
    max_iter: int = 200,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 8,
    min_samples_leaf: int = 200,
    l2: float = 5.0,
    seed: int = 0,
    row_stride: int = 2,
    early_stopping: bool | str = "auto",
    record: list | None = None,
) -> ModelFn:
    """HistGradientBoosting on every `row_stride`-th training row.

    `early_stopping="auto"` is scikit-learn's default and what production has always used: with more than 10,000 training rows it holds out a
    RANDOM 10% of the rows (not the latest dates, not whole origins), so the iteration count is chosen on a validation set that is a
    time-mixed sample of the training window. That is a statement about how the count is chosen, not evidence of leakage into an outside
    evaluation. `record` (a list) receives {"n_train", "n_iter", "max_iter", "early_stopping_active"} per fit so real runs can report it.
    """

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        idx = np.arange(0, len(X_train), row_stride)
        # a column that is entirely NaN / constant in this training window (e.g. a survey that starts later) breaks
        # HGB's binning; such a column carries no information here, so leave it out for this fit
        usable = [c for c in X_train.columns if X_train[c].iloc[idx].nunique(dropna=True) >= 2]
        if not usable:
            return np.full(len(X_pred), float(np.mean(y_train)))
        X_train, X_pred = X_train[usable], X_pred[usable]
        model = _hgb(max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2, seed, early_stopping)
        model.fit(X_train.iloc[idx].to_numpy(dtype=np.float32), y_train[idx])
        if record is not None:
            record.append({"n_train": int(len(idx)), "n_iter": int(model.n_iter_), "max_iter": int(max_iter), "early_stopping_active": bool(getattr(model, "do_early_stopping_", False))})
        return model.predict(X_pred.to_numpy(dtype=np.float32))

    return fit_predict


def time_split_masks(dates: pd.DatetimeIndex, purge_weeks: int, val_share: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """(fit rows, validation rows): the newest `val_share` of the distinct dates validate; fit rows end `purge_weeks` before they start,
    so no fit label window reaches into the validation block."""
    uniq = np.sort(pd.DatetimeIndex(dates).unique())
    cut = uniq[int(len(uniq) * (1.0 - val_share))]
    d = pd.DatetimeIndex(dates)
    return np.asarray(d < cut - pd.Timedelta(weeks=purge_weeks)), np.asarray(d >= cut)


def hgb_model_timeval(
    purge_weeks: int,
    max_iter: int = 400,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 8,
    min_samples_leaf: int = 200,
    l2: float = 5.0,
    seed: int = 0,
    row_stride: int = 2,
    val_share: float = 0.2,
    min_iter: int = 20,
    record: list | None = None,
) -> ModelFn:
    """HGB whose number of iterations is chosen on a purged, time-ordered validation block inside the training window.

    The newest `val_share` of the training dates is the validation block; training rows whose label window (`purge_weeks` = the
    horizon) reaches into it are dropped. The model is grown to `max_iter` without early stopping, the iteration with the lowest
    validation MSE is kept (at least `min_iter`), and the final model is refit on all rows with that many iterations.
    """

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        usable = [c for c in X_train.columns if X_train[c].iloc[::row_stride].nunique(dropna=True) >= 2]
        if not usable:
            return np.full(len(X_pred), float(np.mean(y_train)))
        Xt, Xp = X_train[usable], X_pred[usable]
        fit_rows, val = time_split_masks(Xt.index.get_level_values("date"), purge_weeks, val_share)
        best = max_iter
        if fit_rows.sum() >= 2000 and val.sum() >= 500:
            tr_idx = np.flatnonzero(fit_rows)[::row_stride]
            va_idx = np.flatnonzero(val)[::row_stride]
            probe = _hgb(max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2, seed, False)
            probe.fit(Xt.iloc[tr_idx].to_numpy(dtype=np.float32), y_train[tr_idx])
            yv = y_train[va_idx]
            errs = [float(np.mean((p - yv) ** 2)) for p in probe.staged_predict(Xt.iloc[va_idx].to_numpy(dtype=np.float32))]
            best = int(max(min_iter, int(np.argmin(errs)) + 1))
        idx = np.arange(0, len(Xt), row_stride)
        final = _hgb(best, learning_rate, max_leaf_nodes, min_samples_leaf, l2, seed, False)
        final.fit(Xt.iloc[idx].to_numpy(dtype=np.float32), y_train[idx])
        if record is not None:
            record.append({"n_train": int(len(idx)), "n_iter": int(best), "max_iter": int(max_iter), "early_stopping_active": False, "selected_on": "purged time-ordered block"})
        return final.predict(Xp.to_numpy(dtype=np.float32))

    return fit_predict


def blend(models: list[ModelFn], weights: list[float] | None = None) -> ModelFn:
    w = np.asarray(weights if weights is not None else [1.0] * len(models), dtype=float)
    w = w / w.sum()

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        return sum(wi * m(X_train, y_train, X_pred) for wi, m in zip(w, models))

    return fit_predict
